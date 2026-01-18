#!/usr/bin/env python3
"""
lora_attention_geometry_demo.py

End-to-end demonstration of how LoRA changes attention geometry.
- Tiny self-attention
- LoRA applied to Query projection only
- Base weights frozen; only LoRA is trained
- Forces a target attention distribution
- Visualizes attention maps and prints geometry metrics

Run:
  python lora_attention_geometry_demo.py

Optional:
  python lora_attention_geometry_demo.py --steps 400 --rank 4 --seed 0
"""

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    KL(p || q) along last dim. p,q are probability distributions.
    """
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def plot_heatmap(mat: torch.Tensor, title: str) -> None:
    """
    mat: (n, n) on CPU
    """
    plt.figure(figsize=(5.2, 4.2))
    plt.imshow(mat, aspect="auto")
    plt.title(title)
    plt.xlabel("Key position")
    plt.ylabel("Query position")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


# -----------------------------
# LoRA Linear
# -----------------------------
class LoRALinear(nn.Module):
    """
    LoRA wrapper for a bias-free Linear:
      y = xW^T + scale * x(BA)^T
    where:
      A: (r x in_features)
      B: (out_features x r)
      scale = alpha/r

    Base layer weights frozen; only A,B trained.
    """
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        if base.bias is not None:
            raise ValueError("Use bias=False for clean LoRA geometry demo.")
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = base.in_features
        out_features = base.out_features

        # Freeze base
        for p in self.base.parameters():
            p.requires_grad = False

        # LoRA params
        self.A = nn.Parameter(torch.zeros(r, in_features))         # (r, d_in)
        self.B = nn.Parameter(torch.zeros(out_features, r))        # (d_out, r)

        # Init: A small random, B zeros => ΔW starts at 0
        nn.init.normal_(self.A, mean=0.0, std=0.01)
        nn.init.zeros_(self.B)

    def delta_weight(self) -> torch.Tensor:
        """
        Returns ΔW = B A (out_features x in_features)
        """
        return self.B @ self.A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)  # (..., d_out)
        lora_out = (self.dropout(x) @ self.A.t()) @ self.B.t()  # (..., d_out)
        return base_out + self.scale * lora_out


# -----------------------------
# Tiny Attention Block
# -----------------------------
@dataclass
class AttentionCache:
    Q: torch.Tensor
    K: torch.Tensor
    V: torch.Tensor
    scores: torch.Tensor
    attn: torch.Tensor
    deltaQ: torch.Tensor


class TinySelfAttention(nn.Module):
    """
    Single-head self-attention with explicit Q,K,V projections.
    Optionally wraps Q projection with LoRA.
    """
    def __init__(self, d_model: int, use_lora_on_q: bool, r: int, alpha: float, dropout: float):
        super().__init__()
        self.d = d_model

        # Base projections
        q_base = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        if use_lora_on_q:
            self.q_proj = LoRALinear(q_base, r=r, alpha=alpha, dropout=dropout)
        else:
            self.q_proj = q_base

        # Freeze all base weights except LoRA params (if present)
        # For non-LoRA run, everything is frozen anyway in the experiment below.
        for name, p in self.named_parameters():
            if "A" in name or "B" in name:  # LoRA trainable
                p.requires_grad = True
            else:
                p.requires_grad = False

    def forward(self, x: torch.Tensor, return_cache: bool = False):
        """
        x: (batch, n, d)
        """
        # Identify whether q_proj is LoRA-wrapped
        if isinstance(self.q_proj, LoRALinear):
            Q_base = self.q_proj.base(x)
            Q = self.q_proj(x)
            deltaQ = Q - Q_base
        else:
            Q = self.q_proj(x)
            deltaQ = torch.zeros_like(Q)

        K = self.k_proj(x)
        V = self.v_proj(x)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d)  # (b, n, n)
        attn = F.softmax(scores, dim=-1)
        out = attn @ V
        out = self.o_proj(out)

        if return_cache:
            cache = AttentionCache(Q=Q, K=K, V=V, scores=scores, attn=attn, deltaQ=deltaQ)
            return out, cache
        return out


# -----------------------------
# Main experiment
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_tokens", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--target_query", type=int, default=0)
    parser.add_argument("--target_key", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Create a fixed input sequence X
    # We keep X fixed to isolate how LoRA changes the geometry of attention.
    X = torch.randn(1, args.n_tokens, args.d_model, device=device)

    # Build attention module with LoRA on Q only
    model = TinySelfAttention(
        d_model=args.d_model,
        use_lora_on_q=True,
        r=args.rank,
        alpha=args.alpha,
        dropout=args.dropout
    ).to(device)

    # Convenience: get LoRA params
    lora_params = [p for p in model.parameters() if p.requires_grad]
    if len(lora_params) == 0:
        raise RuntimeError("No trainable LoRA params found.")

    opt = torch.optim.AdamW(lora_params, lr=args.lr)

    # Define a target attention distribution for a single query position.
    # Goal: force query `target_query` to attend heavily to `target_key`.
    # We'll use cross-entropy between attention row and target distribution.
    tq, tk = args.target_query, args.target_key
    target_dist = torch.zeros(args.n_tokens, device=device)
    target_dist[tk] = 1.0  # one-hot target (can also use smoothed target)

    # Record BEFORE training
    model.eval()
    with torch.no_grad():
        _, cache0 = model(X, return_cache=True)

    A0 = cache0.attn[0].detach().cpu()  # (n, n)
    S0 = cache0.scores[0].detach().cpu()
    Q0 = cache0.Q[0].detach().cpu()
    dQ0 = cache0.deltaQ[0].detach().cpu()

    # Train LoRA to reshape attention
    model.train()
    for step in range(args.steps):
        opt.zero_grad()
        _, cache = model(X, return_cache=True)

        # attention distribution for query tq: shape (n,)
        attn_row = cache.attn[0, tq, :]  # (n,)

        # Cross-entropy between target_dist and attn_row
        # CE(target, pred) = - sum target * log(pred)
        loss = -(target_dist * (attn_row.clamp_min(1e-9).log())).sum()

        # Small regularizer to prevent extreme score blow-up (optional)
        # This keeps the demonstration stable.
        loss = loss + 1e-4 * cache.scores.pow(2).mean()

        loss.backward()
        opt.step()

        if (step + 1) % max(1, args.steps // 5) == 0:
            with torch.no_grad():
                p_target = float(cache.attn[0, tq, tk].detach().cpu())
                print(f"step {step+1:4d}/{args.steps} | loss={float(loss):.4f} | A[{tq},{tk}]={p_target:.4f}")

    # Record AFTER training
    model.eval()
    with torch.no_grad():
        _, cache1 = model(X, return_cache=True)

    A1 = cache1.attn[0].detach().cpu()
    S1 = cache1.scores[0].detach().cpu()
    Q1 = cache1.Q[0].detach().cpu()
    dQ1 = cache1.deltaQ[0].detach().cpu()

    # -----------------------------
    # Geometry metrics
    # -----------------------------

    # 1) Query rotation (cosine similarity) per token
    # Compare Q_before vs Q_after
    cos_q = F.cosine_similarity(Q0, Q1, dim=-1)  # (n,)
    # Compare Q_base vs (Q_base + deltaQ_after) via deltaQ magnitude
    dq_norm = dQ1.norm(dim=-1)  # (n,)

    # 2) Score matrix change magnitude
    score_delta_fro = (S1 - S0).norm(p="fro") / (S0.norm(p="fro") + 1e-9)

    # 3) Attention distribution change for the target query: KL(A_before || A_after)
    kl_q = float(kl_divergence(A0[tq], A1[tq]).detach().cpu())

    print("\n=== Geometry Summary ===")
    print(f"Target: query={tq}, key={tk}")
    print(f"Before: A[{tq},{tk}] = {float(A0[tq, tk]):.4f}")
    print(f"After : A[{tq},{tk}] = {float(A1[tq, tk]):.4f}")
    print(f"Relative Frobenius change in scores ||S1-S0||_F / ||S0||_F = {float(score_delta_fro):.4f}")
    print(f"KL divergence for query row KL(A0[q] || A1[q]) = {kl_q:.4f}")
    print(f"Mean cosine similarity between Q0 and Q1 across tokens = {float(cos_q.mean()):.4f}")
    print(f"Mean ||ΔQ|| across tokens (after training) = {float(dq_norm.mean()):.4f}")

    # Print a few per-token values (for "directional" insight)
    print("\nPer-token query cosine similarity cos(Q0_i, Q1_i):")
    print(cos_q.numpy().round(3))

    # -----------------------------
    # Plots
    # -----------------------------
    plot_heatmap(A0, f"Attention map BEFORE LoRA training\nA[{tq},{tk}]={A0[tq, tk]:.3f}")
    plot_heatmap(A1, f"Attention map AFTER LoRA training\nA[{tq},{tk}]={A1[tq, tk]:.3f}")

    # Also plot score matrices to show geometry shift pre-softmax
    plot_heatmap(S0, "Score matrix BEFORE (QK^T / sqrt(d))")
    plot_heatmap(S1, "Score matrix AFTER  (QK^T / sqrt(d))")

    # Visualize per-token Q rotation and ΔQ magnitude
    plt.figure(figsize=(6, 3.5))
    plt.plot(cos_q.numpy(), marker="o")
    plt.ylim(0.0, 1.0)
    plt.title("Per-token query rotation: cosine similarity cos(Q_before, Q_after)")
    plt.xlabel("Token index")
    plt.ylabel("Cosine similarity")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 3.5))
    plt.plot(dq_norm.numpy(), marker="o")
    plt.title("Per-token ΔQ magnitude after LoRA training")
    plt.xlabel("Token index")
    plt.ylabel("||ΔQ||")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
