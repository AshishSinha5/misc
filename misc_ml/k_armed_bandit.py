import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
class env:
    def __init__(self, k=10):
        self.k = k 
        self.create_env()

    def create_env(self):
        """
        creates the env with k arms 
        each arm dispenses rewards from a stationary normal distribution
        """
        means = np.random.uniform(size = self.k)*2
        std = [0]*self.k
        self.means = means 
        self.std = std

    def play(self, action):
        assert 0 <= action < self.k
        return np.random.normal(self.means[action], 0)

class agent:
    def __init__(self, eps = 0.1, eps_decay = 0.99999, k = 10):
        self.eps = eps 
        self.eps_decay = eps_decay
        self.k = k
        self.q = defaultdict(float)
        self.iter = defaultdict(float)

    def choose_action(self):
        if np.random.random() < self.eps:
            return np.random.randint(0, self.k)
        max_q = max([self.q[a] for a in range(self.k)]) 
        max_action = [a for a in range(self.k) if self.q[a] == max_q]
        return np.random.choice(max_action, 1)[0]

    def learn(self, action, reward):
        self.iter[action] += 1
        self.q[action] = self.q[action] + (reward - self.q[action])/self.iter[action]


def simulate(args):
    eps = args.eps
    eps_decay = args.eps_decay
    k = args.k
    num_ep = args.num_ep
    cummulative_eps_reward = defaultdict(list)
    for eps in [0.1, 0.01, 0]:
        env_ = env(k)
        agent_ = agent(eps, eps_decay, k)
        average_reward = 0
        cummulative_rewards = []
        for ep in range(num_ep):
            action = agent_.choose_action()
            reward = env_.play(action)
            average_reward += reward
            cummulative_rewards.append(average_reward/(ep + 1))
            agent_.learn(action, reward)
        cummulative_eps_reward[eps] = cummulative_rewards
    return cummulative_eps_reward

def main(args):
    average_rewards_eps = {} 
    for sample in tqdm(range(2000)):
        cummulative_eps_reward = simulate(args)
        for k, v in cummulative_eps_reward.items():
            if k not in average_rewards_eps:
                average_rewards_eps[k] = [0]*len(v)
            average_rewards_eps[k] = [r1 + r2 for r1, r2 in zip(average_rewards_eps[k], v)]
    for k,v in average_rewards_eps.items():
        average_rewards_eps[k] = [v_/2000 for v_ in v] 
        plt.plot(average_rewards_eps[k], label = k)
    plt.legend()
    plt.xlabel("steps")
    plt.ylabel("average rewards")
    plt.savefig("plots/ave_rewards.png")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ep", type = int, default=1000, required=False)
    parser.add_argument("--eps", type=float, default=0.1, required=False)
    parser.add_argument("--k", type=int, default=10, required=False)
    parser.add_argument("--eps_decay", type=float, default=0.9999, required=False)
    args = parser.parse_args()
    main(args)


















