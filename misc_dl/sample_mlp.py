# write a dummy MLP with one hidden layer, no non-linearity, and a softmax output layer
# with driver function 
import torch
import logging
torch._logging.set_logs(graph_breaks=True, dynamic=logging.DEBUG)


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)  # No non-linearity
        x = self.fc2(x)
        return torch.nn.functional.softmax(x, dim=1)
    

if __name__ == "__main__":
    input_size = 10
    hidden_size = 5
    output_size = 3
    model = MLP(input_size, hidden_size, output_size)

    # Create a dummy input tensor
    dummy_input = torch.randn(1, input_size)

    # Forward pass through the model
    output = model(dummy_input)
    print("Output:", output)
