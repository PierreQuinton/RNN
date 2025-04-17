import torch
import torch.nn as nn

from torchjd.aggregation import UPGrad, Mean
from torchjd import backward

from rnn import RNN, backpropagate_hidden_states

if __name__ == '__main__':
    # Settings.
    seq_length = 5
    batch_size = 1
    input_size = 3
    hidden_size = 4
    output_size = 2

    # Instantiate the model.
    model = RNN(input_size, hidden_size, output_size)

    # Create random input and target data.
    inputs = torch.randn(seq_length, batch_size, input_size)
    target = torch.randn(batch_size, output_size)

    # Define a loss function.
    loss_fn = nn.MSELoss()

    # Forward pass, keep the backward graph
    output, graph = model.forward_sequence(inputs)
    loss = loss_fn(output, target)

    # Manual backpropagation
    hidden_states, hidden_state_grads = backpropagate_hidden_states(loss, graph)

    # This enables to set the correct hidden_state_grad for backpropagation
    backpropagation_scalars = [
        torch.dot(h.reshape([-1]), h_grad.reshape([-1]))
        for h, h_grad in zip(hidden_states, hidden_state_grads)
    ]

    aggregator = UPGrad()

    backward([loss] + backpropagation_scalars, aggregator, model.parameters())

    for name, param in model.named_parameters():
        print(name + f"of shape: {param.shape} gets\n{param.grad}")
