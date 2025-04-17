import torch
import torch.nn as nn
from torch.optim import SGD

from torchjd.aggregation import UPGrad, Mean
from torchjd import backward

from rnn import RNN, backpropagate_hidden_states

def generate_sample(seq_length, batch_size, input_size, hidden_size):
    inputs = torch.randint(0, 1, [seq_length, batch_size, input_size]).float()
    target = (inputs[:-hidden_size].sum() % 2).float()
    return inputs, target


if __name__ == '__main__':
    # Settings.
    seq_length = 9
    batch_size = 1
    input_size = 1
    hidden_size = 4
    output_size = 1
    train_iterations = 100
    test_iterations = 100

    model = RNN(input_size, hidden_size, output_size)
    loss_fn = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)
    aggregator = UPGrad()

    # Train.
    for i in range(train_iterations):
        # Create random input and target data.
        inputs, target = generate_sample(seq_length, batch_size, input_size, hidden_size)

        # Forward pass, keep the backward graph
        output, graph = model.forward_sequence(inputs)
        loss = loss_fn(output, target)
        print(f"Train {i} loss: {loss}")

        # Manual backpropagation
        hidden_states, hidden_state_grads = backpropagate_hidden_states(loss, graph)

        # This enables to set the correct hidden_state_grad for backpropagation
        backpropagation_scalars = [
            torch.dot(h.reshape([-1]), h_grad.reshape([-1]))
            for h, h_grad in zip(hidden_states, hidden_state_grads)
        ]

        optimizer.zero_grad()
        backward([loss] + backpropagation_scalars, aggregator, model.parameters())
        optimizer.step()

    loss_sum = 0.0

    for i in range(test_iterations):
        inputs, target = generate_sample(seq_length, batch_size, input_size, hidden_size)

        # Forward pass, keep the backward graph
        output, graph = model.forward_sequence(inputs)
        loss = loss_fn(output, target)
        loss_sum = loss_sum + loss
        print(f"Test {i} loss: {loss}")

    print(f"Test average loss: {loss_sum/test_iterations}")