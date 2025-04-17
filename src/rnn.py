import torch
import torch.nn as nn
from torch import Tensor


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Standard RNN architecture components.
         - i2h: input-to-hidden (for current input)
         - h2h: hidden-to-hidden (for previous state's contribution)
         - h2o: final projection (only used on the final live state)
        """
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, input: Tensor, hidden_state: Tensor) -> tuple[Tensor, Tensor]:
        h_current = torch.tanh(self.i2h(input) + self.h2h(hidden_state))
        output = self.h2o(h_current)

        return output, h_current

    def forward_sequence(self, inputs: Tensor) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """
        Run recursively the rnn on an input sequence. Disconnects the backward graph at each
        recursion so that we can control the flow.
        :param inputs: input sequence
        :return: the final output and a backward graph as a list of tuple of consecutive hidden
            states of the form [(h0, h1), (h1, h2), (h2, h3), ..., (h{n-1}, hn)] where n is the
            length of the sequence. the couple (hi, h{i+1}) here is connected through a backward
            graph, however hi in (h{i-1}, hi) and (hi, h{i+1}) are disconnected to stop the
            backpropagation.
        """
        seq_length, batch_size, _ = inputs.shape

        h = torch.zeros(batch_size, self.hidden_size, device=inputs.device)

        graph = []
        for input in inputs:
            output, h_next = self.forward(input, h)
            graph.append((h, h_next))
            h = h_next.detach().requires_grad_()

        return output, graph


def backpropagate_hidden_states(
        loss: Tensor,
        graph: list[tuple[Tensor, Tensor]],
    ) -> tuple[list[Tensor], list[Tensor]]:
    """
    From a loss and a computation graph returns the gradients of the hidden states as computed by
    the backpropagation algorithm.
    :param loss: the loss to differentiate
    :param graph: the graph of differentiation for the hidden states.
    :return: two list corresponding to hidden_states and the gradients of the loss with respect to
        them
    """
    last_h = graph[-1][1]
    h_grad = torch.autograd.grad(loss, last_h, retain_graph=True)[0]
    hidden_states = []
    hidden_state_grads = []

    for h, h_next in reversed(graph[1:]):
        hidden_states.append(h_next)
        hidden_state_grads.append(h_grad)
        h_grad = torch.autograd.grad(h_next, h, h_grad, retain_graph=True)[0]

    return hidden_states, hidden_state_grads
