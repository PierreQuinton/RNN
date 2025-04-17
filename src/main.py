import torch
import torch.nn as nn
import torch.nn.functional as F


# Define a simple RNN using truncated backpropagation through time.
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Standard RNN architecture components.
         - i2h: input-to-hidden (for current input)
         - h2h: hidden-to-hidden (for previous state's contribution)
         - h2o: final projection (only used on the final live state)
        """
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward_truncated(self, inputs):
        """
        Computes the forward pass with truncated gradients.

        For each time step, the next step's hidden state is computed
        from a detached version of the current hidden state.

        Returns:
            output: the final output (computed from the final live state)
            h_live: a list of non-detached hidden states (used in the loss).
            h_det:  a list of detached hidden states (leaf nodes with requires_grad)
                    that were used as inputs for each recurrence.
                    h_det[t] is the input to time step t+1.
        """
        seq_length, batch_size, _ = inputs.shape

        # Initialize the hidden state (live version). We treat this as the starting state.
        h0 = torch.zeros(batch_size, self.hidden_size, device=inputs.device, requires_grad=True)
        # For truncated BPTT, we also prepare a detached version of h0 that still requires grad.
        h_prev_det = h0.detach().clone().requires_grad_()

        h_live = []  # will store the "live" hidden states (with full computation history)
        h_det = []  # will store the detached hidden states (used as inputs at each recurrence)

        for t in range(seq_length):
            # Compute the current live hidden state from the current input and the detached previous state.
            # Note: The detached h_prev_det ensures that the computational graph for h_live[t]
            # only involves parameters for the current time step.
            h_current = torch.tanh(self.i2h(inputs[t]) + self.h2h(h_prev_det))
            h_live.append(h_current)

            # Prepare h_current for the next recurrence by detaching but re-enabling gradient tracking.
            h_prev_det = h_current.detach().clone().requires_grad_()
            h_det.append(h_prev_det)

        # Compute the output from the final live hidden state.
        output = self.h2o(h_live[-1])
        return output, h_live, h_det


def compute_per_step_gradients(model, inputs, target, loss_fn):
    """
    Perform the forward pass using truncated backpropagation, then manually
    propagate gradients through time one step at a time to compute the contribution
    of each recurrence (time step) to the parameters.

    For each time step t, we compute:
         dL/dθ   (local) = (dL/dh_live[t])  * (dh_live[t]/dθ)
    and also backpropagate:
         dL/dh_det[t-1] = (dL/dh_live[t]) * (dh_live[t]/dh_det[t-1])

    Returns:
         loss: The computed loss.
         per_step_grads: A dictionary mapping each parameter name to a tensor of shape
                         (seq_length, numel(parameter)) collecting each time step's contribution.
    """
    # Run the truncated forward pass.
    output, h_live, h_det = model.forward_truncated(inputs)
    loss = loss_fn(output, target)

    # Container for per-step gradient contributions.
    per_step_grads = {name: [] for name, param in model.named_parameters() if param.requires_grad}

    # Begin by computing the gradient of the loss with respect to the final live hidden state.
    # This is the starting gradient to be propagated backward in time.
    grad_h = torch.autograd.grad(loss, h_live[-1], retain_graph=True, allow_unused=True)[0]

    # Propagate gradients back through time (from last step to first).
    # At each time step t, we compute the local gradient for each parameter.
    seq_length = len(h_live)
    for t in reversed(range(seq_length)):
        # For every parameter, compute the gradient from the current live state.
        for name, param in model.named_parameters():
            if param.requires_grad:
                # grad_param corresponds to d(h_live[t])/d(param) multiplied by the current grad_h.
                grad_param = torch.autograd.grad(
                    h_live[t],
                    param,
                    grad_outputs=grad_h,
                    retain_graph=True,
                    allow_unused=True
                )[0]
                if grad_param is not None:
                    per_step_grads[name].append(grad_param.detach().clone().view(-1))

        # If not at the first step, propagate grad_h backward from h_live[t] to the detached hidden state
        # that was used as input at this time step (i.e. from time t-1).
        if t > 0:
            # h_det[t-1] is the detached input to the t-th recurrence.
            grad_h = torch.autograd.grad(
                h_live[t],
                h_det[t - 1],
                grad_outputs=grad_h,
                retain_graph=True,
                allow_unused=True
            )[0]

    outputs = {}

    # After the backward loop, the contributions have been collected in reversed order.
    # We reverse each list so that row 0 corresponds to time step 0, and so on.
    for name in per_step_grads:
        if len(per_step_grads[name]) != 0:
            per_step_grads[name].reverse()  # now index 0 is the earliest time step
            outputs[name] = torch.stack(per_step_grads[name])

    return loss, outputs


from torchjd.aggregation import UPGrad, Mean

# Example usage:
if __name__ == '__main__':
    # Settings.
    seq_length = 5
    batch_size = 1
    input_size = 3
    hidden_size = 4
    output_size = 2

    # Instantiate the model.
    model = SimpleRNN(input_size, hidden_size, output_size)

    # Create random input and target data.
    inputs = torch.randn(seq_length, batch_size, input_size)
    target = torch.randn(batch_size, output_size)

    # Define a loss function.
    loss_fn = nn.MSELoss()

    # Compute the loss and per-step gradient contributions.
    loss, per_step_grads = compute_per_step_gradients(model, inputs, target, loss_fn)

    print("Loss:", loss.item())
    # For each parameter, print out the shape of the per-step gradient contributions.
    for name, grad_matrix in per_step_grads.items():
        print(f"Parameter: {name}, Per-step gramians: \n{grad_matrix @ grad_matrix.T}")
        # For example, you might inspect inner products between steps:
        # conflict_matrix = torch.mm(grad_matrix, grad_matrix.t())
        # print
