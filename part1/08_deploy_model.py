import torch

class Model(torch.nn.Module):
    def forward(self, x):
        # This wonderful model gives you
        # back 10 times the goodness
        return 10 * x

model = Model()
# Put model in eval mode
model.eval()

# Use a dummy input for JIT trace
x = torch.ones(1, 1)
trace = torch.jit.trace(model, x)

# Save model
torch.jit.save(trace, 'model.zip')
