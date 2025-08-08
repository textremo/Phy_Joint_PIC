from IPython import get_ipython
get_ipython().magic('reset -f')
get_ipython().magic('clear')

# this is to test the pytorch ability when real to complex
import torch
from torch import nn
import torch.optim as optim

torch.set_default_dtype(torch.float64)

# define the data for training
batch_size = 64;
N0 = 4
N1 = 8
N2 = 20
N3 = 10
N4 = 2

H = torch.rand(1, N0).repeat(batch_size, 1, 1) + 1j*torch.rand(1, 4).repeat(batch_size, 1, 1)

# define the model
class TinyModel0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(N1, N2)
        self.linear2 = torch.nn.Linear(N3, N4)
        
        self.register_buffer("H_inner", torch.rand(int(N3/2), N3).repeat(batch_size, 1, 1) + 1j*torch.rand(int(N3/2), N3).repeat(batch_size, 1, 1))
        
        
    def forward(self, x):
        # to real
        x = torch.cat([x.real, x.imag], -1)
        x = self.linear1(x)
        # to complex
        x = torch.complex(x[..., :N3], x[..., N3:]).unsqueeze(-1)
        x = (self.H_inner @ x).squeeze(-1)
        # to real
        x = torch.cat([x.real, x.imag], -1)
        
        x = self.linear2(x)
        
        
        return x
    
    
model0 = TinyModel0();

# define the loss and optimizer
criterion = nn.MSELoss();
optimizer0 = optim.Adam(model0.parameters(), lr=1e-3, weight_decay=1e-5);


# train
model0.train()
for i in range(5000):
    features = torch.randn(batch_size, 4) + 1j*torch.randn(batch_size, 4);
    labels = (H @ features.unsqueeze(-1)).squeeze(-1)
    labels_r = torch.cat([labels.real, labels.imag], -1)
    
    out0 = model0(features)
    loss0 = criterion(out0, labels_r)
    optimizer0.zero_grad()
    loss0.backward()
    optimizer0.step()
    if i % 500 == 0:
        print(f"  - Iter: {i:03d}, loss: {loss0.item():.4f}")
        



