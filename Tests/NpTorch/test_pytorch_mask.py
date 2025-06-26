from IPython import get_ipython
get_ipython().magic('reset -f')
get_ipython().magic('clear')

# this is to test the pytorch ability when real to complex
import torch
from torch import nn
import torch.optim as optim

batch_size = 64;

# define the model
class TinyModel0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 10)
        self.linear2 = torch.nn.Linear(12, 1)
        
        self.register_buffer("a", torch.zeros(batch_size, 12))
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.a.masked_scatter(mask0, x)
        x = self.linear2(x)
        return x
    
class TinyModel1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 10)
        #self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(10, 1)
        #self.softmax = torch.nn.Softmax()
        self.register_buffer("a", torch.zeros(batch_size, 10))
        
    def forward(self, x):
        x = self.linear1(x)
        #x = self.activation(x)
        
        
        x = self.a.masked_scatter(mask1, x)
        
        x = self.linear2(x)
        #x = self.softmax(x)
        return x
    

model0 = TinyModel0();
model1 = TinyModel1();

# define the loss and optimizer
criterion = nn.MSELoss();
optimizer0 = optim.Adam(model0.parameters(), lr=1e-3, weight_decay=1e-5);
optimizer1 = optim.Adam(model1.parameters(), lr=1e-3, weight_decay=1e-5);

# define the data for training

mask0 = torch.as_tensor([True, True, False, False, True, True, True, True, True, True, True, True]).repeat(batch_size, 1) 
mask1 = torch.ones(batch_size, 10, dtype=torch.bool)

H = torch.rand(1, 4).repeat(batch_size, 1, 1)

# model 0
print("- model 0")
model0.train()
for i in range(5000):
    features = torch.randn(batch_size, 4);
    labels = (H @ features.unsqueeze(-1)).squeeze(-1)
    out0 = model0(features)
    loss0 = criterion(out0, labels)
    optimizer0.zero_grad()
    loss0.backward()
    optimizer0.step()
    if i % 500 == 0:
        print(f"  - Iter: {i:03d}, loss: {loss0.item():.4f}")
        
print("- model 1")
model1.train()
for i in range(5000):
    features = torch.randn(batch_size, 4);
    labels = (H @ features.unsqueeze(-1)).squeeze(-1)
    out1 = model1(features)
    loss1 = criterion(out1, labels)
    optimizer1.zero_grad()
    loss1.backward()
    optimizer1.step()
    if i % 500 == 0:
        print(f"  - Iter: {i:03d}, loss: {loss1.item():.4f}")