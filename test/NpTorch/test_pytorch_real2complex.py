# this is to test the pytorch ability when real to complex
import torch
from torch import nn

# define the model
class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        self.linear1 = torch.nn.Linear(4, 10)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(10, 1)
        self.softmax = torch.nn.Softmax()
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
model = TinyModel();

# define the loss and optimizer
criterion = nn.MSELoss();
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5);

# define the data for training
batch_size = 64;
features = torch.randn(batch_size, 4);
labels = torch.randn(batch_size, 1);

