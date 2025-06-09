from IPython import get_ipython
get_ipython().magic('reset -f')
get_ipython().magic('clear')

import torch
import torch.nn as nn
import torch.optim as optim

x = torch.randn(128, 20)
y_true = torch.randn(128, 30)


model = nn.Linear(20, 30)

# 损失函数
criterion = nn.MSELoss()
# 优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)


# 训练一步示范
optimizer.zero_grad()           # 梯度清零
y_pred = model(x.detach())               # 前向计算
loss = criterion(y_pred, y_true)  # 计算损失
loss.backward()                 # 反向传播
optimizer.step()                # 参数更新


print("- grad")
print(f"  - x: {x.requires_grad}")
print(f"  - y_true: {y_true.requires_grad}")
print(f"  - y_pred: {y_pred.requires_grad}")


print(f"损失: {loss.item()}")