import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
from time import time
batch_size = 10 
input_size = 5
hidden_size = 6
input_size = 3
num_classes = 4
learning_rate = 0.001
torch.manual_seed(100)

input = autograd.Variable(torch.rand(batch_size, input_size))-0.5
target = autograd.Variable((torch.rand(batch_size)*num_classes).long())
print ('===> input  : ',input)
print ('===> target : ',target)


class Net(nn.Module):
    def __init__(self, input_size, hidden_size,num_classes):
        super().__init__()
        self.h1 = nn.Linear(input_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, num_classes)
    def forward(self,x):
        x = self.h1(x)
        x = F.tanh(x)
        x = self.h2(x)
        x = F.softmax(x)
        return x

model = Net(input_size = input_size, hidden_size=hidden_size, num_classes = num_classes)
opt = optim.Adam(params=model.parameters(), lr=learning_rate)

for i in range(1,5000):
    print('### Epoch ### ',i)
    start = time()
    output = model(input)
    _, pred = output.max(1)
    loss = F.nll_loss(output,target)
    print('===> target     : ',target) 
    print('===> prediction : ',pred)
    print('===> loss       : ',loss.view(1,-1))
    print('===> evaluation time: ',time() - start)
    # optiminaztion
    model.zero_grad()
    loss.backward()
    opt.step()
#mode.zero_grad() # put grad to zero
