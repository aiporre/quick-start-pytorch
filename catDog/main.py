import cv2
import numpy as np
import os
import glob
from random import shuffle
from tqdm import tqdm
import os, fnmatch

import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
from time import time

#input = autograd.Variable(torch.rand(batch_size, input_size))-0.5
#target = autograd.Variable((torch.rand(batch_size)*num_classes).long())
#print ('===> input  : ',input)
#print ('===> target : ',target)



TRAIN_DIR = 'data_c_vs_d/PetImages/'
TEST_DIR = ''
IMG_SIZE = 50 
LR  = 1e-3
batch_size = 10
input_size = IMG_SIZE*IMG_SIZE
hidden_size = 6
input_size = 3
num_classes = 2
learning_rate = 0.001


MODEL_NAME = 'dogvscat-{}-{}-model'.format(LR,'2conv-basic')



def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


def get_jpg_files(path):
    for filename in find_files(path, '*.jpg'):
        label = filename.split('/')[2]
        yield label,filename

def create_train_data():
    train_data = []
    for label,path in tqdm(get_jpg_files(TRAIN_DIR)):
        original_img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        # print('----->',original_img)
        if original_img != None: 
            img = cv2.resize(original_img,(IMG_SIZE,IMG_SIZE))
            train_data.append([np.array(img), np.array(label)])
        # training_Data.append(np.array(img), np.arrat(label))
    shuffle(train_data)
    np.save('train_data.npy',train_data)
    return train_data
'''class Net(nn.Module):
    def __init__(self, input_size, hidden_size,num_classes):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)

        self.h1 = nn.Linear(input_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, num_classes)
    def forward(self,x):
        x = self.h1(x)
        x = F.tanh(x)
        x = self.h2(x)
        x = F.softmax(x)
        return x'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def to_num(label):
    if label=='dog':
        return 1
    else:
        return 0
def separate_train_data(train_data):
    train = train_data[:-500]
    test = train_data[-500:]
    h = np.array([i[0] for i in train])
    print('h  ===>', h.shape)
    X = h.reshape(-1,1,IMG_SIZE,IMG_SIZE)
    Y = np.array([to_num(i[1]) for i in train])
    test_X = np.array([i[0] for i in test]).reshape(-1,1,IMG_SIZE,IMG_SIZE)
    test_Y = np.array([to_num(i[1]) for i in test])
    return X, Y, test_X, test_Y
def torch_npvars(x):
    # cast fails 
    y = torch.tensor(x,dtype=torch.float32)
    print('y data yype:', y.dtype)
    z = autograd.Variable(y)
    print('z.dtype==',z.dtype)
    return y

train_data_files = [f for f in find_files('./','train_data.npy')]



# data prepatetion

if len(train_data_files)==0:
    train_data = create_train_data()
else:
    train_data = np.load('train_data.npy')

X, Y, test_X, test_Y = separate_train_data(train_data)

# defining model and optimizer
#model = Net(input_size = input_size, hidden_size=hidden_size, num_classes = num_classes)
model = Net()
opt = optim.Adam(params=model.parameters(), lr=learning_rate)

#training process
X = torch_npvars(X)
Y = torch_npvars(Y)
print(X.size())
g = torch.randn(1,400)
print('g type: =====> ',g.dtype)
output = model(X)
#print(output)
'''for i in range(10):
    print('### Epoch ### ',i)
    start = time()
    output = model(X)
    _, pred = output.max(1)
    loss = F.nll_loss(output,Y)
    print('===> target     : ',y[1:10])
    print('===> prediction : ',pred[1:10])
    print('===> loss       : ',loss.view(1,-1))
    print('===> evaluation time: ',time() - start)
    # optiminaztion
    model.zero_grad()
    opt.step()
'''
