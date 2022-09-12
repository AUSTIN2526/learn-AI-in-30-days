import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from cv2 import imread
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
#拿範例1的資料作使用

class CRFAR10(Dataset):
    def __init__(self, path, transform):
        self.data = []
        for label in os.listdir(path):
            for pic in os.listdir(path + '/' + label):
                cv_pic = imread(f'{path}/{label}/{pic}')
                self.data.append([cv_pic, int(label)])
    
    def __getitem__(self,index):
        datas = transform(self.data[index][0])
        labels = torch.tensor(self.data[index][1])
        return datas, labels
    
    def __len__(self):
        return len(self.data)
        
        
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    #(輸入 + 2*(padding) - 捲積核) / 移動 + 1
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(train_loader,test_loader, model ,optimizer, criterion):
    epochs = 10
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        train = tqdm(train_loader)
      
        model.train()
        for cnt,(data,label) in enumerate(train, 1):
            data,label = data.cuda() ,label.cuda()
            outputs = model(data)
            loss = criterion(outputs, label)
            _,predict_label = torch.max(outputs, 1)
            
            optimizer.zero_grad()           
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (predict_label==label).sum()
            train.set_description(f'train Epoch {epoch}')
            train.set_postfix({'loss':float(train_loss)/cnt,'acc': float(train_acc)/cnt})
            
        model.eval()
        test = tqdm(test_loader)
        test_acc = 0
        for cnt,(data,label) in enumerate(test, 1):
            data,label = data.cuda() ,label.cuda()
            outputs = model(data)
            _,predict_label = torch.max(outputs, 1)
            test_acc += (predict_label==label).sum()
            test.set_description(f'test Epoch {epoch}')
            test.set_postfix({'acc': float(test_acc)/cnt})



           


    
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                               ])
                               
train_set = CRFAR10(r'pic/train/', transform)
test_set = CRFAR10(r'pic/test/', transform)
train_loader = DataLoader(train_set, batch_size = 128,shuffle = True, num_workers = 0)
test_loader = DataLoader(test_set, batch_size = 128, shuffle = True, num_workers = 0)

model = CNN().cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
train(train_loader, test_loader, model,optimizer,criterion)