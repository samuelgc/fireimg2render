import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class ConvNet(nn.Module):
    def __init__(self,num_out=10):
        super(ConvNet,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=5,stride = 2 , padding = 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=5, stride = 2, padding = 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=5, stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(8*8*64, num_out)
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # print out.shape
        out = out.reshape(out.size(0),-1)
        out = self.fc(out)
        out = out[0]
        return out
class MyData(Dataset):
    def __init__(self,pathToDir):
        self.imgName = []
        self.imgLab = []
        self.pathToDir = pathToDir
        self.datalen = len(os.listdir(pathToDir))
        for filename in os.listdir(pathToDir):
            num = filename.find("_")
            num2 = filename.find(".")
            numReal = filename[num+1:num2]
            self.imgName += [filename]
            self.imgLab += [float(numReal)]
        #Normalize Data
        self.imgLab = np.asarray(self.imgLab)
        maxVal = np.max(self.imgLab)
        minVal = np.min(self.imgLab)
        print "Max[{}] Min[{}]".format(maxVal,minVal)
        self.imgLab = (self.imgLab - minVal)/ (maxVal - minVal)

        self.transTen = transforms.Compose([
            transforms.CenterCrop(512),
            transforms.ToTensor()
        ]) 

    def __getitem__(self,index):
        single_img_name = self.imgName[index]
        # print single_img_name
        single_img = Image.open(self.pathToDir + single_img_name)
        img_tensor = self.transTen(single_img)
        return (img_tensor,torch.tensor(float(self.imgLab[index])))

    def __len__(self):
        return self.datalen
num_epochs = 100
num_classes = 10
batch_size = 1
learning_rate = 0.001

model = ConvNet(1)
#LossFunc = nn.MSELoss()
LossFunc = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_steps = 100
loadData = MyData("../fireTemp/")
# testData = MyData("../fire_images/google/fire/")
train_loader = torch.utils.data.DataLoader(dataset=loadData,
                                            batch_size= batch_size,
                                            shuffle= True)
# test_loader = torch.utils.data.DataLoader(dataset=testData,
#                                             batch_size= batch_size,
#                                             shuffle= False)
x = []
y = []
total_step = len(train_loader)
# print loadData[0][0]
# imgTemp = torch.Tensor.numpy(loadData[0][0])
# imgTemp *= 255
# imgTemp = np.uint8(imgTemp)
# imgTemp = np.transpose(imgTemp, (1, 2, 0))
# print loadData[0][1].item()  * 24900
# img = Image.fromarray(imgTemp)
# lol= plt.imshow(imgTemp)
# plt.show()
# bob = input()
# print imgTemp
for epoch in range(num_epochs):
    # for i, (images, labels) in enumerate(train_loader):
    for i, (images, labels) in enumerate(train_loader):
        # images = images.to(device)
        # labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        # print outputs , labels
        loss = LossFunc(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 50 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.10f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            y += [loss.item()]
plt.plot(np.asarray(y))
plt.show()
model.eval()
# for img in os.listdir(file_dir)
totLoss = 0
b = 0 
for i , (images,labels) in enumerate(train_loader):
    b += 1
    outputs = model(images)
    out2 = round(outputs[0] *9900 / 100.0) * 100
    out3 = torch.tensor(out2/9900)
    lab2 = round(labels[0] *9900 / 100.0) * 100
    if((i+1) % 10 == 0):
        print "Out[{}] -- Expected[{}]".format(out2,lab2)
    loss = LossFunc(out3,labels)
    totLoss += loss

print "B = " , b
print "Mean L1 = " ,totLoss/b
print "I = " , i

# outputs = model()


plt.plot(np.asarray(y),'b')
plt.show()
# bob = input()
