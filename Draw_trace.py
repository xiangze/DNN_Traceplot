import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import torch.nn as nn
import torch.optim as optim
import getweights as gw
import simplenet as sn

###data 
#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?highlight=cifar
###
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def get_paramsize(net:torch.nn.Module):
    paramsize=0
    for w in gw.get_weights(net):
        p=1
        for i in w.size():
            p=p*i
        paramsize=paramsize+p
    print(paramsize)
    return paramsize

def init_weights(net:torch.nn.Module,seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)        
    for module in net.modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=0.1)
            if(module.bias is not None):
                module.bias.data.zero_()

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda:0")
print(device)

initnum=10
logperiod=1000
epochnum=3

def dump_weights(net,netname,epochnum=3,initnum=10,logperiod=1000):
    get_paramsize(net)
    net.to(device)
    fploss=open("losslog_%s.csv"%(netname),"w")

    for inits in range(initnum):
        f=open("params_%s_%d.csv"%(netname,inits),"a")
        init_weights(net,inits)
        #net.to(device)
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(epochnum):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                w0=gw.get_weights(net)
        
                running_loss += loss.item()

                if i % logperiod == logperiod-1:
                    print(f'[{inits}, {epoch + 1}, {i + 1:5d}] loss: {running_loss / logperiod:.3f}')
                    fploss.write(f'[{inits}, {epoch + 1}, {i + 1:5d}] loss: {running_loss / logperiod:.3f}\n')
                    running_loss = 0.0

        #            f.write('# length of weight {}'.format(len(w0)))
                    for w in w0:
                        np.savetxt(f,w.detach().cpu().numpy().flatten(),newline=",")
                    f.write("\n")                
                    #f.write(w0)                     
        #del net
        print('Finished %dth Training'%inits)
        f.close()
    fploss.close()

#dump_weights(sn.Simplenet(0),"simples",4,10,2000)
dump_weights(torchvision.models.resnet18(),"resnet18",4,10,1000)
dump_weights(torchvision.models.resnet50(),"resnet50",4,10,1000)
#torchvision.models.regnet_x_16gf