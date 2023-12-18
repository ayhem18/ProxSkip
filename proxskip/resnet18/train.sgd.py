import torchvision
import torch
import torch.nn as nn
import time

torch.manual_seed(hash('Xt:!'))

from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224,), antialias=True),
])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms)
trainloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
)
testloader = DataLoader(
    testset,
    batch_size=64,
    shuffle=False,
)

resnet = torchvision.models.resnet18()
resnet.fc = nn.Linear(512, 10)
resnet.to(device)

epochs = 10
learning_rate = 1E-4
optimizer = torch.optim.SGD(resnet.parameters(), lr=learning_rate, momentum=0)
lossf = torch.nn.CrossEntropyLoss()
train_losses = []

for t in range(1, epochs + 1):
    resnet.train()
    train_loss = 0
    test_loss = 0
    n_train = 0
    n_test = 0

    for images, labels in trainloader:
        x = images.to(device)
        y_hat = resnet(x)
        loss_value = lossf(y_hat, labels.to(device))
        
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        
        train_loss += loss_value.item()
        train_losses.append(loss_value)
        n_train += 1

    
    with torch.no_grad():
        resnet.eval()
        
        for images, labels in testloader:
            x = images.to(device)
            y_hat = resnet(x)
            loss_value = lossf(y_hat, labels.to(device))
        
            
            test_loss += loss_value.item()
            n_test += 1
            
    print(f"Epoch {t}\tTrain Loss: {train_loss / n_train:.4f}\tTest Loss: {test_loss / n_test:.4f}")
    

torch.save(train_losses, 'resnet18.sgd.losses.pt')
torch.save(resnet.state_dict(), 'resnet18.sgd.weights.pt')