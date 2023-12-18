import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm


parser = argparse.ArgumentParser('ResNet-18 Evaluation')
parser.add_argument('-w', help='Path to `resnet18.weights.pt`', default='resnet18.weights.pt')
args = parser.parse_args()

device = 'cpu'

final_model = torchvision.models.resnet18()
final_model.fc = nn.Linear(512, 10)
final_model.to(device)
final_model.load_state_dict(torch.load(args.w))

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224,), antialias=True)
])

# Loadeing training part of CIFAR-10
dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False, 
    download=True, 
    transform=transforms
)


dataloader = DataLoader(dataset, batch_size=16)
cre = nn.CrossEntropyLoss()
sum_acc = 0
sum_macro_prec = 0
sum_macro_recall = 0
sum_loss = 0

n = 0

for images, labels in tqdm(dataloader):
    probs = final_model(images.to(device))
    outputs = probs.argmax(dim=-1) 
    labels = labels.to(device)
    acc = (outputs == labels).sum() / labels.numel()
    
    ma_prec = 0
    mi_prec = 0
    ma_recall = 0
    mi_recall = 0
    
    for i in range(10):
        if ((outputs == i).sum() + ((outputs != i) & (labels == i)).sum()).item() != 0:
            ma_prec += (outputs == i).sum() / ((outputs == i).sum() + ((outputs != i) & (labels == i)).sum())
        if ((outputs == i).sum() + ((outputs == i) & (labels != i)).sum()).item() != 0:
            ma_recall += (outputs == i).sum() / ((outputs == i).sum() + ((outputs == i) & (labels != i)).sum())
        
    
    sum_acc += acc
    sum_macro_prec += ma_prec / 10
    sum_macro_recall += ma_recall / 10
    
    sum_loss += cre(probs, labels)

    n += 1
    
    
print(f"Loss: {sum_loss / n:.4f}")
print(f"Accuracy: {sum_acc / n:.4f}")
print(f"Precision: {sum_macro_prec / n:.4f}")
print(f"Recall: {sum_macro_recall / n:.4f}")
print(f"F1: {2 * (sum_macro_prec / n * sum_macro_recall / n) / (sum_macro_recall / n + sum_macro_prec / n)}")
    
    