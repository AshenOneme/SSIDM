import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from ResNet18 import ResNet18
from Dataset import DiffusionDataset

# Create an instance of the Vim model
model = ResNet18(classes_num=65)

# 打印模型中的参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

transform = transforms.Compose(
    [transforms.Resize((224,224))]
)

dataset_train = DiffusionDataset("F:/Python_venv/Deep_Learning/pythonProject/Metamaterials/Dataset/Train/Dataset_Train.h5")
trainloader = torch.utils.data.DataLoader(dataset=dataset_train,batch_size=32,shuffle=True)
dataset_val = DiffusionDataset("F:/Python_venv/Deep_Learning/pythonProject/Metamaterials/Dataset/Test/Dataset_Test.h5")
valloader = torch.utils.data.DataLoader(dataset=dataset_val,batch_size=32,shuffle=True)

# Define model
device = torch.device("cuda")
# model_state_dict = torch.load('./model_pretrain.pt')
# model.load_state_dict(model_state_dict.state_dict())
model.to(device)

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

# Define loss
criterion = nn.MSELoss()

for epoch in range(200):
    # Train
    model.train()
    with tqdm(trainloader) as pbar:
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels= labels[:,:,1].to(device)
            optimizer.zero_grad()
            output = model(images)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            accuracy = ((output-labels)**2).float().mean()
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

    # Validation
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for images, labels in valloader:
            images = images.to(device)
            labels = labels[:,:,1].to(device)
            output = model(images)
            val_loss += criterion(output, labels).item()
            val_accuracy += (
                ((output-labels)**2).float().mean().item()
            )
    val_loss /= len(valloader)
    val_accuracy /= len(valloader)

    # Update learning rate
    scheduler.step()

    print(
        f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
    )
    if epoch % 10==0:
        torch.save(model,f"./model{epoch}.pt")
    torch.save(model, f"./ResNet18.pt")

