import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from Dataset import DiffusionDataset
# 直接调用，实例化模型，pretrained代表是否下载预先训练好的参数
model = torchvision.models.vgg16(pretrained = True)
# 获取原始的第一个卷积层参数
original_conv1 = model.features[0]

# 创建一个新的卷积层，将输入通道数设为1，其他参数与原始层相同
new_conv1 = nn.Conv2d(
    in_channels=1,
    out_channels=original_conv1.out_channels,
    kernel_size=original_conv1.kernel_size,
    stride=original_conv1.stride,
    padding=original_conv1.padding
)

# 替换模型中的第一个卷积层
model.features[0] = new_conv1
model.classifier.add_module("add_linear",nn.Linear(1000,65)) # 在vgg16的classfier里加一层

# model=torch.load('F:/Python_venv/Deep_Learning/pythonProject/Metamaterials/VGG/model90.pt')

# 打印模型中的参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

dataset_train = DiffusionDataset("F:/Python_venv/Deep_Learning/pythonProject/Metamaterials/Dataset/Train/Dataset_Train.h5")
trainloader = torch.utils.data.DataLoader(dataset=dataset_train,batch_size=32,shuffle=True)
dataset_val = DiffusionDataset("F:/Python_venv/Deep_Learning/pythonProject/Metamaterials/Dataset/Test/Dataset_Test.h5")
valloader = torch.utils.data.DataLoader(dataset=dataset_val,batch_size=32,shuffle=True)

# Define model
device = torch.device("cuda")
model.to(device)
# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
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
    torch.save(model, f"./VGG.pt")

