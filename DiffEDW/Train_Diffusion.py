import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler,DDIMScheduler,UNet2DModel
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import Dataset
import h5py
from Dataset import DiffusionDataset
from ConditionDiffusionModel48 import ClassConditionedUnet
import torch.optim as optim

filepath='F:/Python_venv/Deep_Learning/pythonProject/Metamaterials/DDPM'

diffusiondataset = DiffusionDataset(filepath+"/Dataset_Train48.h5")
train_loader = torch.utils.data.DataLoader(dataset=diffusiondataset,batch_size=128,shuffle=True)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建一个调度器
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
# noise_scheduler = DDIMScheduler(num_train_timesteps=50)

n_epochs = 100
loss_fn = nn.MSELoss()
net = ClassConditionedUnet().to(device)
# model_state_dict = torch.load('./model_pretrain.pt')
# net.load_state_dict(model_state_dict.state_dict())
# Define optimizer
optimizer = optim.AdamW(net.parameters(), lr=1e-3)
# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# 训练开始
for epoch in range(n_epochs):
    with tqdm(train_loader) as pbar:
        for i, (x, y) in enumerate(pbar):
            # 获取数据并添加噪声
            x = x.to(device)*2-1  # 数据被归一化到区间(-1, 1)
            y = y.to(device)

            noise = torch.randn_like(x)
            timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
            pred = net(noisy_x, timesteps,y)  # 注意这里输入了类别信息
            # 计算损失值
            loss = loss_fn(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(Epoch=epoch+1,loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

    scheduler.step()
    if epoch % 10==0:
        torch.save(net,f"./model{epoch}.pt")

torch.save(net, r"./diffusionmodel.pt")