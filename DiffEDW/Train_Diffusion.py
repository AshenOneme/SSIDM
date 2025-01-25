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

noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
# noise_scheduler = DDIMScheduler(num_train_timesteps=50)

n_epochs = 100
loss_fn = nn.MSELoss()
net = ClassConditionedUnet().to(device)
optimizer = optim.AdamW(net.parameters(), lr=1e-3)
# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


for epoch in range(n_epochs):
    with tqdm(train_loader) as pbar:
        for i, (x, y) in enumerate(pbar):
            x = x.to(device)*2-1
            y = y.to(device)

            noise = torch.randn_like(x)
            timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
            pred = net(noisy_x, timesteps,y)

            loss = loss_fn(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(Epoch=epoch+1,loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

    scheduler.step()
    if epoch % 10==0:
        torch.save(net,f"./model{epoch}.pt")

torch.save(net, r"./diffusionmodel.pt")
