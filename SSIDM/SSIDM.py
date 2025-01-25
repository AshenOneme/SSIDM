import sys 
dir='F:/Metamatreial_Walls/DDPM/CNN_DDPM'
Valdir='Val1'
outputNO='x1'
sys.path.append(dir) 
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

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noise_scheduler = DDIMScheduler(beta_schedule='squaredcos_cap_v2')
noise_scheduler.set_timesteps(50)
ddpm = torch.load(dir+'/diffusionmodel.pt',weights_only=False)
ddpm=ddpm.to(device)
ddpm.eval()

resnet = torch.load(dir+'/ResNet18.pt',weights_only=False)
resnet = resnet.to(device)
resnet.eval()

choice=[20]

diffusiondataset = DiffusionDataset(dir+"/Dataset_Test48.h5")

y=diffusiondataset[choice[0]][1].view(-1,64)
np.savetxt(f'{dir}/{Valdir}/curve.txt',y.reshape(-1,1))

output=np.zeros((50,48,48))
while True:
    sample = torch.randn(1, 1, 48, 48).to(device)
    for i, t in enumerate(noise_scheduler.timesteps):
        model_input = noise_scheduler.scale_model_input(sample, t)
        with torch.no_grad():
            noise_pred = ddpm(sample, t, y.to(device))
        scheduler_output = noise_scheduler.step(noise_pred, t, sample)
        sample = scheduler_output.prev_sample
        pred_x0 = scheduler_output.pred_original_sample
        results_eachstep = sample[0, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5
        output[i,:,:]=results_eachstep
    results = pred_x0[0, 0, :, :].detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5
    results[results<0.5]=0
    results[results>0.5]=1
    results= np.tile(results, (5, 10))

    Inputs=torch.Tensor(results).unsqueeze(0).unsqueeze(0).to(device)
    loop=resnet(Inputs)
    prediction=loop[0].detach().cpu().numpy()[1:]
    observation=y[0].cpu().numpy()
    error=np.mean(np.abs((prediction-observation)/observation))*100
    print(error)
    if error<10:
        np.savetxt(f'{dir}/{Valdir}/img/{outputNO}.csv',results,delimiter=',',fmt='%d')
        output= np.tile(output, (1,5, 10))
        output[output < 0.5] = 0
        output[output > 0.5] = 1
        np.save(file=f"{dir}/{Valdir}/img/{outputNO}.npy", arr=output)
        break

