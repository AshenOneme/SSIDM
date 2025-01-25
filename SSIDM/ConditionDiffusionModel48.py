from diffusers import UNet2DConditionModel
import torch
from torch import nn

class ClassConditionedUnet(nn.Module):
    def __init__(self, num_emb=64, emb_size=256):
        super().__init__()
        self.image_size = (48, 48)

        self.emb = nn.Sequential(nn.Linear(num_emb, 128),
                                 nn.GELU(),
                                 nn.Linear(128, 128),
                                 nn.GELU(),
                                 nn.Linear(128, 256),
                                 nn.GELU(),
                                 nn.Linear(256, emb_size))

        self.model = UNet2DConditionModel(
            sample_size=self.image_size,
            in_channels=1,
            out_channels=1,
            layers_per_block=4,  # 残差层个数
            block_out_channels=(64,128,256,512),
            down_block_types=(
                "DownBlock2D",  # 常规的ResNet下采样模块
                "DownBlock2D",  # 常规的ResNet下采样模块
                "AttnDownBlock2D",  # 含有spatial self-attention的ResNet下采样模块
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",  # 含有spatil self-attention的ResNet上采样模块
                "AttnUpBlock2D",  # 含有spatil self-attention的ResNet上采样模块
                "UpBlock2D",  # 上采样模块
                "UpBlock2D",  # 上采样模块
            ),
        )

    def forward(self, x, t, labels):
        encoder_hidden_states = self.emb(labels)
        encoder_hidden_states = encoder_hidden_states.unsqueeze(2).repeat(1, 1, 1280)
        out = self.model(sample=x, timestep=t, encoder_hidden_states=encoder_hidden_states).sample
        return out

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassConditionedUnet()
    print(model)
    model = model.to(device)
    X = torch.rand(size=(1, 1, 48, 48), dtype=torch.float32)
    X = X.cuda()
    Y = torch.rand(size=(1, 64), dtype=torch.float32).to(device)
    labels = model(X, torch.Tensor([1]).long().to(device), Y)
    print(labels.shape)