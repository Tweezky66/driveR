import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

STAGE_INDICES = [1, 3, 6, 12]
SKIP_CHANNELS = [16, 24, 40, 112]
BOTTLENECK_CHANNELS = 960

class Encoder(nn.Module):
  def __init__(self, freeze=True) -> None:
      super().__init__()

      self.features = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights).features

      if freeze:
        for p in self.features.parameters():
          p.requires_grad =  False # freeze layers if needed

  def forward(self, x):
    skips = []

    for i, block in enumerate(self.features):
      x = block(x)
      if i in STAGE_INDICES:
        skips.append(x) # append skipped layers if its  index matching a defined above shape

    return skips, x

class UpBlock(nn.Module):
  def __init__(self, in_ch, skip_ch, out_ch) -> None:
     super().__init__()
     self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
     self.conv = nn.Sequential(
         nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
         nn.BatchNorm2d(out_ch),
         nn.ReLU(inplace=True)
     )

  def forward(self, x, skip):
    x = self.up(x)
    if x.shape[-2:] != skip.shape[-2:]:
      x = nn.functional.interpolate(x, skip.shape[-2:], mode="bilinear", align_corners=True)

    return self.conv(torch.cat([x, skip], dim=1))


class DepthNet(nn.Module):
  def __init__(self, freeze_encoder=True) -> None:
     super().__init__()

     self.encoder = Encoder(freeze=freeze_encoder)

     c1, c2, c3, c4 = SKIP_CHANNELS
     self.up_d = UpBlock(BOTTLENECK_CHANNELS, c4, 128)
     self.up_c = UpBlock(128, c3, 64)
     self.up_b = UpBlock(64, c2, 32)
     self.up_a = UpBlock(32, c1, 16)
     self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
     self.head = nn.Conv2d(16, 1, kernel_size=1)

  def forward(self, x):
    (a, b, c, d), bottleneck = self.encoder(x)

    y = self.up_d(bottleneck, d)
    y = self.up_c(y, c)
    y = self.up_b(y,  b)
    y = self.up_a(y, a)
    y = self.final_up(y)
    return self.head(y)