import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np

# =================================================
# Configurations
# =================================================
DATA_DIR = "E:/New folder/Health-Based-APP/data/chest_xray"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

image_size = 128
batch_size = 64
nz = 100
ngf = 64
ndf = 64
nc = 3
num_epochs = 20
lr = 0.0002
beta1 = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =================================================
# Dataset
# =================================================
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = torchvision.datasets.ImageFolder(root=DATA_DIR, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"Dataset loaded: {len(dataset)} images")

# =================================================
# Generator & Discriminator
# =================================================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# Initialize models
netG = Generator().to(device)
netD = Discriminator().to(device)

# ---------- Resume from checkpoint ----------
resume_epoch = 0
all_checkpoints = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
if all_checkpoints:
    latest_epoch = max([int(f.split('_')[-1].split('.')[0]) for f in all_checkpoints])
    netG.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"netG_epoch_{latest_epoch}.pth")))
    netD.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"netD_epoch_{latest_epoch}.pth")))
    resume_epoch = latest_epoch
    print(f"Resuming training from epoch {resume_epoch+1}")

# =================================================
# Loss and Optimizers
# =================================================
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_label = 1.
fake_label = 0.

# =================================================
# Training Loop
# =================================================
print("Starting Training...")

for epoch in range(resume_epoch, num_epochs):
    for i, (data, _) in enumerate(dataloader):
        ############################
        # (1) Train Discriminator
        ############################
        netD.zero_grad()
        real = data.to(device)
        b_size = real.size(0)
        output = netD(real)
        labels = torch.full_like(output, real_label, device=device)  # fix shape issue
        lossD_real = criterion(output, labels)
        lossD_real.backward()

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        output = netD(fake.detach())
        labels = torch.full_like(output, fake_label, device=device)  # fix shape issue
        lossD_fake = criterion(output, labels)
        lossD_fake.backward()
        optimizerD.step()

        ############################
        # (2) Train Generator
        ############################
        netG.zero_grad()
        output = netD(fake)
        labels = torch.full_like(output, real_label, device=device)  # trick discriminator
        lossG = criterion(output, labels)
        lossG.backward()
        optimizerG.step()

    # Print progress
    print(f"Epoch [{epoch+1}/{num_epochs}]  LossD: {(lossD_real+lossD_fake).item():.4f}  LossG: {lossG.item():.4f}")

    # Save generated images every epoch
    fake_images = netG(fixed_noise).detach().cpu()
    save_image(fake_images, os.path.join(OUTPUT_DIR, f"generated_epoch_{epoch+1}.png"), normalize=True)

    # Show images in real-time
    grid = make_grid(fake_images, normalize=True)
    np_grid = grid.numpy()
    plt.figure(figsize=(8,8))
    plt.imshow(np.transpose(np_grid, (1,2,0)))
    plt.title(f"Generated Images - Epoch {epoch+1}")
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()

    # Save models every epoch
    torch.save(netG.state_dict(), os.path.join(MODEL_DIR, f"netG_epoch_{epoch+1}.pth"))
    torch.save(netD.state_dict(), os.path.join(MODEL_DIR, f"netD_epoch_{epoch+1}.pth"))

# =================================================
# Save final models
# =================================================
torch.save(netG.state_dict(), os.path.join(MODEL_DIR, "netG_final.pth"))
torch.save(netD.state_dict(), os.path.join(MODEL_DIR, "netD_final.pth"))

print("Training Complete. Models saved.")
