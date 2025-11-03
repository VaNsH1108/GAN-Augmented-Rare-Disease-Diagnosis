import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os

# -----------------------------
# Configurations
# -----------------------------
dataroot = r"E:\New folder\Health-Based-APP\data\chest_xray"

workers = 0
batch_size = 64
image_size = 64
nc = 1
nz = 100
ngf = 64
ndf = 64
num_epochs = 50
lr_G = 0.0001
lr_D = 0.0001
beta1 = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = "outputs"
checkpoint_dir = "checkpoints"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# -----------------------------
# Dataset & Loader
# -----------------------------
dataset = dset.ImageFolder(
    root=dataroot,
    transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# -----------------------------
# Define Generator
# -----------------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# -----------------------------
# Define Discriminator
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# -----------------------------
# Initialize models & setup
# -----------------------------
netG = Generator().to(device)
netD = Discriminator().to(device)
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta1, 0.999))

# -----------------------------
# Track losses for plotting
# -----------------------------
losses_D = []
losses_G = []

# -----------------------------
# Training Loop
# -----------------------------
print(f"Starting Training for {num_epochs} epochs...")

for epoch in range(1, num_epochs + 1):
    lossD_epoch = 0.0
    lossG_epoch = 0.0

    for i, (data, _) in enumerate(dataloader, 0):
        ############################
        # (1) Update Discriminator
        ############################
        netD.zero_grad()
        real = data.to(device)
        b_size = real.size(0)
        label_real = torch.full((b_size,), 0.8, device=device)
        output_real = netD(real).view(-1)
        lossD_real = criterion(output_real, label_real)
        lossD_real.backward()

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = torch.clamp(netG(noise).detach(), -1, 1)
        label_fake = torch.full((b_size,), 0.0, device=device)
        output_fake = netD(fake).view(-1)
        lossD_fake = criterion(output_fake, label_fake)
        lossD_fake.backward()
        optimizerD.step()

        ############################
        # (2) Update Generator
        ############################
        netG.zero_grad()
        label_gen = torch.full((b_size,), 0.8, device=device)
        output = netD(fake).view(-1)
        lossG = criterion(output, label_gen)
        lossG.backward()
        optimizerG.step()

        lossD_epoch += (lossD_real + lossD_fake).item()
        lossG_epoch += lossG.item()

    # Average losses
    avg_lossD = lossD_epoch / len(dataloader)
    avg_lossG = lossG_epoch / len(dataloader)
    losses_D.append(avg_lossD)
    losses_G.append(avg_lossG)

    print(f"Epoch [{epoch}/{num_epochs}]  LossD: {avg_lossD:.4f}  LossG: {avg_lossG:.4f}")

    # Save generated samples every 10 epochs
    if epoch % 10 == 0:
        fake = netG(torch.randn(64, nz, 1, 1, device=device))
        vutils.save_image(fake.detach(), f"{output_dir}/fake_samples_epoch_{epoch}.png", normalize=True)

    # -----------------------------
    # Plot and save loss graph
    # -----------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(losses_D, label="Discriminator Loss", color="red")
    plt.plot(losses_G, label="Generator Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()

print("✅ Training complete. Loss plot saved to outputs/loss_plot.png")
