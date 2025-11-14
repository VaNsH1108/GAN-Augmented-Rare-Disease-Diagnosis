import os
import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils

# --------------------
# CONFIG
# --------------------
DATA_DIR = r"E:\New folder\Health-Based-APP\data\chest_xray\train"
BATCH_SIZE = 64
IMAGE_SIZE = 64
LATENT_DIM = 100
EPOCHS = 300  # change if needed

# --------------------
# DEVICE CHECK
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n✅ Training on: **{device.type.upper()}**")

# --------------------
# DATA TRANSFORMS
# --------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
total_images = len(dataset)
half_size = total_images // 2
dataset, _ = random_split(dataset, [half_size, total_images - half_size])

print(f"✅ Using {half_size} images (50% of dataset).")

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --------------------
# DCGAN MODELS
# --------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)


generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
opt_g = optim.Adam(generator.parameters(), lr=0.0002)
opt_d = optim.Adam(discriminator.parameters(), lr=0.0002)

# --------------------
# TRAINING LOOP
# --------------------
# --------------------
# CHECKPOINT RESUME
# --------------------
os.makedirs("checkpoints", exist_ok=True)
checkpoint_path = "checkpoints/gan_checkpoint.pth"

start_epoch = 1

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    opt_g.load_state_dict(checkpoint["opt_g"])
    opt_d.load_state_dict(checkpoint["opt_d"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"🔄 Resuming training from epoch {start_epoch}...\n")
else:
    print("✨ Starting fresh training...\n")

# --------------------
# TRAINING LOOP
# --------------------
for epoch in range(start_epoch, EPOCHS + 1):
    for real, _ in loader:
        real = real.to(device)
        batch_size = real.size(0)

        # Smoother labels (fix LossG spike)
        real_labels = 0.9 * torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        z = torch.randn(batch_size, LATENT_DIM, 1, 1).to(device)
        fake = generator(z)

        pred_real = discriminator(real)
        pred_fake = discriminator(fake.detach())

        loss_d_real = criterion(pred_real, real_labels)
        loss_d_fake = criterion(pred_fake, fake_labels)
        loss_d = loss_d_real + loss_d_fake

        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        # Train Generator
        z = torch.randn(batch_size, LATENT_DIM, 1, 1).to(device)
        fake = generator(z)
        pred_fake = discriminator(fake)
        loss_g = criterion(pred_fake, real_labels)

        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

    print(f"🔥 Epoch [{epoch}/{EPOCHS}] | LossD: {loss_d.item():.4f} | LossG: {loss_g.item():.4f}")

    # Save sample every 10 epochs
    if epoch % 10 == 0:
        os.makedirs("samples", exist_ok=True)
        utils.save_image(fake[:25], f"samples/epoch_{epoch}.png", nrow=5, normalize=True)
        print(f"📸 Saved → samples/epoch_{epoch}.png")

    # ✅ SAVE CHECKPOINT EVERY EPOCH
    torch.save({
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
    }, checkpoint_path)

print("\n✅ Training complete!")
