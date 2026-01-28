import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# STEP 1: DATASET
# ---------------------------
batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Scale to [-1, 1]
])

dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

# ---------------------------
# STEP 2: HYPERPARAMETERS
# ---------------------------
image_size = 28
nz = 100      # latent vector size
ngf = 64      # generator feature maps
ndf = 64      # discriminator feature maps
epochs = 10

# ---------------------------
# STEP 3: DEVICE
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ---------------------------
# STEP 4: GENERATOR
# ---------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, ngf * 7 * 7),
            nn.BatchNorm1d(ngf * 7 * 7),
            nn.ReLU(True),

            nn.Unflatten(1, (ngf, 7, 7)),

            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf // 2, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# ---------------------------
# STEP 5: DISCRIMINATOR
# ---------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(ndf * 2 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# ---------------------------
# STEP 6: INITIALIZATION
# ---------------------------
G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()

optimizerD = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

# ---------------------------
# STEP 7: TRAINING LOOP
# ---------------------------
print("\nStarting GAN Training...\n")

for epoch in range(epochs):
    for real, _ in dataloader:

        real = real.to(device)
        b = real.size(0)

        # Labels
        real_labels = torch.ones(b, 1, device=device)
        fake_labels = torch.zeros(b, 1, device=device)

        # ---- Train Discriminator ----
        optimizerD.zero_grad()

        output_real = D(real)
        loss_real = criterion(output_real, real_labels)

        noise = torch.randn(b, nz, device=device)
        fake = G(noise)

        output_fake = D(fake.detach())
        loss_fake = criterion(output_fake, fake_labels)

        lossD = loss_real + loss_fake
        lossD.backward()
        optimizerD.step()

        # ---- Train Generator ----
        optimizerG.zero_grad()

        output = D(fake)
        lossG = criterion(output, real_labels)

        lossG.backward()
        optimizerG.step()

    print(f"Epoch [{epoch+1}/{epochs}]  LossD: {lossD.item():.4f}  LossG: {lossG.item():.4f}")

# ---------------------------
# STEP 8: GENERATE IMAGES
# ---------------------------
print("\nGenerating images...")

noise = torch.randn(16, nz, device=device)
fake = G(noise).cpu().detach()

# Convert from [-1, 1] → [0, 1]
fake = (fake + 1) / 2

grid = vutils.make_grid(fake, nrow=4)

plt.figure(figsize=(6, 6))
plt.imshow(grid.permute(1, 2, 0))
plt.axis("off")
plt.title("Generated MNIST Digits (GAN)")
plt.show()
