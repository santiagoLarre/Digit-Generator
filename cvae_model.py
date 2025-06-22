import torch
import torch.nn as nn

latent_dim = 20
image_size = 28 * 28
num_classes = 10

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(image_size + num_classes, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        h = self.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + num_classes, 400)
        self.fc2 = nn.Linear(400, image_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, y):
        z = torch.cat([z, y], dim=1)
        h = self.relu(self.fc1(z))
        return self.sigmoid(self.fc2(h))

class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, y), mu, logvar

    def decode(self, z, y):
        return self.decoder(z, y)
