from learning.linear_mask import LinearMask
import torch.nn as nn
import torch


class VariationalAutoencoder(nn.Module):
    def __init__(self, m, n, biadj_mat):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(m, n)
        self.decoder = Decoder(m, n, biadj_mat)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        latent = self.latent_sample(mu, logvar)
        x_rec = self.decoder(latent)

        return x_rec, mu, logvar

    def latent_sample(self, mu, logvar):
        # the re-parameterization trick
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


class Block(nn.Module):
    def __init__(self, m, n):
        super(Block, self).__init__()
        self.input_dim = n
        self.latent_dim = m
        self.output_dim = n


class Encoder(Block):
    def __init__(self, m, n):
        super(Encoder, self).__init__(m, n)

        # first encoder layer
        self.inter_dim = self.input_dim
        self.enc1 = nn.Linear(in_features=self.inter_dim, out_features=self.inter_dim)

        # second encoder layer
        self.enc2 = nn.Linear(in_features=self.inter_dim, out_features=self.inter_dim)

        # map to mu and variance
        self.fc_mu = nn.Linear(in_features=self.inter_dim, out_features=self.latent_dim)
        self.fc_logvar = nn.Linear(in_features=self.inter_dim, out_features=self.latent_dim)

    def forward(self, x):
        # encoder layers
        inter = torch.relu(self.enc1(x))
        inter = torch.relu(self.enc2(inter))

        # calculate mu & logvar
        mu = self.fc_mu(inter)
        logvar = self.fc_logvar(inter)

        return mu, logvar


class Decoder(Block):
    def __init__(self, m, n, biadj_mat):
        super(Decoder, self).__init__(m, n)
        self.mask = biadj_mat.astype(float)

        # decoder layer -- estimate mean
        self.dec_mean = LinearMask(in_features=self.latent_dim, out_features=self.output_dim, mask=self.mask)

    def forward(self, z):
        # linear layer
        mean = self.dec_mean(z)

        return mean