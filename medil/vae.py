import torch.nn as nn
import torch


class VariationalAutoencoder(nn.Module):
    def __init__(self, m, n):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(m, n)
        self.decoder = Decoder(m, n)

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
    def __init__(self, m, n):
        super(Decoder, self).__init__(m, n)

        # linear layer
        self.inter_dim = self.output_dim
        self.fc = nn.Linear(in_features=self.latent_dim, out_features=self.inter_dim)

        # first decoder layer
        self.dec2 = nn.Linear(in_features=self.inter_dim, out_features=self.inter_dim)

        # second decoder layer -- mean and logs2
        self.dec1_mean = nn.Linear(in_features=self.inter_dim, out_features=self.output_dim)
        if self.fit_s2:
            self.dec1_logs2 = nn.Linear(in_features=self.inter_dim, out_features=1)

    def forward(self, z):
        # linear layer
        inter = self.fc(z)
        inter = torch.relu(self.dec2(inter))

        if not self.fit_s2:
            mean = self.dec1_mean(inter)
            return mean
        else:
            mean = self.dec1_mean(inter)
            logs2 = self.dec1_logs2(inter)
            return mean, logs2


class DecoderMeDIL(Block):
    def __init__(self, m, n, biadj_mat):
        super(DecoderMeDIL, self).__init__(m, n)

        # linear layer
        self.biadj_mat = biadj_mat
        self.inter_dim = self.output_dim
        self.fc = nn.Linear(in_features=self.latent_dim, out_features=self.inter_dim)

        # decoder layer -- mean and logs2
        self.dec_mean = nn.Identity(in_features=self.inter_dim, out_features=self.output_dim)
        if self.fit_s2:
            self.dec_logs2 = nn.Linear(in_features=self.inter_dim, out_features=1)

    def forward(self, z):
        # decoder layers
        inter = self.activation(self.fc(z))
        inter = torch.masked_select(inter, self.biadj_mat)

        if not self.fit_s2:
            mean = self.dec_mean(inter)
            return mean
        else:
            mean = self.dec_mean(inter)
            logs2 = self.dec_logs2(inter)
            return mean, logs2
