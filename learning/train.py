from vae import VariationalAutoencoder
from params import train_dict
from datetime import datetime
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_vae(m, n, train_loader, biadj_mat, cov_train):
    """ Training VAE with the specified image dataset
    :param m: dimension of the latent variable
    :param n: dimension of the observed variable
    :param train_loader: training image dataset loader
    :param biadj_mat: the adjacency matrix of the directed graph
    :param cov_train: covariance matrix for the training set
    :return: trained model and training loss history
    """

    # load parameters
    epoch, lr, beta = train_dict["epoch"], train_dict["lr"], train_dict["beta"]

    # building VAE
    model = VariationalAutoencoder(m, n, biadj_mat)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.8)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Number of parameters: {num_params}")

    # training loop
    model.train()
    train_loss = []
    for epoch in range(epoch):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Training on epoch {epoch}...")
        epoch_loss, nbatch = 0., 0

        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            recon_batch, mu_batch, logvar_batch = model(x_batch)
            loss = elbo_gaussian(x_batch, recon_batch, cov_train, mu_batch, logvar_batch, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update loss and nbatch
            epoch_loss += loss.item()
            nbatch += 1

        scheduler.step()

        # append training loss
        epoch_loss = epoch_loss / nbatch
        train_loss.append(epoch_loss)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Finish epoch {epoch} with loss {epoch_loss}")

    train_loss = np.array(train_loss)

    return model, train_loss


def valid_vae(model, valid_loader, cov_valid):
    """ Training VAE with the specified image dataset
    :param model: trained VAE model
    :param valid_loader: validation image dataset loader
    :param cov_valid: covariance matrix for the validation set
    :return: validation loss
    """

    # load parameters
    beta = train_dict["beta"]

    # set to evaluation mode
    model.eval()
    valid_loss, nbatch = 0., 0
    for x_batch, _ in valid_loader:
        with torch.no_grad():
            x_batch = x_batch.to(device)
            recon_batch, mu_batch, logvar_batch = model(x_batch)
            loss = elbo_gaussian(x_batch, recon_batch, cov_valid, mu_batch, logvar_batch, beta)

            # update loss and nbatch
            valid_loss += loss.item()
            nbatch += 1

    # report validation loss
    valid_loss = valid_loss / nbatch
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Finish validation with loss {valid_loss}")

    return valid_loss


def elbo_gaussian(x, x_recon, logs2, mu, logvar, beta):
    """ Calculating loss for variational autoencoder
    :param x: original image
    :param x_recon: reconstruction in the output layer
    :param logs2: log of the variance in the output layer
    :param mu: mean in the fitted variational distribution
    :param logvar: log of the variance in the variational distribution
    :param beta: beta
    :return: reconstruction loss + KL
    """

    # KL-divergence
    kl_div = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # reconstruction loss
    recon_loss = - torch.sum(
        logs2.mul(x.size(dim=1)/2) + torch.norm(x - x_recon, 2, dim=1).pow(2).div(logs2.exp().mul(2))
    )

    # loss
    loss = - beta * kl_div + recon_loss

    return - loss
