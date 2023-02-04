from learning.vae import VariationalAutoencoder
from learning.params import train_dict
from datetime import datetime
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_vae(m, n, biadj_mat, train_loader, valid_loader, cov_train, cov_valid):
    """ Training VAE with the specified image dataset
    :param m: dimension of the latent variable
    :param n: dimension of the observed variable
    :param train_loader: training image dataset loader
    :param valid_loader: validation image dataset loader
    :param biadj_mat: the adjacency matrix of the directed graph
    :param cov_train: covariance matrix for the training set
    :param cov_valid: covariance matrix for the validation set
    :return: trained model and training loss history
    """

    # load parameters
    epoch, lr, beta = train_dict["epoch"], train_dict["lr"], train_dict["beta"]

    # building VAE
    mask = biadj_mat.T.astype("float32")
    mask = torch.tensor(mask).to(device)
    model = VariationalAutoencoder(m, n, mask)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.90)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Number of parameters: {num_params}")

    # training loop
    model.train()
    train_loss = []
    valid_loss = []
    cov_train = cov_train.astype("float32")
    cov_train = torch.tensor(cov_train).to(device)

    for epoch in range(epoch):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Training on epoch {epoch}...")
        train_ls, nbatch = 0., 0

        for x_batch, _ in train_loader:
            batch_size = x_batch.shape[0]
            x_batch = x_batch.to(device)
            recon_batch, mu_batch, logvar_batch = model(x_batch)
            loss = elbo_gaussian(x_batch, recon_batch, cov_train, mu_batch, logvar_batch, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update loss and nbatch
            train_ls += loss.item() / batch_size
            nbatch += 1

        # for p in model.parameters():
        #     print(p.grad)

        # finish training epoch
        scheduler.step()
        train_ls = train_ls / nbatch
        train_loss.append(train_ls)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Finish training epoch {epoch} with loss {train_ls}")

        # append validation loss
        valid_ls = valid_vae(model, valid_loader, cov_valid)
        valid_loss.append(valid_ls)

    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)

    return train_loss, valid_loss


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
    valid_ls, nbatch = 0., 0
    cov_valid = cov_valid.astype("float32")
    cov_valid = torch.tensor(cov_valid).to(device)

    for x_batch, _ in valid_loader:
        with torch.no_grad():
            batch_size = x_batch.shape[0]
            x_batch = x_batch.to(device)
            recon_batch, mu_batch, logvar_batch = model(x_batch)
            loss = elbo_gaussian(x_batch, recon_batch, cov_valid, mu_batch, logvar_batch, beta)

            # update loss and nbatch
            valid_ls += loss.item() / batch_size
            nbatch += 1

    # report validation loss
    valid_ls = valid_ls / nbatch
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Finish validation with loss {valid_ls}")

    return valid_ls


def elbo_gaussian(x, x_recon, cov, mu, logvar, beta):
    """ Calculating loss for variational autoencoder
    :param x: original image
    :param x_recon: reconstruction in the output layer
    :param cov: covariance matrix of the data distribution
    :param mu: mean in the fitted variational distribution
    :param logvar: log of the variance in the variational distribution
    :param beta: beta
    :return: reconstruction loss + KL
    """

    # KL-divergence
    kl_div = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # reconstruction loss
    diff = x - x_recon
    recon_loss = torch.sum(
        torch.det(cov) + torch.diagonal(torch.mm(torch.mm(diff, torch.inverse(cov)), torch.transpose(diff, 0, 1)))
    ).mul(-1/2)

    # loss
    loss = - beta * kl_div + recon_loss

    return - loss
