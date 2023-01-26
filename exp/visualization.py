import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


def plot_learning(loss_medil, loss_vanilla, biadj_mat):
    """ Plot learning curve of medil and vanilla VAE
    Parameters
    ----------
    loss_medil: loss history of medil + VAE
    loss_vanilla: loss history of vanilla VAE
    biadj_mat: adjacency matrix of the bipartite graph

    Returns
    -------
    fig: figure of learning curve
    """

    # obtain data and define figure
    m, n = biadj_mat.shape
    [train_loss_medil, valid_loss_medil] = loss_medil
    [train_loss_vanilla, valid_loss_vanilla] = loss_vanilla
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    # plot train_llh and valid_llh
    ax.set_title(f"Learning curve of loss functions [dim_latent={m}, dim_obs={n}]")
    ax.plot(train_loss_medil, color=sns.color_palette()[0], label="medil_train")
    ax.plot(train_loss_vanilla, color=sns.color_palette()[1], label="vanilla_train")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training ELBO")

    # calculate disparity score
    ax_ = ax.twinx()
    ax_.grid(False)
    ax_.plot(valid_loss_medil, color=sns.color_palette()[2], label="medil_valid")
    ax_.plot(valid_loss_vanilla, color=sns.color_palette()[3], label="vanilla_valid")
    ax_.set_ylabel("Validation ELBO")

    handles, labels = ax.get_legend_handles_labels()
    handles_, labels_ = ax_.get_legend_handles_labels()
    ax.legend(handles + handles_, labels + labels_, loc="upper right")

    return fig
