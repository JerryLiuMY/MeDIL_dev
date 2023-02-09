import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


def plot_learning(loss_true, loss_exact, loss_hrstc, loss_vanilla, biadj_mat):
    """ Plot learning curve of medil and vanilla VAE
    Parameters
    ----------
    loss_true: loss history of ground truth + VAE
    loss_exact: loss history of exact medil + VAE
    loss_hrstc: loss history of hrstc medil + VAE
    loss_vanilla: loss history of vanilla VAE
    biadj_mat: adjacency matrix of the bipartite graph

    Returns
    -------
    fig: figure of learning curve
    """

    # obtain data and define figure
    m, n = biadj_mat.shape
    [train_loss_true, valid_loss_true] = loss_true
    [train_loss_exact, valid_loss_exact] = loss_exact
    [train_loss_hrstc, valid_loss_hrstc] = loss_hrstc
    [train_loss_vanilla, valid_loss_vanilla] = loss_vanilla
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    # plot train_llh and valid_llh
    ax.set_title(f"Learning curve of loss functions [dim_latent={m}, dim_obs={n}]")
    ax.plot(train_loss_true, color=sns.color_palette()[0], label="true_train")
    ax.plot(train_loss_exact, color=sns.color_palette()[1], label="exact_train")
    ax.plot(train_loss_hrstc, color=sns.color_palette()[1], label="hrstc_train")
    ax.plot(train_loss_vanilla, color=sns.color_palette()[2], label="vanilla_train")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training ELBO")

    # calculate disparity score
    ax_ = ax.twinx()
    ax_.grid(False)
    ax_.plot(valid_loss_true, color=sns.color_palette()[0], linestyle="dashed", label="true_valid")
    ax_.plot(valid_loss_exact, color=sns.color_palette()[1], linestyle="dashed", label="exact_valid")
    ax_.plot(valid_loss_hrstc, color=sns.color_palette()[1], linestyle="dashed", label="hrstc_valid")
    ax_.plot(valid_loss_vanilla, color=sns.color_palette()[2], linestyle="dashed", label="vanilla_valid")
    ax_.set_ylabel("Validation ELBO")

    handles, labels = ax.get_legend_handles_labels()
    handles_, labels_ = ax_.get_legend_handles_labels()
    ax.legend(handles + handles_, labels + labels_, loc="upper right")

    return fig
