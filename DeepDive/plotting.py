import matplotlib.pyplot as plt
import seaborn as sns


def plot_history(model):
    history = model.history
    epoch = "epoch"
    recon = "loss_reconstruction"
    kl = "loss_kl"
    kn_adv_dis = "known_adversary_discrete_covariates_loss"
    kn_adv_con = "known_adversary_continuous_covariates_loss"
    un_adv_dis = "unknown_adversary_discrete_covariates_loss"
    un_adv_con = "unknown_adversary_continuous_covariates_loss"
    r2 = 'r2'
    fig, ax = plt.subplots(1, 7, figsize=(30, 4))


    sns.lineplot(
        x=history[epoch],
        y=history["Train_" + recon],
        label="Train",
        ax=ax[0],
        legend=False,
    )
    if model.use_validation:
        sns.lineplot(
            x=history[epoch],
            y=history["Validation_" + recon],
            label="Validation",
            ax=ax[0],
            legend=False,
        )
    ax[0].set_title("Reconstruction loss")

    sns.lineplot(
        x=history[epoch],
        y=history["Train_" + kl],
        label="Train",
        ax=ax[1],
        legend=False,
    )
    if model.use_validation:
        sns.lineplot(
            x=history[epoch],
            y=history["Validation_" + kl],
            label="Validation",
            ax=ax[1],
            legend=False,
        )
    ax[1].set_title("KL loss")


    sns.lineplot(
        x=history[epoch],
        y=history["Train_" + kn_adv_dis],
        label="Train",
        ax=ax[2],
        legend=False,
    )
    if model.use_validation:
        sns.lineplot(
            x=history[epoch],
            y=history["Validation_" + kn_adv_dis],
            label="Validation",
            ax=ax[2],
            legend=False,
        )
    ax[2].set_title("Known adversaries discrete loss")

    sns.lineplot(
        x=history[epoch], y=history["Train_" + kn_adv_con], label="Train", ax=ax[3]
    )
    if model.use_validation:
        sns.lineplot(
            x=history[epoch],
            y=history["Validation_" + kn_adv_con],
            label="Validation",
            ax=ax[3],
        )
    ax[3].set_title("Known adversaries continuous loss")
    
    sns.lineplot(
        x=history[epoch],
        y=history["Train_" + un_adv_dis],
        label="Train",
        ax=ax[4],
        legend=False,
    )
    if model.use_validation:
        sns.lineplot(
            x=history[epoch],
            y=history["Validation_" + un_adv_dis],
            label="Validation",
            ax=ax[4],
            legend=False,
        )
    ax[4].set_title("Unknown adversaries discrete loss")

    sns.lineplot(
        x=history[epoch], y=history["Train_" + un_adv_con], label="Train", ax=ax[5]
    )
    if model.use_validation:
        sns.lineplot(
            x=history[epoch],
            y=history["Validation_" + un_adv_con],
            label="Validation",
            ax=ax[5],
        )
    ax[5].set_title("Unknown adversaries continuous loss")



    
    sns.lineplot(x=history[epoch], y=history["Train_" + r2], label="Train", ax=ax[6])
    if model.use_validation:
        sns.lineplot(
            x=history[epoch], y=history["Validation_" + r2], label="Validation", ax=ax[6]
        )
    ax[6].set_title("R2")
    plt.tight_layout()
    plt.show()
