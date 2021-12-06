import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def plot_fig(x,
             y_true,
             y_pred,
             batch_number=0,
             channel_pred=0,
             path="test.png"):
    fig, axs = plt.subplots(1, 3, figsize=(16, 8))
    fig.subplots_adjust(left=0.05,
                        bottom=0.06,
                        right=0.95,
                        top=0.94,
                        wspace=0.4)

    im1 = axs[0].axis('off')
    im1 = axs[0].imshow(x[batch_number, :, :, 0], cmap='gray')
    im1 = axs[0].imshow(y_true[batch_number, :, :, channel_pred],
                        cmap='jet',
                        alpha=0.5)
    axs[0].set_title("VOI L")

    im2 = axs[1].axis('off')
    im2 = axs[1].imshow(x[batch_number, :, :, 0], cmap='gray')
    im2 = axs[1].imshow(x[batch_number, :, :, 1],
                        cmap='hot',
                        alpha=0.5,
                        norm=Normalize(vmin=0.0, vmax=2.5, clip=True))
    axs[1].margins(2, 2)
    cax2 = fig.add_axes([
        axs[1].get_position().x1 + 0.01, axs[1].get_position().y0, 0.02,
        axs[1].get_position().height
    ])
    plt.colorbar(im2, cax=cax2)  # Similar to fig.colorbar(im, cax = cax)
    axs[1].set_title("PET/CT")

    im3 = axs[2].axis('off')
    im3 = axs[2].imshow(x[batch_number, :, :, 0], cmap='gray')
    im3 = axs[2].imshow(y_pred[batch_number, :, :, channel_pred],
                        cmap='jet',
                        alpha=0.5,
                        norm=Normalize(vmin=0.0, vmax=0.5, clip=True))
    cax3 = fig.add_axes([
        axs[2].get_position().x1 + 0.01, axs[2].get_position().y0, 0.02,
        axs[2].get_position().height
    ])
    plt.colorbar(im3, cax=cax3)  # Similar to fig.colorbar(im, cax = cax)
    axs[2].set_title("Prediction")

    fig.savefig(path, transparent=False, facecolor="white")
    plt.close(fig=fig)