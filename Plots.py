import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import json
import itertools
from statistics import mean

def plotLosses(loss, epochs, name):
    iter = range(len(epochs) * len(loss[0]))
    l = list(itertools.chain.from_iterable(loss))

    fig, ax = plt.subplots()
    ax.plot(iter, l, color="tab:purple", marker=".", linestyle='none', alpha=0.7)
    fig.suptitle('Loss Vs. Iterations', fontsize=12, fontweight="bold")
    ax.set_xlabel("Iterations", fontsize=10, fontweight="bold")
    ax.set_ylabel("Loss", color="black", fontsize=10, fontweight="bold")
    plt.grid()
    fig.savefig('{:}_loss_vs_Iteration.png'.format(name),
                format='png',
                dpi=100,
                bbox_inches='tight')

    l = list(map(mean, loss))

    fig, ax = plt.subplots()
    ax.plot(epochs, l, color="tab:purple", marker="^", linestyle='none', alpha=0.7)
    fig.suptitle('Mean Loss Vs. Epochs', fontsize=12, fontweight="bold")
    ax.set_xlabel("Epochs", fontsize=10, fontweight="bold")
    ax.set_ylabel("Loss (Avg)", color="tab:purple", fontsize=10, fontweight="bold")
    plt.grid()
    fig.savefig('{:}_Loss_vs_Epochs.png'.format(name),
                format='png',
                dpi=100,
                bbox_inches='tight')
    return

def plotValidationAccuracy(mean_iou, mean_dice, epochs, name):
    fig, ax = plt.subplots()
    ax.plot(epochs, mean_dice, color="tab:blue", marker="^", linestyle='none', alpha=0.7)
    fig.suptitle('Mean Scores Vs. Epochs', fontsize=12, fontweight="bold")
    ax.set_xlabel("Epochs", fontsize=10, fontweight="bold")
    ax.set_ylabel("DICE Score (Avg)", color="tab:blue", fontsize=10, fontweight="bold")

    ax2 = ax.twinx()
    ax2.plot(epochs, mean_iou, color="tab:red", marker="*", linestyle='none', alpha=0.7)
    ax2.set_ylabel("IoU (Avg)", color="tab:red", fontsize=10, fontweight="bold")
    plt.grid()
    fig.savefig('{:}_Scores_vs_Epochs.png'.format(name),
                format='png',
                dpi=100,
                bbox_inches='tight')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="prostate", help="name of the dataset")
    parser.add_argument("--n_epochs", type=int, default=75, help="number of epochs of training")
    opt = parser.parse_args()
    with open("{:}_{:}_train_stat.json".format(opt.dataset_name, opt.n_epochs)) as json_file:
        data = json.load(json_file)
        plotLosses(data['loss'], data['epochs'], opt.dataset_name)
        plotValidationAccuracy(data['mIoU'], data['mDice'], data['epochs'], opt.dataset_name)
