import matplotlib.pyplot as plt
import numpy as np


def plot_images(real_img, fake_img, reco_img, savefile):
    fig, axes = plt.subplots(ncols=3, figsize=(10, 5))
    axes[0].imshow(real_img)
    axes[1].imshow(fake_img)
    axes[2].imshow(reco_img)
    fig.tight_layout()
    fig.savefig(savefile, dpi=150)


def to_numpy(tensor):
    return np.clip(tensor.numpy().squeeze(), 0.0, 1.0)


def plot_examples(real_gen, fake_gen, reco_gen, num=10, base_name="example"):
    for i, (real, fake, reco) in enumerate(zip(real_gen, fake_gen, reco_gen)):
        filename = base_name + f"_{i}.png"
        plot_images(to_numpy(real), to_numpy(fake), to_numpy(reco), filename)
        if i == num - 1:
            break
