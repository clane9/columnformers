from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from timm.data import create_dataset
from tqdm import tqdm


def create_most_activating_image_grid(
    layers: List[str],
    features_path: str,
    dataset: str,
    num_img: int = 9,
    img_size: int = 100,
    desired_variance: float = 0.9,
) -> Dict[str, Image.Image]:
    """
    Takes the extracted features as a .h5 file and creates a (3 x num_patches) x (3 x num_patches)
    grid of the images that most strongly activate each layer of interest (i.e. a 3x3 grid of most strongly
    activating images per input patch for each specified layer).

    Args:
        - layers: a list of the layers to build visualisation for
        - features_path: path to saved features for layers to visualise (each
                         layer should be num_images x seq_len x embedding_dim)
        - dataset: path to the dataset used to visualise features in features_path
        - num_img: number of top activating images to find for each patch, should be a square number
        - img_size: desired size of the plotted images in pixels
        - desired_variance: the fraction of variance we want to explain when using PCA to calculate activation strength
    Returns:
        - a dictionary with keys the layer names and calues the compiled
          PIL.Image.Image images
    """

    assert num_img**0.5 % 1 == 0, "num_img should be a square number"
    assert 0 <= desired_variance <= 1, "desired_variance must be in the interval [0,1]"

    dataset = create_dataset(dataset, root=None, download=True)

    results = {}

    with h5py.File(features_path, "r") as f:
        for layer in layers:
            activations = torch.tensor(
                np.array(f[layer])
            )  # -> num_images x seq_len x embedding_dim
            activations = pca_activations(
                activations, desired_variance=desired_variance
            )  # -> num_images x seq_len
            activations = torch.topk(
                activations, num_img, 0
            ).indices  # -> num_img x seq_len

            seq_grid_dim = int(num_img**0.5)
            full_grid_dim = int(activations.shape[1] ** 0.5)

            sequence_images = [
                create_image_grid(
                    [dataset[j][0] for j in activations[:, i].tolist()],
                    grid_size=(seq_grid_dim, seq_grid_dim),
                    image_size=(img_size, img_size),
                )
                for i in range(activations.shape[1])
            ]
            grid_img = create_image_grid(
                sequence_images,
                grid_size=(full_grid_dim, full_grid_dim),
                image_size=(img_size, img_size),
            )

            results[layer] = grid_img

    return results


def create_image_grid(
    images: List[Image.Image],
    grid_size: Tuple[int] | None = None,
    image_size: Tuple[int] = (100, 100),
) -> Image.Image:
    """
    Arrange a list of PIL images into a grid.

    Args:
        - images: List of PIL.Image.Image objects.
        - grid_size: Tuple indicating the grid size (rows, cols).
                     If None, it will try to create a square grid.
        - image_size: Desired size for each image (width, height).
    Returns:
        - A single PIL.Image.Image object containing the grid.
    """
    images = [img.resize(image_size) for img in images]

    if grid_size is None:
        num_images = len(images)
        cols = int(num_images**0.5)
        rows = (num_images + cols - 1) // cols
    else:
        rows, cols = grid_size

    grid_img = Image.new("RGB", (cols * image_size[0], rows * image_size[1]))

    for i, img in enumerate(images):
        x = (i % cols) * image_size[0]
        y = (i // cols) * image_size[1]
        grid_img.paste(img, (x, y))

    return grid_img


def pca_activations(activations: torch.Tensor, desired_variance: float) -> torch.Tensor:
    """
    Perform PCA over the activations for each patch individually.
    Pick the k components that explain the desired percentage of variance,
    project activation vectors onto these and calculate the L2 norms of the projections.

    Args:
        - activations: tensor of activations of shape num_images, seq_len, embedding_dim
        - desired_variance: what fraction of the variance we want to explain (float in range [0,1])
    Returns:
        - a tensor of shape num_images, seq_len with a measure of the activation strength for each image and patch
    """
    # estimate the number of principal components that explain the desired percentage of variance
    pca_est = PCA()
    pca_est.fit(activations[:, 0, :].numpy())

    explained_variance_ratio = pca_est.explained_variance_ratio_

    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    k = np.argmax(cumulative_explained_variance >= desired_variance) + 1

    activation_strengths_list = []

    # project activations onto these components independently for each patch
    # and get their L2 norms
    for s in tqdm(range(activations.shape[1])):
        pca = PCA(n_components=k)
        projected_activations = pca.fit_transform(activations[:, s, :].numpy())

        activation_strengths_list.append(
            torch.norm(torch.tensor(projected_activations), dim=1)
        )

    return torch.stack(activation_strengths_list, dim=1)


def mean_activations(activations: torch.Tensor) -> torch.Tensor:
    """
    Calculates the mean of the activation vectors.

    Args:
        - activations: tensor of activations of shape num_images, seq_len, embedding_dim
    Returns:
        - a tensor of shape num_images, seq_len with a measure of the activation strength for each image and patch
    """

    return torch.mean(activations, 2)


layers = [
    "stages.0.blocks.1.mlp.act",
    "stages.1.blocks.1.mlp.act",
    "stages.2.blocks.1.mlp.act",
]
features_path = "topomoe_features/topomoe_tiny_3s_patch16_128/validation_features.h5"

dataset = "hfds/clane9/imagenet-100"

visualisations = create_most_activating_image_grid(layers, features_path, dataset)

for layer, img in visualisations.items():
    plt.imshow(img)
    plt.title(layer)
    plt.axis("off")
    plt.show()
