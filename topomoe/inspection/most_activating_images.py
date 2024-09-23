from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from timm.data import create_dataset

from columnformers.models.geometry import multilayer_geometry


def create_most_activating_image_grid(
    layers: List[str],
    features_path: str,
    dataset: str,
    num_img: int = 9,
    img_size: int = 100,
    desired_variance: Optional[float] = 0.9,
    p: Optional[int] = 2,
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
        - p: the order of the norm if measuring activation strength using lp norm
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
            activations = torch.tensor(np.array(f[layer]))
            # -> num_images x seq_len x embedding_dim
            activations = lp_activations(activations, p)
            # -> num_images x seq_len
            activations = torch.topk(activations, num_img, 0).indices
            # -> num_img x seq_len

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


def get_most_activating_image_for_feature(
    layer: str,
    feature: int,
    features_path: str,
    dataset: str,
    num_img: int = 1,
    img_size: int = 100,
) -> Tuple[Image.Image, torch.Tensor]:

    dataset = create_dataset(dataset, root=None, download=True)

    with h5py.File(features_path, "r") as f:
        activations = torch.tensor(np.array(f[layer]))[:, :, feature]
        # -> num_images x seq_len
        total_activations = activations.sum(dim=1)
        # -> num_images
        top_indices = torch.topk(total_activations, num_img, 0).indices
        # -> num_img

        data_images = [
            dataset[i][0].resize((img_size, img_size)) for i in top_indices.tolist()
        ]

        dim = int(activations.shape[1] ** 0.5)

        saliency_maps = []

        for i in top_indices.tolist():
            act = activations[i, :].view(dim, dim).detach().numpy()

            # Normalise the activation values to [0, 255]
            act_min = act.min()
            act_max = act.max()
            normalized_act = 255 * (act - act_min) / (act_max - act_min)

            saliency_img = Image.fromarray(normalized_act.astype(np.uint8)).convert("L")
            saliency_maps.append(saliency_img.resize((img_size, img_size)))

        grid_img = Image.new("RGB", (2 * img_size, num_img * img_size))

        for idx, (data_img, saliency_img) in enumerate(zip(data_images, saliency_maps)):
            y_offset = idx * img_size
            grid_img.paste(data_img, (0, y_offset))
            grid_img.paste(saliency_img.convert("RGB"), (img_size, y_offset))

        return grid_img, top_indices


def threed_plot_activations(
    layer: str,
    image_idx: int,
    features_path: str,
    dataset: str,
    desired_variance: float = 0.9,
) -> Image.Image:

    dataset = create_dataset(dataset, root=None, download=True)

    with h5py.File(features_path, "r") as f:
        dim = int(f[layer].shape[1] ** 0.5)
        acts = np.array(f[layer])[image_idx, :, :]

        pca_est = PCA()
        pca_est.fit(acts)
        explained_variance_ratio = pca_est.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)
        k = np.argmax(cumulative_explained_variance >= desired_variance) + 1

        pca = PCA(n_components=k)
        tensor_reduced = pca.fit_transform(acts)
        activations = tensor_reduced.reshape((dim, dim, k))

        x, y, z = np.indices(activations.shape)

        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        values = activations.flatten()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        sc = ax.scatter(x, y, z, c=values, cmap="copper")

        ax.set_xlabel("H")
        ax.set_ylabel("W")
        ax.set_zlabel("F")

        plt.title(f"IMGPCA_L({layer})_I({image_idx})_K({k})")

        plt.colorbar(sc)
        plt.show()


def threed_plot_activations_v2(
    layer: str,
    image_idx: int,
    features_path: str,
    dataset: str,
    desired_variance: float = 0.9,
) -> Image.Image:

    dataset = create_dataset(dataset, root=None, download=True)

    with h5py.File(features_path, "r") as f:
        activations = torch.tensor(np.array(f[layer]))

        num_images, seq_len, _ = activations.shape

        dim = int(seq_len**0.5)

        flattened_activations = activations.flatten(0, 1).numpy()

        pca_est = PCA()
        pca_est.fit(flattened_activations)
        explained_variance_ratio = pca_est.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)
        k = np.argmax(cumulative_explained_variance >= desired_variance) + 1

        pca = PCA(n_components=k)
        tensor_reduced = pca.fit_transform(flattened_activations)
        pca_img_acts = tensor_reduced.reshape((num_images, dim, dim, k))[
            image_idx, :, :, :
        ]

        x, y, z = np.indices(pca_img_acts.shape)

        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        values = pca_img_acts.flatten()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        sc = ax.scatter(x, y, z, c=values, cmap="copper")

        ax.set_xlabel("H")
        ax.set_ylabel("W")
        ax.set_zlabel("F")

        plt.title(f"DATAPCA_L({layer})_I({image_idx})_K({k})")

        plt.colorbar(sc)
        plt.show()


def dino_rgb_viz(layer: str, image_idx: list[int], features_path: str):
    """
    Args:
        - layer: layer to take activations from
        - image_idx: list of image indices from the same category
        - features_path: path to saved extracted features

    Colour the three first components with three different colours for each image and display in 2d
    """

    with h5py.File(features_path, "r") as f:
        activations = np.array(f[layer])

        dim = int(activations.shape[1] ** 0.5)

        first_pca = PCA(n_components=1)
        second_pca = PCA(n_components=3)

        positive_idx = []
        positive_acts = []

        final_img = np.zeros((len(image_idx), activations.shape[1], 3))

        # For each image, get the first principal component and remove patches with negative values
        for idx in image_idx:
            img_acts = activations[idx, :, :]
            img_acts_reduced = first_pca.fit_transform(img_acts)
            img_pos_idx = np.where(img_acts_reduced >= 0)[0]
            positive_idx.append(img_pos_idx)
            positive_acts.append(img_acts[img_pos_idx, :])

        # Find top 3 components over the remaining patches across the three images
        non_negative_patches = np.vstack(positive_acts)
        positive_acts_reduced = second_pca.fit_transform(non_negative_patches)
        start = 0
        for i, idx in enumerate(positive_idx):
            final_img[i, idx, :] = positive_acts_reduced[start : start + len(idx), :]
            start += len(idx)

        final_img = final_img.reshape((len(image_idx), dim, dim, 3))

        return final_img


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
    num_images, seq_len, _ = activations.shape

    flattened_activations = activations.flatten(0, 1).numpy()
    # estimate the number of principal components that explain the desired percentage of variance
    pca_est = PCA()
    pca_est.fit(flattened_activations)

    explained_variance_ratio = pca_est.explained_variance_ratio_

    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    k = np.argmax(cumulative_explained_variance >= desired_variance) + 1

    # project activations onto these components independently for each patch
    # and get their L2 norms
    pca = PCA(n_components=k)
    projected_activations = pca.fit_transform(flattened_activations)

    l2_norm_activations = torch.norm(torch.tensor(projected_activations), dim=1)

    return l2_norm_activations.reshape(num_images, seq_len)


def lp_activations(activations: torch.Tensor, p: int) -> torch.Tensor:
    """
    Calculates the l1 norm of the activation vectors.

    Args:
        - activations: tensor of activations of shape num_images, seq_len, embedding_dim
        - p: the order of the norm
    Returns:
        - a tensor of shape num_images, seq_len with a measure of the activation strength for each image and patch
    """

    return torch.linalg.vector_norm(activations, ord=p, dim=2)


def morans_i(activations: torch.Tensor):
    # activations shape num_patches x embedding_dim
    num_patches, embedding_dim = activations.shape
    root_num_patches = int(num_patches**0.5)

    activations = activations.reshape(embedding_dim, root_num_patches, root_num_patches)
    distances = multilayer_geometry(
        [root_num_patches] * embedding_dim, depth_offset=1.0
    )
    w = 1 / (distances + 1) ** 2
    flattened_acts = activations.flatten()

    mean_act = torch.mean(flattened_acts).item()

    s = torch.sum(
        w
        * (flattened_acts - mean_act).unsqueeze(1)
        * (flattened_acts - mean_act).unsqueeze(0)
    ).item()

    sum_sqr_acts = torch.sum((flattened_acts - mean_act) ** 2).item()
    return (flattened_acts.shape[0] / torch.sum(w).item()) * (s / sum_sqr_acts)


def activation_difference(activations1: torch.Tensor, activations2: torch.Tensor):
    assert activations1.shape == activations2.shape
    return (
        torch.sum((activations1 - activations2) ** 2).item() / activations1.numel()
    ) ** 0.5


def activation_structural_similarity(
    activations1: torch.Tensor, activations2: torch.Tensor
) -> float:
    assert activations1.shape == activations2.shape

    # activations of shape num_patches x embedding_dim
    num_patches, embedding_dim = activations1.shape
    dim = int(num_patches**0.5)

    activations1 = activations1.reshape(dim, dim, embedding_dim).numpy()
    activations2 = activations2.reshape(dim, dim, embedding_dim).numpy()

    act_max = max(activations1.max(), activations2.max())
    act_min = min(activations1.min(), activations2.min())

    ssim_const = ssim(
        activations1, activations2, data_range=act_max - act_min, channel_axis=2
    )
    return ssim_const


def clustering_similarity(
    activations1: torch.Tensor, activations2: torch.Tensor, n_clusters: int
):

    activations1_flat = activations1.flatten().numpy()
    activations2_flat = activations2.flatten().numpy()

    kmeans_1 = KMeans(n_clusters=n_clusters).fit(activations1_flat.reshape(-1, 1))
    kmeans_2 = KMeans(n_clusters=n_clusters).fit(activations2_flat.reshape(-1, 1))

    return (kmeans_1.labels_ == kmeans_2.labels_).mean()


def plot_top_activated_features(
    img_idx: int,
    layer: str,
    num_features: int,
    features_path: str,
    dataset: str,
    img_size: int = 100,
):

    dataset = create_dataset(dataset, root=None, download=True)

    with h5py.File(features_path, "r") as f:
        activations = torch.tensor(np.array(f[layer]))[
            img_idx, :, :
        ]  # num_patches x embedding_dim
        feature_acts = torch.linalg.vector_norm(
            activations, ord=2, dim=0
        )  # embedding_dim
        top_feature_idx = torch.topk(
            feature_acts, num_features, 0
        ).indices.tolist()  # num_features

        dim = int(activations.shape[0] ** 0.5)

        images = [dataset[img_idx][0]]

        for i in top_feature_idx:
            act = activations[:, i].view(dim, dim).detach().numpy()

            # Normalise the activation values to [0, 255]
            act_min = act.min()
            act_max = act.max()
            normalized_act = 255 * (act - act_min) / (act_max - act_min)

            saliency_img = Image.fromarray(normalized_act.astype(np.uint8)).convert("L")
            images.append(saliency_img.resize((img_size, img_size)))

        grid_img = Image.new("RGB", ((num_features + 1) * img_size, img_size))

        for idx, img in enumerate(images):
            x_offset = idx * img_size
            grid_img.paste(img.convert("RGB"), (x_offset, 0))

        return grid_img, top_feature_idx
