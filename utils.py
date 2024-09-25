import torch
import numpy as np
from typing import Tuple
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split as skl_train_test_split


def get_data(
        data_path: str = 'data/cifar10_train.npz', is_linear: bool = False,
        is_binary: bool = False, grayscale: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Load CIFAR-10 dataset from the given path and return the images and labels.
    If is_linear is True, the images are reshaped to 1D array.
    If grayscale is True, the images are converted to grayscale.

    Args:
    - data_path: string, path to the dataset
    - is_linear: bool, whether to reshape the images to 1D array
    - is_binary: bool, whether to convert the labels to binary
    - grayscale: bool, whether to convert the images to grayscale

    Returns:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    '''
    data = np.load(data_path)
    X = data['images']
    try:
        y = data['labels']
    except KeyError:
        y = None
    X = X.transpose(0, 3, 1, 2)
    if is_binary:
        idxs0 = np.where(y == 0)[0]
        idxs1 = np.where(y == 1)[0]
        idxs = np.concatenate([idxs0, idxs1])
        X = X[idxs]
        y = y[idxs]
    if grayscale:
        X = convert_to_grayscale(X)
    if is_linear:
        X = X.reshape(X.shape[0], -1)
    
    # HINT: rescale the images for better (and more stable) learning and performance
    X = X/255.0

    return X, y


def convert_to_grayscale(X: np.ndarray) -> np.ndarray:
    '''
    Convert the given images to grayscale.

    Args:
    - X: np.ndarray, images in RGB format

    Returns:
    - X: np.ndarray, grayscale images
    '''
    return np.dot(X[..., :3], [0.2989, 0.5870, 0.1140])


def train_test_split(
        X: np.ndarray, y: np.ndarray, test_ratio: int = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Split the given dataset into training and test sets.

    Args:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    - test_ratio: float, ratio of the test set

    Returns:
    - X_train: np.ndarray, training images
    - y_train: np.ndarray, training labels
    - X_test: np.ndarray, test images
    - y_test: np.ndarray, test labels
    '''

    assert test_ratio < 1 and test_ratio > 0

    num_samples = len(X)
    num_test_samples = int(test_ratio * num_samples)
    num_train_samples = num_samples - num_test_samples

    # Shuffle the indices
    indices = np.random.permutation(num_samples)

    # Split the indices into training and test indices
    train_indices = indices[:num_train_samples]
    test_indices = indices[num_train_samples:]

    # Split the data based on the indices
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    # raise NotImplementedError('Split the dataset here')
    
    return X_train, y_train, X_test, y_test


def get_data_batch(
        X: np.ndarray, y: np.ndarray, batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Get a batch of the given dataset.

    Args:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    - batch_size: int, size of the batch

    Returns:
    - X_batch: np.ndarray, batch of images
    - y_batch: np.ndarray, batch of labels
    '''
    # idxs = # TODO: get random indices of the batch size without replacement from the dataset
    idxs = np.random.choice(len(X), size=batch_size, replace=False)
    return X[idxs], y[idxs]


# TODO: Read up on generator functions online
def get_contrastive_data_batch(
        X: np.ndarray, y: np.ndarray, batch_size: int
):  # Yields: Tuple[np.ndarray, np.ndarray, np.ndarray]
    '''
    Get a batch of the given dataset for contrastive learning.

    Args:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    - batch_size: int, size of the batch

    Yields:
    - X_a: np.ndarray, batch of anchor samples
    - X_p: np.ndarray, batch of positive samples
    - X_n: np.ndarray, batch of negative samples
    '''
    # raise NotImplementedError('Get a batch of here')
    np.random.seed(42)
    indexes = np.random.permutation(len(X))

    #selecting anchor samples
    anchors = X[indexes[0:0+batch_size]]
    
    #selecting positive and negative samples
    positives = []
    negatives = []
    for _ , label in enumerate(y[indexes[0:0+batch_size]]):
        positive_indexes = np.where(y == label)[0]
        negative_indexes = np.where(y != label)[0]
        positives.append(X[np.random.choice(positive_indexes)])
        negatives.append(X[np.random.choice(negative_indexes)])
    
    anchors , positives , negatives = np.array(anchors) , np.array(positives) , np.array(negatives) #converting to numpy array
    return anchors, positives ,negatives #returning the batch 

    

def plot_losses(
        train_losses: list, val_losses: list, title: str
) -> None:
    '''
    Plot the training and validation losses.

    Args:
    - train_losses: list, training losses
    - val_losses: list, validation losses
    - title: str, title of the plot
    '''
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig('images/loss.png')
    plt.close()


def plot_accuracies(
        train_accs: list, val_accs: list, title: str
) -> None:
    '''
    Plot the training and validation accuracies.

    Args:
    - train_accs: list, training accuracies
    - val_accs: list, validation accuracies
    - title: str, title of the plot
    '''
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.savefig('images/acc.png')
    plt.close()


def plot_tsne(
       z: torch.Tensor, y: torch.Tensor 
) -> None:
    '''
    Plot the 2D t-SNE of the given representation.

    Args:
    - z: torch.Tensor, representation
    - y: torch.Tensor, labels
    '''
    # z2 = # TODO: get 2D t-SNE of the representation
    z2 = TSNE(n_components=2).fit_transform(z.detach().numpy())
    scatter = plt.scatter(z2[:, 0], z2[:, 1], c=y.to('cpu'), cmap='tab10')
    plt.legend(*scatter.legend_elements(), title='Classes')
    plt.savefig('images/tsne.png')
    plt.close()
