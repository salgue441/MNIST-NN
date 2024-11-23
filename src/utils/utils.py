from typing import Tuple, Union
from pathlib import Path
import numpy as np
import urllib.request
import gzip
import os

def load_mnist(path: Union[str, Path] = "data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST dataset from the raw files.
    Downloads the files if they don't exist.
    
    Args:
        path (Union[str, Path]): Directory to store/load the MNIST files
        
    Returns:
        Tuple containing training images, training labels, test images, test labels
    """

    urls = {
        'train_images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'train_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'test_images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'test_labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    }

    Path(path).mkdir(parents=True, exist_ok=True)
    files = {}

    for name, url in urls.items():
        filepath = Path(path) / Path(url).name

        if not filepath.exists():
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, filepath)

        with gzip.open(filepath, 'rb') as file:
            if 'images' in name:
                file.read(16)
                files[name] = np.frombuffer(file.read(), dtype=np.uint8)

            else:
                file.read(8)
                files[name] = np.frombuffer(file.read(), dtype=np.uint8)

    X_train = files['train_images'].reshape(-1, 28*28).T / 255.0
    X_test = files['test_images'].reshape(-1, 28*28).T / 255.0
    y_train = files['train_labels']
    y_test = files['test_labels']

    return X_train, y_train, X_test, y_test

def create_validation_split(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.2,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a validation split from the training data.
    
    Args:
        X (np.ndarray): Input features
        y (np.ndarray): Labels
        val_size (float): Proportion of data to use for validation
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple containing training features, validation features,
        training labels, validation labels
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    m = X.shape[1]
    indices = np.random.permutation(m)
    val_size = int(m * val_size)
    
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    return X[:, train_idx], X[:, val_idx], y[train_idx], y[val_idx]