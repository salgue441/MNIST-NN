from typing import Dict, Optional, Tuple
from tqdm import tqdm
import numpy as np

def stratify_indices(y: np.ndarray) -> Dict[int, np.ndarray]:
  """
  Group indices by class label for stratified sampling.

  Args:
    y (np.ndarray): Array of class labels.

  Returns:
    Dict[int, np.ndarray]: Dictionary of class labels to indices.
  """

  unique_classes = np.unique(y)
  return {
    label: np.where(y == label)[0] 
    for label in unique_classes
  }

def get_stratified_split_indices(
    y: np.ndarray,
    split_size: float,
    random_state: Optional[int] = None
) -> np.ndarray:
  """
  Get indices for a stratified split of the data.

  Args:
    y (np.ndarray): Array of class labels.
    split_size (float): Proportion of the data to include in the split.
    random_state (Optional[int], optional): Seed for the random number 
                                            generator. Defaults to None.

  Returns:
    np.ndarray: Indices for the stratified split.
  """

  if random_state is not None:
    np.random.seed(random_state)

  stratified_indices = stratify_indices(y)
  split_indices = []

  for class_indices in stratified_indices.values():
    n_samples: int = len(class_indices)
    n_split: int = int(np.round(n_samples * split_size))

    shuffled_indices = np.random.permutation(class_indices)
    split_indices.extend(shuffled_indices[:n_split])

  return np.array(split_indices)

def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.2,
    test_size: float = 0.2,
    stratify: bool = False,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """
  Split dataset into train, validation, and test sets with optional stratification.

  Args:
    X (np.ndarray): Input data of shape (n_features, n_samples)
    y (np.ndarray): Target labels of shape (n_samples, )
    val_size (float): Proportion of dataset to use for validation
    test_size (float): Proportion of dataset to use for testing
    stratify (bool): Whether to perform stratified split
    random_state (Optional[int]): Random seed for reproducibility

  Returns:
    Tuple containing:
      X_train (np.ndarray): Training features of shape (n_features, n_samples)
      X_val (np.ndarray): Validation features of shape (n_features, n_samples)
      X_test (np.ndarray): Testing features of shape (n_features, n_samples)
      y_train (np.ndarray): Training labels of shape (n_samples, )
      y_val (np.ndarray): Validation labels of shape (n_samples, )
      y_test (np.ndarray): Testing labels of shape (n_samples, )
  """

  if random_state is not None:
    np.random.seed(random_state)

  n_samples: int = X.shape[1]
  all_indices: np.ndarray = np.arange(n_samples)

  if stratify:
    test_indices = get_stratified_split_indices(y, test_size, random_state)

  else:
    n_test: int = int(np.round(n_samples * test_size))
    shuffled_indices: np.ndarray = np.random.permutation(all_indices)
    test_indices = shuffled_indices[:n_test]

  remaining_indices = np.setdiff1d(all_indices, test_indices)
  y_remaining = y[remaining_indices]

  val_proportion = val_size / (1 - test_size)
  if stratify:
    val_indices_subset = get_stratified_split_indices(y_remaining, val_proportion, random_state)
    val_indices = remaining_indices[val_indices_subset]

  else:
    n_val: int = int(np.round(n_samples * val_proportion))
    shuffled_indices = np.random.permutation(remaining_indices)
    val_indices = shuffled_indices[:n_val]

  train_indices = np.setdiff1d(remaining_indices, val_indices)

  X_train, X_val, X_test = X[:, train_indices], X[:, val_indices], X[:, test_indices]
  y_train, y_val, y_test = y[train_indices], y[val_indices], y[test_indices]

  return X_train, X_val, X_test, y_train, y_val, y_test

def print_split_info(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray
) -> None:
    """
    Print information about the dataset split.
    
    Args:
        X_train, X_val, X_test: Split feature sets
        y_train, y_val, y_test: Split target sets
    """
    total_samples = X_train.shape[1] + X_val.shape[1] + X_test.shape[1]

    print("Dataset Split Information:")
    print(f"Training set   : {X_train.shape[1]} samples ({(X_train.shape[1]/total_samples)*100:.1f}%)")
    print(f"Validation set : {X_val.shape[1]} samples ({(X_val.shape[1]/total_samples)*100:.1f}%)")
    print(f"Test set      : {X_test.shape[1]} samples ({(X_test.shape[1]/total_samples)*100:.1f}%)")
    
    unique_classes = np.unique(y_train)
    print("\nClass Distribution in Training Set:")

    if len(unique_classes) <= 10:
        print("\nClass distribution:")

        for split_name, y_split in tqdm(
            [("Training", y_train), ("Validation", y_val), ("Test", y_test)],
            desc="Calculating class distribution",
            unit="split"
        ):
            for class_label in unique_classes:
                class_count = np.sum(y_split == class_label)
                print(f"{split_name} set - Class {int(class_label)}: {class_count} samples ({(class_count/len(y_split))*100:.1f}%)")