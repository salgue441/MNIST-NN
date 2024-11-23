import numpy as np

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """
  Compute the accuracy score.

  Args:
    y_true (np.ndarray): True class labels
    y_pred (np.ndarray): Predicted class labels

  Returns:
    float: Accuracy score
  """

  return np.mean(y_true == y_pred)

def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """
  Compute the precision score.

  Args:
    y_true (np.ndarray): True class labels
    y_pred (np.ndarray): Predicted class labels

  Returns:
    float: Precision score
  """

  true_positives = np.sum((y_true == 1) & (y_pred == 1))
  false_positives = np.sum((y_true == 0) & (y_pred == 1))

  return true_positives / (true_positives + false_positives)

def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """
  Compute the recall score.

  Args:
    y_true (np.ndarray): True class labels
    y_pred (np.ndarray): Predicted class labels

  Returns:
    float: Recall score
  """

  true_positives = np.sum((y_true == 1) & (y_pred == 1))
  false_negatives = np.sum((y_true == 1) & (y_pred == 0))

  return true_positives / (true_positives + false_negatives)

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """
  Compute the F1 score.

  Args:
    y_true (np.ndarray): True class labels
    y_pred (np.ndarray): Predicted class labels

  Returns:
    float: F1 score
  """

  precision = precision_score(y_true, y_pred)
  recall = recall_score(y_true, y_pred)

  return 2 * (precision * recall) / (precision + recall)

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
  """
  Compute the confusion matrix.

  Args:
    y_true (np.ndarray): True class labels
    y_pred (np.ndarray): Predicted class labels

  Returns:
    np.ndarray: Confusion matrix
  """

  true_positives = np.sum((y_true == 1) & (y_pred == 1))
  false_positives = np.sum((y_true == 0) & (y_pred == 1))
  true_negatives = np.sum((y_true == 0) & (y_pred == 0))
  false_negatives = np.sum((y_true == 1) & (y_pred == 0))

  return np.array([
    [true_negatives, false_positives],
    [false_negatives, true_positives]
  ])

