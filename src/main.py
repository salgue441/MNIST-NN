from typing import List, Tuple
from neural_network import NeuralNetwork
from utils.utils import load_mnist, create_validation_split
import numpy as np

def train_with_validation(
    model: NeuralNetwork,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_iterations: int = 100,
    eval_interval: int = 100
) -> Tuple[List[float], List[float]]:
  """
  Train the model while monitoring validation performance.
    
    Args:
        model: Neural Network model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        num_iterations: Number of training iterations
        eval_interval: How often to evaluate on validation set
        
    Returns:
        Lists containing training and validation accuracies
  """

  train_acc_history, val_acc_history = [], []
  Y_encoded = model.one_hot_encode(y_train, 10)

  for i in range(num_iterations):
    cache_list, Y_hat = model.forward_propagation(X_train)
    cost = model.compute_cost(Y_encoded, Y_hat)
    gradients = model.backward_propagation(X_train, Y_encoded, cache_list)
    model.update_parameters(gradients)

    if (i + 1) % eval_interval == 0:
      train_acc = model.evaluate(X_train, y_train)
      val_acc = model.evaluate(X_val, y_val)

      train_acc_history.append(train_acc)
      val_acc_history.append(val_acc)

      print(f"Iteration {i + 1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}, Cost = {cost:.6f}")

  return train_acc_history, val_acc_history

if __name__ == "__main__":
  X_train, y_train, X_test, y_test = load_mnist()
  X_train, X_val, y_train, y_val = create_validation_split(
        X_train, y_train,
        val_size=0.2,
        random_state=42
    )
  
  model = NeuralNetwork(
    layer_sizes=[784, 128, 64, 10],
    learning_rate=0.1
  )

  train_acc_history, val_acc_history = train_with_validation(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    num_iterations=1000,
    eval_interval=50
  )

  test_accuracy = model.evaluate(X_test, y_test)
  print(f"Test Accuracy: {test_accuracy:.4f}")

