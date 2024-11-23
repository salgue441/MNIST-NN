from typing import Dict, List, Union
from pathlib import Path
from dataclasses import dataclass
import numpy as np

@dataclass
class LayerCache:
  """
  Cache for storing intermediate values during forward propagation.
  """

  Z: np.ndarray
  A: np.ndarray

class NeuralNetwork:
  """
  Defines a simple feedforward neural network with ReLU activation functions
  for hidden layers and softmax activation function for the output layer.
  """

  def __init__(self, layer_sizes: List[int], learning_rate: float = 0.001):
    """
    Initialize neural network with the specified layer sizes and learning rate.

    Args:
      layer_sizes (List[int]): List of integers representing the number of
                               neurons in each layer of the network.
      learning_rate (float): Learning rate used for gradient descent.
    """

    self.layer_sizes = layer_sizes
    self.learning_rate = learning_rate
    self.parameters = self._initialize_parameters()

  def _initialize_parameters(self) -> Dict[str, np.ndarray]:
    """
    Initialize the weights and biases for each layer of the network.
    
    Returns:
      Dict[str, np.ndarray]: Dictionary containing the weights and biases for
                             each layer of the network.
    """

    parameters: Dict[str, np.ndarray] = {}
    for layer in range(1, len(self.layer_sizes)):
      parameters[f'W{layer}'] = np.random.randn(self.layer_sizes[layer], self.layer_sizes[layer - 1]) * 0.01

      parameters[f'b{layer}'] = np.zeros((self.layer_sizes[layer], 1))

    return parameters
  
  @staticmethod
  def relu(Z: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit (ReLU) activation function for a given input.

    Args:
      Z (np.ndarray): Input to the activation function.

    Returns:
      np.ndarray: Output of the activation function.
    """

    return np.maximum(0, Z)
  
  @staticmethod
  def relu_derivative(Z: np.ndarray) -> np.ndarray:
    """
    Derivative of the Rectified Linear Unit (ReLU) activation function.

    Args:
      Z (np.ndarray): Input to the activation function.

    Returns:
      np.ndarray: Derivative of the activation function.
    """

    return np.where(Z > 0, 1, 0)
  
  @staticmethod
  def softmax(Z: np.ndarray) -> np.ndarray:
    """
    Softmax activation function for a given input.

    Args:
      Z (np.ndarray): Input to the activation function.

    Returns:
      np.ndarray: Output of the activation function.
    """

    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
  
  @staticmethod
  def one_hot_encode(Y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    One hot encode the target labels Y into a binary matrix.

    Args:
      Y (np.ndarray): Target labels to encode.
      num_classes (int): Number of classes in the classification problem.

    Returns:
      np.ndarray: Binary matrix of shape (num_classes, m) where m is the number
                  of samples in Y.
    """

    return np.eye(num_classes)[Y.reshape(-1)].T
  
  def forward_propagation(self, X: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Forward propagation through the neural network.

    Args:
      X (np.ndarray): Input data of shape (n_x, m) where n_x is the number of
                      features and m is the number of samples.

    Returns:
      Dict[str, np.ndarray]: Dictionary containing the linear and activation
                             values for each layer of the network.
    """

    cache_list, A_prev = [], X
    for layer in range(1, len(self.layer_sizes) - 1):
      Z = np.dot(self.parameters[f'W{layer}'], A_prev) + self.parameters[f'b{layer}']

      A = self.relu(Z)
      cache_list.append(LayerCache(Z=Z, A=A))
      A_prev = A

    Z_out = self.parameters[f'W{len(self.layer_sizes)-1}'].dot(A_prev) + self.parameters[f'b{len(self.layer_sizes)-1}']
    A_out = self.softmax(Z_out)

    cache_list.append(LayerCache(Z=Z_out, A=A_out))
    return cache_list, A_out

  def compute_cost(self, Y: np.ndarray, Y_hat: np.ndarray) -> float:
    """
    Compute the cross-entropy loss between the predicted and true labels.

    Args:
      Y (np.ndarray): True labels of shape (num_classes, m).
      Y_hat (np.ndarray): Predicted labels of shape (num_classes, m).

    Returns:
      float: Cross-entropy loss.
    """

    m = Y.shape[1]
    return -np.sum(Y * np.log(Y_hat)) / m
  
  def backward_propagation(self, X: np.ndarray, Y: np.ndarray, cache_list: List[LayerCache]) -> Dict[str, np.ndarray]:
    """
    Backward propagation through the neural network.

    Args:
      X (np.ndarray): Input data of shape (n_x, m) where n_x is the number of
                      features and m is the number of samples.
      Y (np.ndarray): True labels of shape (num_classes, m).
      cache_list (List[LayerCache]): List of cache objects containing the
                                     intermediate values during forward propagation.

    Returns:
      Dict[str, np.ndarray]: Dictionary containing the gradients of the weights
                             and biases for each layer of the network.
    """

    m = X.shape[1]
    gradients = {}
    dZ = cache_list[-1].A - Y
    dW = np.dot(dZ, cache_list[-2].A.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m

    gradients[f'dW{len(self.layer_sizes)-1}'] = dW
    gradients[f'db{len(self.layer_sizes)-1}'] = db

    for layer in range(len(self.layer_sizes) - 2, 0, -1):
      dA = np.dot(self.parameters[f'W{layer+1}'].T, dZ)
      dZ = dA * self.relu_derivative(cache_list[layer-1].Z)
      dW = np.dot(dZ, cache_list[layer-1].A.T) / m
      db = np.sum(dZ, axis=1, keepdims=True) / m

      gradients[f'dW{layer}'] = dW
      gradients[f'db{layer}'] = db

    return gradients
  
  def update_parameters(self, gradients: Dict[str, np.ndarray]):
    """
    Update the network paremeters using computed gradients.

    Args:
      gradients (Dict[str, np.ndarray]): Dictionary containing the gradients of
                                         the weights and biases for each layer. 
    """

    for layer in range(1, len(self.layer_sizes)):
      self.parameters[f'W{layer}'] -= self.learning_rate * gradients[f'dW{layer}']
      self.parameters[f'b{layer}'] -= self.learning_rate * gradients[f'db{layer}']

  def fit(self, X: np.ndarray, Y: np.ndarray, num_iterations: int):
    """
    Train the neural network using the specified input data.

    Args:
      X (np.ndarray): Input data of shape (n_x, m) where n_x is the number of
                      features and m is the number of samples.
      Y (np.ndarray): True labels of shape (num_classes, m).
      num_iterations (int): Number of iterations to train the network.
    """

    Y_encoded = self.one_hot_encode(Y, self.layer_sizes[-1])
    for _ in range(num_iterations):
      cache_list, Y_hat = self.forward_propagation(X)
      cost = self.compute_cost(Y_encoded, Y_hat)
      gradients = self.backward_propagation(X, Y_encoded, cache_list)
      self.update_parameters(gradients)
      print(f'Cost: {cost:.6f}')

  def predict(self, X: np.ndarray) -> np.ndarray:
    """
    Predict the class labels for the input data.

    Args:
      X (np.ndarray): Input data of shape (n_x, m) where n_x is the number of
                      features and m is the number of samples.

    Returns:
      np.ndarray: Predicted class labels.
    """

    _, Y_hat = self.forward_propagation(X)
    return np.argmax(Y_hat, axis=0)
  
  def evaluate(self, X: np.ndarray, Y: np.ndarray) -> float:
    """
    Evaluate the accuracy of the neural network on the input data.

    Args:
      X (np.ndarray): Input data of shape (n_x, m) where n_x is the number of
                      features and m is the number of samples.
      Y (np.ndarray): True labels of shape (num_classes, m).

    Returns:
      float: Accuracy of the neural network on the input data.
    """

    Y_pred = self.predict(X)
    return np.mean(Y_pred == Y)
  
  def save_model(self, file_path: Union[str, Path]):
    """
    Save the model parameters to a file.

    Args:
      file_path (str): File path to save the model parameters.
    """

    file_path = Path(file_path)
    np.savez(file_path, **self.parameters)

  def load_model(self, file_path: str):
    """
    Load the model parameters from a file.

    Args:
      file_path (str): File path to load the model parameters.
    """

    file_path = Path(file_path)
    data = np.load(file_path)

    self.parameters = {key: data[key] for key in data.files}

