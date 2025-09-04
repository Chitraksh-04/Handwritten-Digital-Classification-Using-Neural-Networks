# Handwritten-Digital-Classification-Using-Neural-Networks

A complete implementation of a 2-layer neural network built from scratch using only NumPy for handwritten digit recognition. This project demonstrates the fundamentals of deep learning without using any high-level frameworks like TensorFlow or PyTorch.

## 🎯 Project Overview

This implementation creates a neural network capable of recognizing handwritten digits (0-9) from the MNIST dataset. The network is built entirely from scratch, implementing forward propagation, backpropagation, and optimization algorithms manually.

## 🚀 Features

- **Pure NumPy Implementation**: No deep learning frameworks used
- **2-Layer Neural Network**: Input layer, one hidden layer, and output layer
- **Multiple Activation Functions**: Tanh, ReLU, and Softmax implementations
- **Backpropagation**: Custom implementation of the backpropagation algorithm
- **Cost Function**: Cross-entropy loss for multi-class classification
- **Gradient Descent**: Manual implementation of parameter optimization
- **Accuracy Evaluation**: Training and testing accuracy calculation
- **Visualization**: Cost function plotting and sample prediction visualization

## 📋 Requirements

```bash
numpy
matplotlib
```

## 📁 Dataset Structure

The project expects the following CSV files in the root directory:

```
neural-network-from-scratch/
├── train_X.csv          # Training images (28x28 pixels flattened)
├── train_label.csv      # Training labels (one-hot encoded)
├── test_X.csv           # Test images (28x28 pixels flattened)
├── test_label.csv       # Test labels (one-hot encoded)
├── neural_network.py    # Main implementation
└── README.md           # This file
```

### Dataset Format

- **Images**: 28x28 pixel images flattened to 784 features
- **Labels**: One-hot encoded vectors of size 10 (for digits 0-9)
- **Training set**: Multiple samples for training
- **Test set**: Separate samples for evaluation

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/neural-network-from-scratch.git
   cd neural-network-from-scratch
   ```

2. **Install dependencies**
   ```bash
   pip install numpy matplotlib
   ```

3. **Prepare your dataset**
   - Download MNIST dataset or use your own handwritten digit dataset
   - Format as CSV files with proper structure
   - Place in the project root directory

## 🚀 Usage

### Basic Usage

```bash
python neural_network.py
```

### Configuration

You can modify the following hyperparameters in the code:

```python
# Network architecture
n_h = 1000          # Number of hidden units
learning_rate = 0.02 # Learning rate for gradient descent
iterations = 100     # Number of training iterations

# Run the model
Parameters, Cost_list = model(X_train, Y_train, n_h=n_h, 
                             learning_rate=learning_rate, 
                             iterations=iterations)
```

## 🏗️ Architecture

### Network Structure

```
Input Layer (784 units) → Hidden Layer (1000 units) → Output Layer (10 units)
```

### Components

1. **Input Layer**: 784 neurons (28x28 pixel values)
2. **Hidden Layer**: 1000 neurons with Tanh activation
3. **Output Layer**: 10 neurons with Softmax activation (one for each digit)

### Activation Functions

- **Tanh**: Used in hidden layer for non-linear transformation
- **ReLU**: Alternative activation function (implemented but not used)
- **Softmax**: Used in output layer for probability distribution

## 🔧 Implementation Details

### Key Functions

1. **`initialize_parameters(n_x, n_h, n_y)`**
   - Initializes weights and biases
   - Uses random initialization for weights
   - Zero initialization for biases

2. **`forward_propagation(x, parameters)`**
   - Computes forward pass through the network
   - Applies activation functions
   - Returns intermediate values for backpropagation

3. **`cost_function(a2, y)`**
   - Calculates cross-entropy loss
   - Measures model performance

4. **`backward_prop(x, y, parameters, forward_cache)`**
   - Implements backpropagation algorithm
   - Computes gradients for all parameters

5. **`update_parameters(parameters, gradients, learning_rate)`**
   - Updates weights and biases using gradient descent
   - Applies learning rate to control step size

6. **`model(x, y, n_h, learning_rate, iterations)`**
   - Main training loop
   - Combines all components for complete training

### Mathematical Foundation

The network uses:
- **Forward Propagation**: `z = W·x + b`, `a = activation(z)`
- **Cost Function**: `J = -1/m * Σ(y*log(a2))`
- **Backpropagation**: Chain rule for gradient computation
- **Parameter Update**: `W = W - α*dW`, `b = b - α*db`

## 📊 Results

The model achieves:
- Training accuracy: ~90-95%
- Test accuracy: ~85-90%

### Sample Output

```
Cost after 0 iterations is : 2.3026
Cost after 10 iterations is : 1.8234
Cost after 20 iterations is : 1.5678
...
Accuracy of Train Dataset 94.2 %
Accuracy of Test Dataset 89.5 %
```

## 📈 Visualization


![image alt](https://github.com/Chitraksh-04/Handwritten-Digital-Classification-Using-Neural-Networks/blob/1736bfec941fa92aa3f679021cef5a4aa66d9599/Screenshot%202025-09-04%20at%2011.56.01%20PM.png)
![image alt](https://github.com/Chitraksh-04/Handwritten-Digital-Classification-Using-Neural-Networks/blob/1736bfec941fa92aa3f679021cef5a4aa66d9599/Screenshot%202025-09-04%20at%2011.52.24%20PM.png)


1. **Cost Function Plot**: Shows training progress over iterations
2. **Sample Image Display**: Shows random training/test images
3. **Prediction Visualization**: Displays model predictions on test samples

## 🔧 Hyperparameter Tuning

### Recommended Configurations

| Parameter | Range | Description |
|-----------|--------|-------------|
| `n_h` | 100-2000 | Number of hidden units |
| `learning_rate` | 0.001-0.1 | Learning rate for optimization |
| `iterations` | 50-1000 | Number of training iterations |

### Tips for Better Performance

1. **Increase hidden units** for more complex patterns
2. **Adjust learning rate** if cost is not decreasing
3. **Add more iterations** for better convergence
4. **Experiment with activation functions**

## 🚨 Troubleshooting

### Common Issues

1. **FileNotFoundError**: Ensure CSV files are in the correct directory
2. **Poor accuracy**: Try different hyperparameters
3. **Slow convergence**: Increase learning rate or iterations
4. **Overfitting**: Reduce network size or add regularization

### Performance Optimization

- Use vectorized operations (already implemented)
- Reduce dataset size for faster experimentation
- Implement mini-batch gradient descent

## 🎯 Future Enhancements

- [ ] Add more hidden layers (deep neural network)
- [ ] Implement different initialization strategies
- [ ] Add regularization techniques (L1/L2)
- [ ] Implement batch normalization
- [ ] Add dropout for regularization
- [ ] Create a more flexible architecture
- [ ] Add support for different datasets
- [ ] Implement learning rate scheduling

## 📚 Learning Resources

This implementation demonstrates:
- Neural network fundamentals
- Backpropagation algorithm
- Gradient descent optimization
- Multi-class classification
- Vectorized computation with NumPy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add some improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request


## 🙏 Acknowledgments

- MNIST dataset for providing the standard benchmark
- NumPy community for excellent mathematical tools
- The deep learning community for foundational research

## 📧 Contact

Email - chitrakshtuli2012@gmail.com
---

⭐ If you found this project helpful for understanding neural networks, please give it a star!

## 🔬 Technical Details

### Mathematical Formulation

**Forward Propagation:**
```
Z¹ = W¹X + b¹
A¹ = tanh(Z¹)
Z² = W²A¹ + b²
A² = softmax(Z²)
```

**Cost Function:**
```
J = -1/m * Σ(Y * log(A²))
```

**Backward Propagation:**
```
dZ² = A² - Y
dW² = 1/m * dZ² * A¹ᵀ
db² = 1/m * Σ(dZ²)
dZ¹ = W²ᵀ * dZ² * tanh'(A¹)
dW¹ = 1/m * dZ¹ * Xᵀ
db¹ = 1/m * Σ(dZ¹)
```

This implementation provides a solid foundation for understanding how neural networks work under the hood!
