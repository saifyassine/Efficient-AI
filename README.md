# Magnitude-Based Pruning for Neural Networks

A TensorFlow/Keras implementation of iterative magnitude-based pruning with fine-tuning, demonstrated on MNIST classification.

## Key Features
- **Magnitude-based pruning**: Removes smallest-magnitude weights to induce sparsity
- **Iterative pruning**: Gradually increases sparsity with fine-tuning between steps
- **Learning rate scheduling**: Automatically reduces learning rate during fine-tuning
- **MNIST example**: Complete workflow from training to pruned model evaluation

## Installation
```bash
pip install tensorflow numpy
