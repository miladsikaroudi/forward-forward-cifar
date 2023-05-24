<!-- # forward-forward-cifar
Here is my implementation for forward-forward techniques proposed recentely that works with Cifar10 and 100 datasets.  -->


# Forward-Forward Neural Network

This is an implementation of a Forward-Forward Neural Network inspired by Geoff Hinton's work. The network is trained on the CIFAR-10 dataset (can be extended to other computer vision datasets) to perform classification tasks.

## Prerequisites

- Python 3.x
- PyTorch
- torchvision

## Getting Started

1. Clone the repository:

   ```shell
   git clone https://github.com/your-username/forward-forward-neural-network.git
   ```

2. Install the required dependencies:

   ```shell
   pip install torch torchvision
   ```

3. Download the CIFAR-10 dataset:

   ```shell
   python download_cifar10.py
   ```

4. Train and evaluate the Forward-Forward Neural Network:

   ```shell
   python main.py
   ```
   
## Configuration

- You can modify the network architecture by adjusting the dimensions in the `Net` class in `train.py`.
- The learning rate and other hyperparameters can be tuned in the `Layer` class in `train.py`.

## Results

After training the model, the training and test errors will be displayed. The trained model will be evaluated on the CIFAR-10 test dataset.

## Contributing

Contributions are welcome! Please create a new branch and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Geoff Hinton for the inspiration behind the Forward-Forward Neural Network.
- PyTorch and torchvision teams for their excellent libraries.
- CIFAR-10 dataset creators for providing the dataset.
