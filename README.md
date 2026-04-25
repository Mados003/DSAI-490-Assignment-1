# DSAI 490 Assignment 1

This project implements a convolutional Autoencoder (AE) and Variational Autoencoder (VAE) for the Medical MNIST dataset using TensorFlow and `tf.data`.

## GPU Setup

Native Windows TensorFlow GPU support is not available for TensorFlow 2.11+. To run on GPU, use one of these supported setups:

1. Recommended: WSL2 + Ubuntu + NVIDIA drivers + TensorFlow with CUDA support.
2. Legacy native Windows option: TensorFlow 2.10 with matching CUDA/cuDNN libraries.

## Install

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

If you are using WSL2, install TensorFlow with CUDA support from inside the Linux environment.

## Dataset

Use the provided Medical MNIST dataset in `data/raw/Medical MNIST/`. The pipeline loads images directly from the extracted class folders with `tf.data`.

## Run

Train the models:

```bash
python -m src.train
```

Open and run the experiment notebook:

- `notebooks/experiment_notebook.ipynb`

## Outputs

The notebook includes:

- AE and VAE training
- Reconstruction visualizations
- Latent space projection
- Generated samples
