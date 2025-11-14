Health-Based X-Ray GAN

This project implements a DCGAN-based system designed to generate synthetic chest X-ray images. The goal is to explore medical image generation, GAN training, and stable checkpointing while keeping the codebase clean, modular, and easy to extend. The training pipeline is fully automated, supports resuming from the last saved epoch, and logs all progress for reproducibility.

Overview

The model is trained on a dataset of chest X-ray images. These images are preprocessed into grayscale, resized to 64×64, normalized, and then used to train a Generator and Discriminator pair. The Generator learns to create realistic X-ray–like images from random noise, while the Discriminator learns to classify input images as real or fake.

The project includes mechanisms for saving checkpoints, recording loss values to a CSV file, and exporting image samples during training to visually monitor progress.

Project Structure

The training logic and model definitions are placed under the src/ directory. A checkpoints/ folder is created automatically to store model weights after every epoch. A samples/ directory is used to store generated images produced at intervals during training. A training_log.csv file keeps track of loss values for both networks.

How the Model Works

The Generator receives a 100-dimensional noise vector and upsamples it through a sequence of ConvTranspose layers to produce a 1×64×64 grayscale image. The architecture follows the standard DCGAN structure, using ReLU activations and a final Tanh output.

The Discriminator processes an image through a series of strided convolutions with LeakyReLU activations, reducing it down to a single probability score indicating whether the input is real or fake. Batch normalization is used in both models (except the first layer of the Discriminator) to maintain stability.

Training uses Binary Cross Entropy loss for both networks. Real images are assigned smoothed labels (0.9) which helps reduce instability during GAN training. Adam optimizers with a learning rate of 0.0002 are used for both models.

Training Pipeline

When training begins, the script automatically checks whether a checkpoint exists. If it does, all model weights and optimizer states are restored, allowing training to continue from the exact epoch it stopped at. If no checkpoint is found, the training starts from scratch.

During each epoch:

The Discriminator is trained on both real and fake images.

The Generator is trained to produce images that the Discriminator classifies as real.

The final loss values for both networks are written to training_log.csv.

Every 10 epochs, a grid of generated samples is saved in the samples/ directory.

After every epoch, a complete checkpoint file is saved, containing the current epoch, the Generator and Discriminator weights, and both optimizer states.

Dataset

Training expects chest X-ray images arranged inside the data/chest_xray/train folder. All images are automatically converted to grayscale, resized, and normalized. To reduce training time during experimentation, the script uses only half of the available dataset, chosen randomly.