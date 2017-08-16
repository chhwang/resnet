#!/bin/sh

# Download CIFAR-10 dataset
python dataset/cifar/cifar_to_records.py

# Run training
python train.py
