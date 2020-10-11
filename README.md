# Yoga Pose Classification and Fake Pose Generation System

The repository consists of source code for a yoga pose classification system and fake pose generation system built using deep learning. The goal of the project is to build a yoga pose classification system and a fake pose generation system. The long term goal of the project is to build a self-assisted yoga pose correction and training system using deep learning. Our primary motivation behind this project is the lack of availability of a good quality dataset for the development of such a system. Consequently, a yoga pose generation system can be used to build a large, good quality dataset of fake as well as real poses.

## Overview

The goal of this project is to build a deep learning model for fake pose generation of yoga poses. We
explore this task by first building a model for yoga pose classification, which is followed by a model
for fake pose generation. Firstly, a dataset consisting of images of four different yoga poses is built
and various body key points are extracted. Thereafter, key points detected are used as features for
the models. The yoga pose classification system consists of Long Short Term Memory(LSTM) cells.
The model is capable of classifying four different yoga poses. In addition, the fake pose generation
system consists of a Variational AutoEncoder. By training a variational autoencoder and interpolating
the latent space, fake poses are obtained.

## Software Stack
1. Python
2. Keras & Tensorflow
3. sklearn
4. Pandas,Numpy
5. matplotlib

## Dependencies
The codebase requires developers to specify the paths for the datasets. 
