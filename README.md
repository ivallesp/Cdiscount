# CDiscount solution for extreme image classification
This is the *quick and dirty* (apologies for that) solution I implemented as a *proof of concept* for testing the convolutional neural networks performance when classifying millions of images in +5000 distinct categories. The main goal of this effort was to **understand if it was possible to train this algorithm from scratch** and arrive to a good solution. The algorithm trained in a NVidia Titan XPascal during one month.

## Getting started
For reproducing the experiment, please, follow the subsequent steps
1) Clone the repository
2) Install the following libraries
    - pandas
    - tensorflow
    - numpy
    - tensorboard
3) Download the data from the Kaggle competition, bulk it into a folder in your system, and specify this path in the `settings.json` file on the root of this repository.
4) Specify the path of this folder in the `settings.json` file.
5) Run the following command from the root folder of the repository: `python model_res.py`
6) Check the performance of the algorithm from tensorboard

## Method
The architecture implemented is based on the **Microsoft ResNet** architecture, though with some differences. The algorithm is trained from scratch.

## Contribution
Feel free to send a pull request and it will be reviewed. Potential further worklines are summarized below
- Research for a way of making the algorithm faster
- Implement a better architecture
- Try pre-trained approaches and compare performance
- Research for a better way of combining predictions of the different images of the same product (now we are using the average of the scores)

## License
This project has been licensed under MIT agreement. Please, read the `LICENSE` file for further details. Copyright (c) 2017 Iván Vallés Pérez
