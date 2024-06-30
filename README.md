# ANN Standard Normalization, XOR Problem, and Graduate Admission Prediction

This project was developed as part of an Artificial Neural Networks (ANN) lecture. It consists of three main parts:
1. Standard Normalization by Vectorization
2. The XOR Problem
3. Prediction of Graduate Admission Chances

## Table of Contents
- [Introduction](#introduction)
- [Project Purpose](#project-purpose)
- [Dataset](#dataset)
- [Part 1: Standard Normalization by Vectorization](#part-1-standard-normalization-by-vectorization)
- [Part 2: The XOR Problem](#part-2-the-xor-problem)
- [Part 3: Graduate Admission Prediction](#part-3-graduate-admission-prediction)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Introduction
This repository contains three main parts: a standard normalization implementation, a solution to the XOR problem, and a neural network to predict graduate admission chances.

## Project Purpose
The primary purpose of this project is to apply various artificial neural network techniques to different practical problems, demonstrating the versatility and application of ANNs in data normalization, logical operations, and regression tasks.

## Dataset
For the Graduate Admission Prediction part, the dataset used is `admissionv2.csv`, which includes the following features:
- **GRE Score:** GRE score of the student (out of 340).
- **TOEFL Score:** TOEFL score of the student (out of 120).
- **University Rating:** Rating of the university (out of 5).
- **SOP:** Strength of Statement of Purpose (out of 5).
- **LOR:** Strength of Letter of Recommendation (out of 5).
- **CGPA:** Undergraduate GPA (out of 10).
- **Research:** Research experience (0 or 1).
- **Chance of Admit:** Probability of admission (0 to 1).

## Part 1: Standard Normalization by Vectorization
In this part, we implemented standard normalization using vectorized operations in NumPy. This technique adjusts the scales of input features, ensuring that they have a mean of zero and a standard deviation of one.

### Steps:
- **Calculate Mean and Standard Deviation:** Compute the mean and standard deviation for each feature.
- **Normalize Data:** Subtract the mean and divide by the standard deviation for each feature to achieve normalization.

## Part 2: The XOR Problem
The XOR problem is a classic test for neural networks. Here, we designed and trained a neural network using the minimum number of neurons required to achieve 100% accuracy in predicting XOR outputs. We created the XOR dataset and used only NumPy for computations, demonstrating the network's performance with appropriate visualizations.

### Steps:
- **Dataset Creation:** Manually creating the XOR dataset.
- **Network Design:** Designing a neural network with the minimum required neurons.
- **Training:** Training the network using backpropagation.
- **Evaluation:** Ensuring 100% accuracy and visualizing the decision boundary.

## Part 3: Graduate Admission Prediction
The third part of the project involves predicting the chances of graduate admission using a neural network. This part demonstrates a practical application of ANN in regression tasks.

### Data Preparation:
- **Loading Data:** The dataset is loaded using `pandas`.
- **Feature and Target Extraction:** The features (input variables) and the target (output variable) are separated. The target variable is the `Chance of Admit`.
- **Data Normalization:** The features are normalized to ensure that they are on the same scale, improving the performance of the neural network. This is done using standard normalization techniques.
- **Train-Test Split:** The dataset is split into training and testing sets to evaluate the model's performance on unseen data.

### Model Architecture:
- **Input Layer:** Takes the input features (7 in this case).
- **Hidden Layers:** Two hidden layers with ReLU activation functions. The first hidden layer has 128 neurons, and the second hidden layer has 64 neurons.
- **Output Layer:** A single neuron with a sigmoid activation function to predict the probability of admission.

### Hyperparameters:
- **Optimizer:** Adam optimizer is used for training the network.
- **Loss Function:** Binary Cross-Entropy loss is used to measure the difference between predicted probabilities and actual class labels.
- **Batch Size:** 32
- **Epochs:** 100

### Training and Evaluation:
- **Training:** The model is trained on the training set using the specified hyperparameters. During training, the loss and accuracy are monitored to ensure the model is learning effectively.
- **Evaluation:** After training, the model is evaluated on the test set. The performance is measured using accuracy, precision, recall, and F1-score. Additionally, the R² score and Root Mean Square Error (RMSE) are calculated to assess the model's regression performance.

## Evaluation Metrics
The model's performance is evaluated using the following metrics:
- **Accuracy:** The proportion of correctly predicted instances out of the total instances.
- **Precision:** The proportion of positive identifications that were actually correct.
- **Recall:** The proportion of actual positives that were correctly identified.
- **F1-Score:** The harmonic mean of precision and recall.
- **R² Score:** Measures the proportion of variance in the dependent variable that is predictable from the independent variables.
- **RMSE:** The standard deviation of the residuals (prediction errors).

## Results
The results section provides a summary of the model's performance on the validation set. The model's accuracy, precision, recall, F1-score, R² score, and RMSE are reported to assess its predictive capability.

## Contributing
We welcome contributions to improve this project. Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a new Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
This project was developed as part of an Artificial Neural Networks lecture. Special thanks to the course instructors and peers for their support and feedback.

## Contact
For any inquiries, please contact:
- Name: Tahsin Berk ÖZTEKİN
- Email: tahsinberkoztekin@gmail.com
