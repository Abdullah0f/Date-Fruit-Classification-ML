# Date Fruit Classification

This project is dedicated to automating the classification of date fruits using machine learning techniques. It aims to improve the accuracy and efficiency in identifying different types of dates based on their external appearance features.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Getting Started](#getting-started)
3. [Usage](#usage)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Project Design](#project-design)
6. [Results](#results)
7. [Future Work](#future-work)

## Project Overview

This project lies within the intersection of machine learning, computer vision, and agriculture. The goal is to automate the classification of date fruits to solve the problem of manual classification, which is time-consuming and requires significant expertise.

The dataset comprises 898 images of seven different types of date fruit: DOKOL, SAFAVI, ROTANA, DEGLET, SOGAY, IRAQI, BERHI. These images were captured by a Computer Vision System (CVS). From these images, 34 features such as morphological characteristics, shape, and color were extracted using image processing techniques.

Three different machine learning models - SVM, Logistic Regression, Decision Tree, and a Neural Network - were trained and evaluated to classify the date fruits. The success of the models was measured using classification metrics, including accuracy, precision, recall, and the F1-score.

## Getting Started

1. Clone the repository: `git clone https://github.com/Abdullah0f/Date-Fruit-Classification-ML.git`
2. Install the prerequisites: You need to have Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn, and Keras installed.

## Usage

Follow the steps in the provided Jupyter notebook to run the code. The main steps are:

1. Import the necessary libraries
2. Load and explore the dataset
3. Preprocess the data
4. Visualize the data
5. Split the dataset into training and testing sets
6. Train several models and evaluate their performances
7. Determine the most important features
8. Summarize the results

## Evaluation Metrics

The success of the models is measured using classification metrics, including accuracy, precision, recall, and the F1-score. The confusion matrix is also used to visualize the performance of the model on each date fruit class.

## Project Design

The project design consists of the following steps:

1. Data Preprocessing
2. Exploratory Data Analysis (EDA)
3. Model Selection and Training
4. Model Evaluation
5. Model Optimization
6. Validation
7. Results and Conclusion
8. Future Work

## Results

The best performing model was SVM with an accuracy of 94.4% on the test set. More details can be found in the Jupyter notebook.

## Future Work

The project is open for improvements and extensions. Future work might include:

1. Testing more complex machine learning models and deep learning architectures.
2. Using a larger dataset or augmenting the current dataset to improve model performance.
3. Deploying the model for practical use.
