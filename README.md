# Spam Detection using Machine Learning

This repository contains code for a spam detection system using machine learning models trained on text data. The system classifies messages into two categories: spam and ham (non-spam).

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)

## Introduction

The spam detection system uses text vectorization with Tf-idf (Term Frequency-Inverse Document Frequency) to transform input text data into numerical features. Four classifiers are employed for comparison:

- Logistic Regression
- Multinomial Naive Bayes
- Decision Tree Classifier
- Linear Support Vector Classifier (LinearSVC)

The system trains these classifiers on a labeled dataset and evaluates their performance on validation data. The best-performing classifier is then used to predict labels for test data.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/spam-detection.git
   cd spam-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure Python 3.x is installed. Install additional packages if necessary.

## Usage

To train and evaluate the spam detection system:

```bash
python spam_detector.py
```

Replace `spam_detector.py` with the actual script name where your main function (`spam_detector`) is implemented.

## File Descriptions

- `spam_detector.py`: Main script to train classifiers and evaluate performance.
- `README.md`: This file, providing an overview of the project.
- `requirements.txt`: List of Python dependencies.
