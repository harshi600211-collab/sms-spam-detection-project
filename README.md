# SMS Spam Detection Using Machine Learning

## Project Description
This project classifies SMS messages as spam or ham (legitimate messages) using machine learning and natural language processing.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes

## Features
- Processes SMS text messages
- Converts text into numerical features using TF-IDF
- Trains a Naive Bayes classification model
- Evaluates model performance
- Predicts whether a new SMS is spam or ham

## Dataset
The project expects a file named `spam.csv` containing SMS messages and their labels. Supported column names are `v1` and `v2`, or `label` and `message`.

## Installation
Install dependencies using:

`pip install -r requirements.txt`

## How to Run
Place the dataset in the project folder and execute:

`python sms_spam_detection.py`

## Disclaimer
This project is intended for educational purposes. Model performance depends on the dataset and evaluation method.
