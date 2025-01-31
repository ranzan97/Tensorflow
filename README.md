# TensorFlow Examples

This repository contains Jupyter notebooks demonstrating machine learning techniques using TensorFlow. These examples are designed to help you understand and implement various models, including linear regression, time series prediction, and sentiment analysis.

## Contents

1. [Simple Linear Regression](Simple_Linear_Regression.ipynb)
2. [Multiple Variable Linear Regression](Multi_Variable_Linear_Regression.ipynb)
3. [Stock_Price_Prediction_LSTM_CNN_Tensorflow.ipynb](Stock_Price_Prediction_LSTM_CNN_Tensorflow.ipynb)
4. [Simple_RNN_Tensorflow.ipynb](Simple_RNN_Tensorflow.ipynb)
5. [Sentiment Analysis Model](Sentiment_analysis_model.ipynb)
6. [California_Housing_Regression](California_Housing_Regression.ipynb)
7. [Transformer_TimeSeries_Forecasting.ipynb](Transformer_TimeSeries_Forecasting.ipynb)


### Simple_Linear_Regression.ipynb

This notebook covers the basics of simple linear regression using TensorFlow. It demonstrates how to:
- Prepare and preprocess data for a simple linear regression model
- Build and train a TensorFlow model for simple linear regression
- Evaluate the model's performance
- Make predictions using the trained model

### Multi_Variable_Linear_Regression.ipynb

This notebook extends the concept to multiple variable (multivariate) linear regression. It includes:
- Handling datasets with multiple features
- Scaling and normalizing multi-dimensional data
- Implementing a TensorFlow model for multiple variable linear regression
- Training and evaluating the multivariate model
- Using the model for predictions with multiple input variables

### Stock_Price_Prediction_LSTM_CNN_Tensorflow.ipynb

This notebook demonstrates time series forecasting for stock price prediction using TensorFlow. It covers how to:
- Acquire and preprocess historical stock price data
- Build and train a hybrid CNN-LSTM model for sequence learning
- Evaluate the model's performance using loss and metrics
- Visualize actual vs. predicted stock prices over time

### Simple_RNN_Tensorflow.ipynb

- This project demonstrates a basic implementation of a Recurrent Neural Network (RNN) using TensorFlow for sequence prediction.
- The model is trained on a simple sequence of numbers and can predict the next number in the sequence.

### Sentiment Analysis Model.ipynb

This script implements a basic sentiment analysis model using TensorFlow and Keras. It demonstrates:
- Text tokenization and sequence preprocessing
- Building a neural network with an embedding layer
- Training a binary sentiment classification model
- Predicting sentiment for new text inputs

### California Housing Regession.ipynb

This notebook showcases a regression model for predicting housing prices in California using TensorFlow and the California Housing dataset2. It covers:
- Loading and preprocessing the California Housing dataset
- Building a neural network model for regression
- Training and evaluating the model on housing price prediction
- Visualizing the results and model performance

### Transformer_TimeSeries_Forecasting.ipynb
The goal of this project is to predict future passenger counts based on historical data, showcasing how Transformers can be applied to sequential data like time series. The implementation includes:
- Data preprocessing (normalization, sequence creation).
- A custom Transformer architecture with multi-head attention and feed-forward layers.
- Model training and evaluation using Mean Squared Error (MSE) and Mean Absolute Error (MAE).
- Visualization of training progress and predictions.
