# Machine Learning Project: Avocado Price Prediction 

![giphy](https://user-images.githubusercontent.com/71875232/156922109-2141a048-ad62-4006-94c8-7495678d5443.gif)

## Table of Contents
1. [General Info](#General-Info)
2. [Installation](#Installation)
3. [Demo](#Demo)
4. [Usage and Main Functionalities](#Usage-and-Main-Functionalities)
5. [Contributing](#Contributing)


## General Info
This is a compact Machine Learning and Data Visualization project I worked on for fun and to acquire some initial knowledge of Scikit-learn. 

I was able to familiarize myself with: 
- Performing train_test_split to divide the dataset into train and test
- Visualizing the result using graphs
- Applying Random Forest for price prediction 
- Using groupby function for combined analysis of variables

The dataset I used for the analytics task was retrieved from kaggle: https://www.kaggle.com/valentinjoseph/avocado-sales-20152021-us-centric


## Installation

**Requirements:** 
Make sure you have Python 3.7+ installed on your computer. You can download the latest version of Python [here](https://www.python.org/downloads/). 

**Req. Packages:**
* pandas
* copy
* sklearn
* dash
* dash_bootstrap_components
* ploty.express


## Demo

https://user-images.githubusercontent.com/71875232/156922122-741e31f7-2134-4cf9-9b77-40515204ae49.mov


## Usage and Main Functionalities

#### 0. Data Preprocessing
* This part cleans up the original data and prepares it for analysis. 
* Deep copy original data table 
* Splitting table columns 
* Checking for correct notation
* Label encoding

#### 1. Exploratory Analysis
* This part of the code is about analyzing the data. 
* Identifying any overarching trend in data over time
* Identifying any repetitive, seasonal patterns in the data
* Creating different line and bar charts for data visualization  

<img width="664" alt="Bildschirmfoto 2022-03-06 um 13 02 07" src="https://user-images.githubusercontent.com/71875232/156922200-6f169d16-fe3d-47e2-97f8-1de43cd0a824.png">

#### 2. Building a forecast model
* Split data into test and train set 
* Use Random Forest Regressor to forecast data 

#### 3. Evaluation Forecast Model
* Calculation of MAE, truncate the float 
* Clean up the result table 
* Creating plot comprising the actual values and forecast

<img width="702" alt="Bildschirmfoto 2022-03-06 um 13 02 25" src="https://user-images.githubusercontent.com/71875232/156922218-78c37e8d-d3ce-4d60-bfc2-d2dc8ced3638.png">

#### 4. Dash App Layout
* plotly's Dash is now used to create an interactive dashboard 
* The individual graphics and texts are arranged in rows and containers. 
* This part also includes a dropdown menu that the user can interact with. 

#### 3. App Callback 
* Here we connect an interactive bar chart to the Dash Components. The chart is filterable by year. 


## Contributing 
Your comments, suggestions, and contributions are welcome. 
Please feel free to contribute pull requests or create issues for bugs and feature requests.

