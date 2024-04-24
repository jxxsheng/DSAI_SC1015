<h1>Hotel Reservation No-Show Prediction</h1>

<h2>Introduction</h2>

Hotel no-shows result in missed revenue opportunities and increased costs due to resource wastage. When guests fail to show up for their reservations, the hotel loses out on potential revenue from the unoccupied rooms. Additionally, the hotel has already incurred costs associated with room preparations, such as cleaning and maintenance. These resources could have been utilised more effectively if the room was occupied by a paying guest.

<h2>Problem Definition</h2>

How could we predict and minimise the number of no-shows in order to maintain and increase profits?

<h2>Flow of Pipeline:</h2>

1. Data Cleaning: Remove rows with missing/abnormal values, standardise data units/formats, and handle data type conversion
2. Exploratory Data Analysis: Examine categorical variables using countplot and numerical variable using histogram
3. Visualisation: Create visual plots to identify variables with potential relationships with the target variable 'no_show'
4. Hypothesis Testing: Use Cramer's V and Chi-squared test to determine significant relationships between variables and 'no_show'
5. Model Training: Train machine learning models (Logistic Regression, Decision Tree, Random Forest, RNN, CNN)
6. Model Evaluation: Evaluate model performance using accuracy, precision, recall, F1 score, ROC-AUC and confusion matrix metrics

<h2>Overview of EDA:</h2>

Our target variable is 'no_show', with '1' indicating No Show and '0' indicating Show. A histogram was used to show the distribution of prices (numerical) for Show and No Show. There is no noticeable relationship between 'price' and 'no_show' as the distributions are quite similar. For other variables (categorical), a countplot was used to compare the proportion of No Show to Show. A higher variation of No Show to Show across the different features in the variable would thus indicate a relationship with 'no_show'. From the observation, we find that 'branch', 'country', 'first_time', 'room', and 'arrival_month' have a significant relationship with 'no_show'. We further confirm the relationship using Cramer's V and Chi-squared test. Therefore, these variables/predictors will be used to train the Machine Learning Models.

<h2>Features Processing:</h2>

| Variables                             | How is it Processed                                                                                |
|-----------------------------------------------------|-----------------------------------------------------------------------------------------------|
| num_adults| Convert the 'num_adults' variable from string to numeric data type|
| price| Convert all prices in the 'price' column from SGD and USD to SGD only, and convert to float|
| checkout_day| Remove rows where the 'checkout_day' variable has negative values|
| arrival_month| Convert all the months in 'arrival_month' to lowercase and capitalise the first letter|
|branch, country, first_time, room, arrival_month| Map categorical variables 'branch', 'country', 'first_time', 'room', and 'arrival_month' to numerical labels for machine learning modelling|

<h2>Machine Learning Models:</h2>

All 5 Machine Learning Models are evaluated using the Confusion Matrix, which uses the Test dataset to test the accuracy of the model  (i.e. to determine the number of entries that were predicted correctly and wrongly using the model). The accuracy of the model is measured by taking (True Positive + True Negative) / Total number of entries. Below is a summary of the model description and results.

<h4>Model 1: Binary Logistic Regression</h4>

A Logistic Regression model is suitable as it takes in two or more predictor variables to predict the outcome of the categorical target variable 'no_show'. In this case, the predictor variables are all categorical.

Accuracy: 72% 
<br>
ROC-AUC: 0.735

<h4>Model 2: Multi-Variate Decision Tree</h4>

A Decision Tree is suitable as it provides a clear and interpretable structure that shows how decisions are made based on the predictor variables. This can help us understand the factors influencing the target variable and determine the predictors that have a greater impact on the target variable 'no_show'. 

Accuracy: 70%
<br>
ROC-AUC: 0.713

<h4>Model 3: Random Forest Classification</h4>

Random Forest can capture non-linear relationships between predictor variables and the target variable 'no_show' by combining multiple decision trees, each trained on a random subset of the data and variables. This ensemble approach helps reduce overfitting and improves generalisation performance compared to a single decision tree. It can also take in more significant predictor variables to produce a higher accuracy.

Accuracy: 77%
<br>
ROC-AUC: 0.827

<h4>Model 4: Recurrent Neural Network</h4>

RNNs are a type of neural network designed to handle sequential data, where the order of data points is crucial. In this context, sequential data could include the booking history of a guest or the sequence of events leading up to a reservation. RNNs are well-suited for capturing temporal dependencies in such data, as they maintain an internal state or memory that allows them to process sequences of inputs. This memory mechanism enables RNNs to learn from past events and make predictions based on the sequence of data points.

Accuracy: 72%
<br>
ROC-AUC: 0.715

<h4>Model 5: Convolutional Neural Network (CNN) and LSTM Hybrid</h4>

The hybrid CNN-LSTM model can analyze both spatial features (e.g., room types, branch locations) and temporal patterns (e.g., booking history, lead times) to make predictions about no-shows. The CNN component can extract spatial features from categorical variables, while the LSTM component can capture temporal dependencies in the sequential booking data. This combination allows the model to leverage the strengths of both architectures for improved prediction accuracy.

Accuracy: 70%
<br>
ROC-AUC: 0.747

<h2>Conclusion</h2>

The Random Forest model produces the highest accuracy, ROC-AUC and F1 Score across all models. Since the model is less prone to overfitting, it can take in more predictor variables to generate a prediction with higher accuracy and performance. Therefore, we can use this model to predict whether a customer will show up for his hotel reservation. In order to reduce expenses incurred due to No-Shows, the hotel chain could consider implementing a deposit fee that is non-refundable on customers which the Random Forest model predicts not showing up.

<h2>What we have learned</h2>

- Handling and cleaning imbalanced datasets
- RNN and CNN with LSTM
- Collaborating using GitHub
- Concepts about Accuracy, Precision, Recall, F1 SCore, ROC-AUC for model evaluation
- Understanding and applying the Data Science pipeline

<h2>Contributors</h2>

- @chenglin2003 (Models and research)
- @izackyy (Models and research)
- @jxxsheng (EDA and research)


