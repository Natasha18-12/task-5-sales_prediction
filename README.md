Sales Prediction
This project focuses on predicting product sales based on advertising spending across different media channels. The analysis uses a linear regression model to determine the relationship between advertising budgets and sales, providing a valuable tool for optimizing marketing strategies.

Project Structure
sales_prediction.py: This Python script contains the complete workflow for data loading, analysis, model training, and performance evaluation.

Advertising.csv: The dataset containing advertising expenditures for TV, Radio, and Newspaper along with the corresponding Sales figures.

Key Features
Data Loading and Exploration: The script loads the advertising dataset and performs initial data exploration to understand its structure, check for missing values, and view descriptive statistics.

Data Visualization:

Pairplot: A pairplot visualizes the relationships between sales and each advertising channel (TV, Radio, Newspaper). This helps to visually identify which channels have a stronger correlation with sales.

Correlation Heatmap: A heatmap displays the correlation matrix, providing a numerical measure of the linear relationship between all variables. This helps confirm which advertising medium is most strongly correlated with sales.

Model Training: A Linear Regression model is trained to predict sales based on the advertising budgets. This model identifies the best-fit line to describe the relationship between the variables.

Model Evaluation: The model's performance is evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared. These metrics provide a quantitative measure of the model's accuracy and predictive power.

Dataset
The Advertising.csv dataset contains 200 data points with four key columns:

TV: Advertising budget spent on TV.

Radio: Advertising budget spent on Radio.

Newspaper: Advertising budget spent on Newspapers.

Sales: The number of units sold.

Results
The sales_prediction.py script outputs the trained linear regression coefficients, along with key performance metrics. The R-squared value indicates how well the advertising variables explain the variance in sales. A high R-squared value suggests that the model is effective at predicting sales based on advertising expenditure.
