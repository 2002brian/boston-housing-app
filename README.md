# Boston Housing ML Model Evaluation Dashboard

This project is an interactive R Shiny dashboard for training, evaluating, and comparing three different machine learning models on the Boston housing dataset.

## How to Run the Application

1.  **Prerequisites**: Make sure you have R and RStudio installed.
2.  **Install Packages**: Open your R console and run the following command to install all the necessary packages:
    ```R
    install.packages(c("shiny", "shinydashboard", "dplyr", "ggplot2", "caret", "randomForest", "xgboost", "MASS"))
    ```
3.  **Launch the App**:
    *   Set your working directory to the root of this project folder.
    *   Run the following command in your R console:
    ```R
    shiny::runApp()
    ```
    The application should launch in a new window or in your web browser.

## Understanding the Evaluation Metrics

The dashboard displays two key metrics to evaluate model performance on the test set:

### R-squared (R²)

*   **What it is**: R-squared, also known as the coefficient of determination, is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model.
*   **Interpretation**:
    *   It is a value between 0 and 1 (or 0% and 100%).
    *   A value of **1** indicates that the model perfectly predicts the dependent variable's values.
    *   A value of **0** indicates that the model fails to explain any of the variability of the target variable.
*   **Goal**: A higher R-squared value is generally better, as it indicates that the model explains a larger portion of the variance in the housing prices.

### Root Mean Squared Error (RMSE)

*   **What it is**: RMSE is the standard deviation of the residuals (prediction errors). Residuals are a measure of how far from the regression line data points are; RMSE is a measure of how spread out these residuals are. In other words, it tells you how concentrated the data is around the line of best fit.
*   **Interpretation**:
    *   It is an absolute measure of fit, measured in the same units as the target variable (in this case, thousands of dollars, since `Price` is in $1000s).
    *   A value of **0** would mean the model's predictions are perfect.
*   **Goal**: A lower RMSE value is better, as it indicates that the model's predictions are, on average, closer to the actual values.
# boston-housing-app
# boston-housing-app
# boston-housing-app
