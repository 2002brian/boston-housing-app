# 🏡 Boston Housing Price Prediction AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://boston-housing-app-be74qvyztoprzwrfrwcjdv.streamlit.app/)
This is an interactive web application for predicting Boston housing prices using advanced Machine Learning models. Built with **Python** and **Streamlit**.

## 🌟 Features
- **AutoML Leaderboard:** Compare Linear Regression, Random Forest, XGBoost, LightGBM, and CatBoost performance instantly.
- **Explainable AI (XAI):** Integrated **SHAP** and **LIME** for model transparency. Understand *why* the model made a specific prediction.
- **What-If Simulator:** Interactive sidebar to adjust house features and see real-time price predictions.
- **Deep Learning:** Includes a PyTorch Neural Network demo.

## 🏠 Input Features Explained (Sidebar Variables)

Use the sidebar to adjust these parameters and simulate different housing conditions:

| Feature | Description | Impact on Price (General) |
| :--- | :--- | :--- |
| **CRIM** | Per capita crime rate by town. | 📉 Higher crime → Lower price |
| **ZN** | Proportion of residential land zoned for lots over 25,000 sq.ft. | 📈 More residential zone → Higher price |
| **INDUS** | Proportion of non-retail business acres per town. | 📉 More industry → Lower price |
| **CHAS** | Charles River dummy variable (1 if tract bounds river; 0 otherwise). | 📈 River view → Higher price |
| **NOX** | Nitric oxides concentration (parts per 10 million). | 📉 High pollution → Lower price |
| **RM** | Average number of rooms per dwelling. | 📈 More rooms → Higher price |
| **AGE** | Proportion of owner-occupied units built prior to 1940. | 📉 Older buildings → Lower price |
| **DIS** | Weighted distances to five Boston employment centres. | 📈 Closer to centers is usually better, but DIS measures distance to work. |
| **RAD** | Index of accessibility to radial highways. | 📈 Better transport access → Higher price |
| **TAX** | Full-value property-tax rate per $10,000. | 📉 Higher tax → Lower price |
| **PTRATIO**| Pupil-teacher ratio by town. | 📉 More students per teacher → Lower price |
| **LSTAT** | % lower status of the population. | 📉 Higher % lower status → Lower price |

## 🚀 How to Run Locally

If you want to run this app on your own machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/2002brian/boston-housing-app.git
   cd boston-housing-app

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Run the app:**
   ```bash
   streamlit run app.py


## 📊 Understanding the Evaluation Metrics

The dashboard displays two key metrics to evaluate model performance on the test set:

1. **R-squared (R²)**

-  What it is: A statistical measure representing the proportion of variance for a dependent variable explained by the independent variables.

-  Interpretation:

-   Value is between 0 and 1.

-   1 means perfect prediction.

-   0 means the model fails to explain any variability.

- Goal: Higher is better.

2. **Root Mean Squared Error (RMSE)**

- What it is: The standard deviation of the residuals (prediction errors). It tells you how concentrated the data is around the line of best fit.

-  Interpretation:

-   Measured in the same units as the target variable ($1000s).

-   0 would mean perfect predictions.

- Goal: Lower is better.

Created by Brian Wu