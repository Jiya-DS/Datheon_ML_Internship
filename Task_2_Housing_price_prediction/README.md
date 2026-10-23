# California Housing Price Prediction (Linear / Ridge / Lasso)

**Task 2 | Datheon Machine Learning & Data Science Internship**

## Project Overview

This project implements an end-to-end supervised machine learning regression pipeline using the California Housing dataset. The objective is to predict median housing values for California districts based on demographic, geographic, and housing-related features.

Three machine learning algorithms, **Linear Regression, Ridge Regression, and Lasso Regression**, are trained and evaluated to compare their prediction performance.

The project also uses **GridSearchCV** to tune the regularization parameters for Ridge and Lasso Regression.

## Dataset

The California Housing dataset contains **20,640 records** and information about housing characteristics across California districts.

The dataset includes the following features:

| Feature | Description |
|---|---|
| Longitude | Geographic longitude of the district |
| Latitude | Geographic latitude of the district |
| Housing Median Age | Median age of houses in the district |
| Total Rooms | Total number of rooms |
| Total Bedrooms | Total number of bedrooms |
| Population | District population |
| Households | Number of households |
| Median Income | Median household income |
| Ocean Proximity | Category describing proximity to the ocean |
| Median House Value | Target variable |

The dataset contains missing values in `total_bedrooms`, which are handled through a preprocessing pipeline using median imputation for numerical features.

**Target variable:** `median_house_value`

The target represents median housing values for districts, not the price of individual houses.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib
- Jupyter Notebook

## Project Workflow

1. **Data Loading:** Load the California Housing dataset using Pandas.
2. **Exploratory Data Analysis:** Examine dataset dimensions, data types, descriptive statistics, and missing values.
3. **Data Visualization:** Analyze relationships between numerical features using a correlation heatmap.
4. **Feature Engineering:** Create rooms-per-household, bedrooms-per-room, and population-per-household features.
5. **Data Preprocessing:** Split the dataset into training and testing sets, impute missing values, scale numerical features, and encode categorical features.
6. **Model Training:** Train Linear Regression, Ridge Regression, and Lasso Regression models.
7. **Hyperparameter Tuning:** Use GridSearchCV to identify regularization parameters for Ridge and Lasso Regression.
8. **Model Evaluation:** Calculate RMSE, MAE, and R² scores on the test dataset.
9. **Visualization:** Generate model performance comparisons, actual-versus-predicted plots, and residual analysis plots.
10. **Model Comparison:** Compare the predictive performance of all three regression models.

## Machine Learning Models

### Linear Regression

Linear Regression establishes a baseline by learning a linear relationship between the input features and median housing values.

### Ridge Regression

Ridge Regression applies **L2 regularization** to penalize large coefficients. This can help control model complexity and reduce overfitting.

### Lasso Regression

Lasso Regression applies **L1 regularization**, which can shrink some feature coefficients to zero and support feature selection.

GridSearchCV is used to tune the regularization parameter for Ridge and Lasso Regression.

## Model Performance

All three models were evaluated using the same test dataset.

| Model | RMSE | MAE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 69,127.04 | 49,645.49 | 0.6353 |
| Ridge Regression | 69,134.97 | 49,652.14 | 0.6353 |
| Lasso Regression | 69,128.09 | 49,646.43 | 0.6353 |

**Linear Regression achieved the lowest RMSE and MAE among the three evaluated models.**

The R² score of approximately **0.6353** indicates that the model explains about 63.53% of the variation in median housing values in the test dataset.

The performance differences between the three models were small.

RMSE and MAE are measured in the dataset's housing-value units, while R² is a unitless metric.

## Evaluation Metrics

- **Root Mean Squared Error (RMSE):** Measures prediction error while penalizing larger errors more heavily.
- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted housing values.
- **R² Score:** Measures the proportion of variation in the target variable explained by the model.

Lower RMSE and MAE indicate smaller prediction errors, while a higher R² score generally indicates a better fit.

## Visualizations

### 1. Correlation Heatmap

The correlation heatmap illustrates relationships between numerical features and the target variable. It helps identify features associated with median housing values.

### 2. Model Performance Comparison

Bar charts compare Linear Regression, Ridge Regression, and Lasso Regression using RMSE, MAE, and R² scores.

### 3. Actual vs. Predicted Values

Scatter plots compare actual housing values with model predictions. A diagonal reference line represents perfect predictions.

### 4. Residual Analysis

Residual distribution plots and residuals-versus-predicted plots help examine prediction errors, identify potential patterns, and understand model limitations.

The visualizations are generated within the Jupyter Notebook.

## Key Findings

- Linear Regression achieved the lowest RMSE and MAE among the three models.
- All three models produced very similar performance results.
- The Linear Regression model achieved an R² score of approximately **0.6353**.
- Median income and geographic characteristics can provide useful information for predicting housing values.
- Ridge and Lasso regularization did not produce a meaningful performance improvement over the baseline Linear Regression model in this experiment.
- Residual analysis helps identify prediction patterns that may not be captured by linear models.

## Project Structure

```text
Task_2_Housing_price_prediction/
│
├── README.md
├── housing_prediction.ipynb
└── california_housing.csv  
```

The CSV file is excluded from the Git repository through `.gitignore`. The dataset must be obtained separately and placed in the project directory before running the notebook.

## How to Run

1. Download or clone the repository.
2. Obtain the California Housing CSV dataset and place it in the same directory as the notebook.
3. Install the required libraries:

   ```bash
   python -m pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

4. Open `housing_prediction.ipynb` in Jupyter Notebook or VS Code.
5. Select a Python kernel with the required libraries installed.
6. Ensure the dataset filename is `california_housing.csv`.
7. Restart the kernel and run all notebook cells to reproduce the results and visualizations.

## Limitations and Future Improvements

- The models do not explain all variation in housing values.
- Linear regression models may not capture complex nonlinear relationships.
- Additional feature engineering and nonlinear algorithms, such as Random Forest Regression, could be explored.
- The dataset reflects historical housing information and should not be treated as a source of current market prices.
- Further validation is needed before applying these models to real-world housing decisions.

## Conclusion

This project demonstrates a complete machine learning regression workflow, including exploratory data analysis, feature engineering, preprocessing, model training, hyperparameter tuning, evaluation, and visualization.

Linear Regression, Ridge Regression, and Lasso Regression achieved similar performance on the test dataset. Linear Regression performed best by a small margin, with an RMSE of **69,127.04**, an MAE of **49,645.49**, and an R² score of **0.6353**.

The results highlight the importance of evaluating multiple models and selecting an approach based on measured performance rather than assumptions.

---

**Datheon Machine Learning & Data Science Internship | Task 2**