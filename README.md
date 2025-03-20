# Loan Approval Prediction Model

## Overview
This project builds and evaluates multiple machine learning models to predict loan approval status based on various customer attributes. The best-performing model is selected, trained on the entire dataset, and saved for production use.

## Features
- Loads and preprocesses loan dataset (`loan_data.csv`).
- Trains multiple machine learning models, including:
  - Logistic Regression
  - Naive Bayes
  - SVM (Linear & RBF)
  - K-Nearest Neighbors
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - AdaBoost
  - Voting Classifier
  - Bagging Classifier
- Evaluates models using accuracy, confusion matrix, ROC curves, and precision-recall curves.
- Selects the best model based on accuracy and retrains it on the entire dataset for production.
- Saves the best model using `joblib`.

## Prerequisites
Ensure you have Python installed and the required dependencies. Install them using:
```bash
pip install -r requirements.txt
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository.git
   cd loan-approval-model
   ```
2. Ensure the dataset (`loan_data.csv`) is available in the project directory.
3. Install dependencies as mentioned above.

## Usage
Run the `main.py` script to train and evaluate models:
```bash
python main.py
```

The script will:
- Preprocess data (handling missing values, encoding categorical features, and scaling numerical features).
- Train and evaluate models.
- Save the best model for production.

## Model Selection & Training
After evaluating all models, the best one (based on accuracy) is selected and retrained on the full dataset using:
```python
final_pipeline = train_final_model_with_test_data(best_model_name, best_model.named_steps['model'], X, y, preprocessor)
```
The model is saved as a `.pkl` file for future predictions:
```bash
best_model.pkl
```

## Data Visualization
The script generates several visualizations for data analysis:
- Distribution of numerical features.
- Correlation heatmap.
- Loan status vs. numerical features.
- Pair plots and scatter plots.

## Logs
The script logs all key steps and model performances using Python's `logging` module.

## Dependencies
The project requires the following libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `joblib`
- `os`, `logging`

## License
This project is open-source and available under the MIT License.

## Author
Michael 

---

