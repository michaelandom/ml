import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.ensemble import BaggingClassifier
import joblib


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.model = LogisticRegression()  
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        y_pred = self.model.predict(X)
        return (y_pred > self.threshold).astype(int)
    
    def predict_proba(self, X):
        y_pred = self.model.predict(X)
        probs = np.zeros((len(y_pred), 2))
        probs[:, 1] = y_pred
        probs[:, 0] = 1 - y_pred
        return np.clip(probs, 0, 1)

def save_images(model_name, image_name, figure):
    folder_name = f'./{model_name.replace(" ", "_").lower()}'
    os.makedirs(folder_name, exist_ok=True)
    figure_path = os.path.join(folder_name, image_name)
    figure.savefig(figure_path)
    #logger.info(f"Saved {image_name} image for {model_name} to {figure_path}")

def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    save_images(model_name, f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png", plt)

def plot_roc_curve(y_test, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    save_images(model_name, f"roc_curve_{model_name.replace(' ', '_').lower()}.png", plt)

def plot_precision_recall_curve(y_test, y_prob, model_name):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.tight_layout()
    save_images(model_name, f"precision_recall_curve_{model_name.replace(' ', '_').lower()}.png", plt)

def plot_learning_curve(pipeline, X_train, y_train, model_name):
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline, X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, color='blue', marker='o', label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', marker='s', label='Cross-validation score')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')
    plt.title(f'Learning Curve - {model_name}')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    save_images(model_name, f"learning_curve_{model_name.replace(' ', '_').lower()}.png", plt)

def evaluate_model(model_name, model, X_train, X_test, y_train, y_test, preprocessor):
    logger.info(f"\n{'-'*50}\nEvaluating {model_name}\n{'-'*50}")
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    try:
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    except Exception as e:
        logger.warning(f"Cross-validation failed: {str(e)}")
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:\n")
    logger.info(classification_report(y_test, y_pred))
    
    plot_confusion_matrix(y_test, y_pred, model_name)
    
    try:
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_prob, model_name)
        plot_precision_recall_curve(y_test, y_prob, model_name)
    except Exception as e:
        logger.warning(f"ROC/PR curve generation failed: {str(e)}")
    
    plot_learning_curve(pipeline, X_train, y_train, model_name)
    
    return pipeline, accuracy

def train_final_model_with_test_data(model_name,model,X,Y,preprocessor):
    logger.info(f"\n{'-'*50}\nTraining final {model_name}\n{'-'*50}")
    pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
    ])
    pipeline.fit(X, Y)
    return pipeline

def distribution_chart(data,numerical_cols):
    for col in numerical_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        save_images("data/distribution", f"distribution_of_{col}.png", plt)

def correlation(data):
    numerical_data = data.select_dtypes(include=['number']) 
    correlation_matrix = numerical_data.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    save_images("data", f"correlation_heatmap.png", plt)


def vs_loan_status(data,numerical_cols):

    for col in numerical_cols:
        if col != 'loan_status':
            plt.figure(figsize=(10, 8))
            sns.boxplot(x='loan_status', y=col, data=data)
            plt.title(f'Loan Amount vs. {col}')
            save_images("data/loan_status", f"loan_status_vs_{col}.png", plt)

def pairplot(data):
    plt.figure(figsize=(10, 8))
    sns.pairplot(data[['person_age', 'loan_amnt', 'person_income', 'credit_score', 'loan_status']], hue='loan_status')
    save_images("data", f"pairplot.png", plt)

def other(data):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='person_income', y='loan_amnt', data=data)
    plt.title('Loan Amount vs. Person Income')
    save_images("data", f"loan_amount_vs_person_income.png", plt)
    plt.figure(figsize=(10, 8))
    sns.boxplot(x='loan_status', y='credit_score', data=data)
    plt.title('Credit Score vs. Loan Status')
    save_images("data", f"credit_score_vs_loan_status.png", plt)
    plt.figure(figsize=(10, 8))
    pd.crosstab(data['loan_status'], data['loan_intent']).plot(kind='bar', stacked=True)
    plt.title('Loan Intent vs. Loan Status')
    save_images("data", f"loan_intent_vs_loan_status.png", plt)
    plt.figure(figsize=(10, 8))
    sns.boxplot(x='previous_loan_defaults_on_file', y='cb_person_cred_hist_length', data=data)
    plt.title('Credit History Length vs. Loan Defaults')
    save_images("data", f"credit_history_length_vs_loan_defaults.png", plt)
    plt.figure(figsize=(10, 8))
    pd.crosstab(data['loan_status'], data['person_home_ownership']).plot(kind='bar', stacked=True)
    plt.title('Loan Status by Home Ownership')
    save_images("data", f"loan_status_by_home_ownership.png", plt)
    plt.figure(figsize=(10, 8))
    sns.histplot(data['loan_percent_income'], kde=True)
    plt.title('Loan Amount as Percentage of Income')
    save_images("data", f"loan_amount_as_percentage_of_income.png", plt)










def load_data(file_path):
    data = pd.read_csv(file_path)
    
    data_quality(data)



    X = data.drop('loan_status', axis=1)
    y = data['loan_status']
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    distribution_chart(data,numerical_cols)
    correlation(data)
    vs_loan_status(data,numerical_cols)
    pairplot(data)
    other(data)
    
    return X, y, preprocessor

def data_quality(data):
    print("Dataset shape:", data.shape)
    
    print("\nData types:")
    print(data.dtypes)


    print("\nMissing values:")
    print(data.isnull().sum())

    print("\nDuplicate rows:", data.duplicated().sum())

def main():
    file_path = 'loan_data.csv'
    X, y, preprocessor = load_data(file_path)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Gaussian Naive Bayes': GaussianNB(),
        # 'SVM (linear)': SVC(kernel='linear', probability=True, random_state=42),
        # 'SVM (rbf)': SVC(kernel='rbf', probability=True, random_state=42),
        # 'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        # 'KNN (k=10)': KNeighborsClassifier(n_neighbors=10),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        # 'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        # 'Voting Classifier': VotingClassifier(estimators=[
        #      ('lr', LogisticRegression(random_state=42)),
        #      ('rf', RandomForestClassifier(random_state=42)),
        #      ('gb', GradientBoostingClassifier(random_state=42))
        #  ], voting='soft'),
        #  'Bagging' : BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=42)
    }
    

    
    results = {}
    for name, model in models.items():
        try:
            pipeline, accuracy = evaluate_model(name, model, X_train, X_test, y_train, y_test, preprocessor)
            results[name] = {'pipeline': pipeline, 'accuracy': accuracy}
        except Exception as e:
            logger.error(f"Error evaluating {name}: {str(e)}")
    
    if results:
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        best_model = results[best_model_name]['pipeline']
        best_accuracy = results[best_model_name]['accuracy']
        logger.info(f"\n{'-'*50}\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}\n{'-'*50}")
        final_pipeline = train_final_model_with_test_data(best_model_name, best_model.named_steps['model'], X, y, preprocessor)
        joblib.dump(final_pipeline, f'best_model.pkl')
        logger.info(f"Saved the best model: {best_model_name}")
if __name__ == '__main__':
    main()