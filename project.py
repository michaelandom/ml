import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('loan_data.csv')  

def clean_data(df):
    df = df.copy()
    
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].str.strip()
    
    df = df.replace('?', np.nan)
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    
    return df

def preprocess_data(df):
    df_processed = df.copy()  
    cat_cols = df_processed.select_dtypes(['object']).columns
    num_cols = df_processed.select_dtypes(['int64', 'float64']).columns.drop('loan_status')
    
    le = LabelEncoder()
    for col in cat_cols:
        df_processed[col] = le.fit_transform(df_processed[col])
    
    scaler = StandardScaler()
    df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])
    
    return df_processed

data = clean_data(data)
processed_data = preprocess_data(data)

X = processed_data.drop('loan_status', axis=1)
y = processed_data['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
    'SVM': SVC(kernel='rbf')
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

print("\nModel Comparison:")
for name, accuracy in results.items():
    print(f"{name}: {accuracy:.4f}")


