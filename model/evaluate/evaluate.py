import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, f1_score, accuracy_score, confusion_matrix

# Define the path to your training data
DATA_PATH = '../../data/train.csv'
TARGET_COLUMN = 'credit_card_default'

try:
    # Load the training data
    df = pd.read_csv(DATA_PATH)

    # Separate features and target
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    # Identify categorical and numerical features (simplified)
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing numerical values with the mean
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing categorical values with the most frequent
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    # Create a pipeline with preprocessing and a Logistic Regression model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression(solver='liblinear', random_state=42))])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Print evaluation metrics
    print("=" * 50)
    print("Confusion Matrix (Test Data):")
    print(confusion_matrix(y_test, y_pred))
    print("=" * 50)
    print("Classification Report (Test Data):")
    print(classification_report(y_test, y_pred))
    print("=" * 50)
    print(f"Accuracy of TEST data: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score (Macro) of TEST data: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"ROC AUC of TEST data: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("=" * 50)

except FileNotFoundError:
    print(f"Error: Data file not found at '{DATA_PATH}'")
except Exception as e:
    print(f"An error occurred: {e}")