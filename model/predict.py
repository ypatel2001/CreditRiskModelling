import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os
import json
import sys

def load_model():
    """Load model, scaler, and feature names atomically"""
    print("Loading model files...", file=sys.stderr)
    try:
        model = joblib.load('model.joblib')
        scaler = joblib.load('scaler.joblib')
        feature_names = joblib.load('feature_names.joblib')
        print(f"Loaded model with {len(feature_names)} features:", ", ".join(feature_names), file=sys.stderr)
        return model, scaler, feature_names
    except Exception as e:
        print(f"Error loading model: {str(e)}", file=sys.stderr)
        sys.exit(1)

def predict(model, scaler, feature_names, input_data):
    """Process single prediction request"""
    print("Processing prediction request:", input_data, file=sys.stderr)
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data], columns=feature_names)
        print("\nDataFrame shape:", df.shape, file=sys.stderr)
        print("DataFrame columns:", df.columns.tolist(), file=sys.stderr)
        # Handle missing values
        null_counts = df.isnull().sum()
        if null_counts.any():
            print("\nMissing values detected:", null_counts, file=sys.stderr)
            df = df.fillna(df.mean(numeric_only=True))
            df = df.fillna(df.mode().iloc[0])
        # Scale features
        scaled_features = scaler.transform(df)
        print("\nScaled features shape:", scaled_features.shape, file=sys.stderr)
        # Get prediction
        probability = model.predict_proba(scaled_features)[0][1]
        percentage = probability * 100
        formatted_probability = f"{percentage:.2f}%"  # Format to 2 decimal places with %
        result = {"probability": formatted_probability}
        print("\nPrediction result:", result, file=sys.stderr)
        return result
    except Exception as e:
        print(f"\nError during prediction: {str(e)}", file=sys.stderr)
        return None

def main():
    """Main prediction function"""
    model, scaler, feature_names = load_model()
    # Read input from stdin
    try:
        input_data = json.load(sys.stdin)
        print("\nReceived input data:", input_data, file=sys.stderr)
        result = predict(model, scaler, feature_names, input_data)
        # Output result to stdout
        if result is None:
            print(json.dumps({"error": "prediction failed"}))
        else:
            print(json.dumps(result))
    except json.JSONDecodeError:
        print("Invalid JSON input", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()