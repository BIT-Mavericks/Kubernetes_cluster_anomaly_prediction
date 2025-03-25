import pandas as pd
import numpy as np
import joblib
import os

# For ANN (Keras) models, attempt to import load_model
try:
    from tensorflow.keras.models import load_model # type: ignore
except ImportError:
    load_model = None

class ModelTester:
    def __init__(self, model_path, scaler_path, relevant_features, model_type='ann', delimiter=','):
        """
        Initializes the tester with paths for the trained model and scaler.
        
        Parameters:
            model_path (str): Path to the saved model. For ANN, a Keras model file; for RF/IF, a joblib file.
            scaler_path (str): Path to the saved scaler (joblib file).
            relevant_features (list): List of feature names used during training.
            model_type (str): 'ann' for Keras ANN, 'rf' for RandomForestClassifier, or 'if' for IsolationForest.
            delimiter (str): CSV delimiter.
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.relevant_features = relevant_features
        self.delimiter = delimiter
        self.model_type = model_type.lower()
        
        # Load model based on type
        if self.model_type == 'ann':
            if load_model is None:
                raise ImportError("TensorFlow/Keras is not installed.")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            self.model = load_model(self.model_path)
        elif self.model_type in ['rf', 'if']:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            self.model = joblib.load(self.model_path)
        else:
            raise ValueError("model_type must be 'ann', 'rf', or 'if'.")
        
        # Load scaler
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
        self.scaler = joblib.load(self.scaler_path)
    
    def load_and_clean_data(data,delimiter=','):
        """
        Loads data from a CSV file or accepts an existing DataFrame,
        and removes columns with names like 'Unnamed: ...'.

        Parameters:
            data (str or pd.DataFrame): Path to the CSV file or a DataFrame.

        Returns:
            pd.DataFrame: DataFrame with the loaded and cleaned data.
        """
        try:
            if isinstance(data, str):
                # Load data from CSV file
                df = pd.read_csv(data,delimiter=delimiter)
            elif isinstance(data, pd.DataFrame):
                # Use the provided DataFrame
                df = data
            else:
                raise ValueError("Input must be a file path (str) or a pandas DataFrame.")

            # Remove columns with names like 'Unnamed: ...'
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def preprocess_data(self, df):
        """
        Selects the relevant features from the new data and scales them.
        
        Parameters:
            df (DataFrame): Input DataFrame.
        
        Returns:
            Scaled feature array.
        """
        try:
            X_new = df[self.relevant_features]
        except KeyError as e:
            print(f"Missing expected feature columns: {e}")
            raise
        try:
            X_new_scaled = self.scaler.transform(X_new)
        except Exception as e:
            print(f"Error during scaling: {e}")
            raise
        return X_new_scaled

    def predict(self, X_new_scaled, threshold=0.5):
        """
        Uses the loaded model to predict anomalies on the new scaled data.
        
        For:
         - ANN: predictions are probabilities; threshold is applied.
         - RF: predictions are binary labels and probabilities via predict_proba.
         - IF: predictions are +1 (normal) and -1 (anomaly); convert -1 to 1 (anomaly) and +1 to 0.
        
        Parameters:
            X_new_scaled (array): Scaled feature array.
            threshold (float): Classification threshold for ANN (default 0.5).
        
        Returns:
            tuple: (binary_predictions, prediction_scores)
        """
        try:
            if self.model_type == 'ann':
                pred_probs = self.model.predict(X_new_scaled).ravel()
                predictions = (pred_probs > threshold).astype("int32")
                return predictions, pred_probs
            elif self.model_type == 'rf':
                predictions = self.model.predict(X_new_scaled)
                try:
                    pred_probs = self.model.predict_proba(X_new_scaled)[:, 1]
                except Exception:
                    pred_probs = None
                return predictions, pred_probs
            elif self.model_type == 'if':
                pred = self.model.predict(X_new_scaled)
                # IsolationForest: -1 indicates anomaly, +1 indicates normal
                predictions = np.where(pred == -1, 1, 0)
                try:
                    # decision_function returns anomaly scores; more negative means more anomalous.
                    pred_scores = self.model.decision_function(X_new_scaled)
                except Exception:
                    pred_scores = None
                return predictions, pred_scores
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise

    def save_predictions(self, df, predictions, output_file):
        """
        Appends the predictions to the DataFrame and saves to a CSV file.
        
        Parameters:
            df (DataFrame): Original DataFrame.
            predictions (array): Binary predictions.
            output_file (str): Output CSV file path.
        """
        try:
            df['Anomaly'] = predictions
            df.to_csv(output_file, index=False)
            print(f"Predictions saved to {output_file}")
        except Exception as e:
            print(f"Error saving predictions: {e}")
            raise
