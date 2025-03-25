import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.optimizers import Nadam # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix


# ----------------------------
# ANN Trainer Class
# ----------------------------
class ANNTrainer:
    def __init__(self, df, relevant_features, label_column='label', model_path='C:\\Users\\ASUS\\Guidewire_Hackathon\\src\\models\\trained_ann_model.keras',scalar_path="C:\\Users\\ASUS\\Guidewire_Hackathon\\src\\scalars\\ann_model"):
        """
        Initializes the ANNTrainer.
        :param df: Input DataFrame containing features and label.
        :param relevant_features: List of feature names to use for training.
        :param label_column: Column name of the label (binary: benign/malicious).
        :param model_path: Path to save the trained ANN model.
        :param scaler
        """
        self.df = df
        self.relevant_features = relevant_features
        self.label_column = label_column
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.scaler_path = scalar_path
        self.model = None

    def prepare_data(self):
        """
        Selects relevant features and converts the label to binary (0: benign, 1: malicious).
        """
        df_model = self.df[self.relevant_features].copy()
        # Convert label to binary: 0 for benign, 1 for malicious.
        df_model[self.label_column] = self.df[self.label_column].apply(lambda x: 0 if x.lower() == 'benign' else 1)
        X = df_model.drop(self.label_column, axis=1)
        y = df_model[self.label_column]
        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Splits the data into training and testing sets.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test

    def scale_data(self, X_train, X_test):
        """
        Scales the training and testing data using StandardScaler.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def build_model(self, input_dim):
        """
        Builds the ANN model architecture.
        """
        model = Sequential()
        # Input layer + first hidden layer with L2 regularization and dropout
        model.add(Dense(32, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.3))
        # Second hidden layer
        model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.3))
        # Third hidden layer
        model.add(Dense(8, activation='relu', kernel_regularizer=l2(0.1)))
        model.add(Dropout(0.3))
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile the model using Nadam optimizer
        model.compile(loss='binary_crossentropy', optimizer=Nadam(learning_rate=0.001), metrics=['accuracy'])
        self.model = model
        return model

    def train(self, X_train_scaled, y_train, epochs=15, batch_size=32, validation_split=0.2):
        """
        Trains the ANN model using early stopping.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = self.model.fit(
            X_train_scaled, y_train.to_numpy(), 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split, 
            callbacks=[early_stopping]
        )
        return history

    def evaluate(self, X_test_scaled):
        """
        Predicts on the test set and returns binary predictions.
        """
        y_pred = (self.model.predict(X_test_scaled) > 0.5).astype("int32")
        return y_pred

    def save(self):
        """
        Saves the trained model and scaler.
        """
        if self.model is None:
            raise ValueError("No model has been trained yet.")
        self.model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path + '_scaler.pkl')
        print("Model and scaler saved to disk.")

    def run_pipeline(self):
        """
        Executes the complete training pipeline for the ANN.
        Returns training history, predictions, and test labels.
        """
        # Prepare data
        X, y = self.prepare_data()
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        X_train_scaled, X_test_scaled = self.scale_data(X_train, X_test)
        
        # --- SMOTE Oversampling ---
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

        # Build and train model
        self.build_model(input_dim=X_train_resampled.shape[1])
        history = self.train(X_train_resampled, y_train_resampled)
        
        # Evaluate model on test data
        y_pred = self.evaluate(X_test_scaled)
        
        # Save the model
        self.save()
        
        return history, y_pred, y_test



class RandomForestTrainer:
    def __init__(self, df, relevant_features, label_column='label', random_state=42, model_path='C:\\Users\\ASUS\\Guidewire_Hackathon\\src\\models\\trained_rf_model.pkl',scalar_path="C:\\Users\\ASUS\\Guidewire_Hackathon\\src\\scalers\\rf_model"):
        """
        Initializes the RandomForestTrainer.
        
        :param df: Input DataFrame containing features and label.
        :param relevant_features: List of feature names to use for training.
        :param label_column: Name of the label column.
        :param random_state: Random seed for reproducibility.
        :param model_path: Path to save the trained model.
        :param scaler_path: Path to save the scaler.
        """
        self.df = df
        self.relevant_features = relevant_features
        self.label_column = label_column
        self.random_state = random_state
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.model = None
        self.scaler_path = scalar_path

    def prepare_data(self):
        """
        Selects relevant features and converts the label to binary (0 for benign, 1 for malicious).
        Returns: X (features DataFrame) and y (binary labels Series).
        """
        df_model = self.df[self.relevant_features].copy()
        # Convert label: assume 'benign' (case-insensitive) is 0 and others (e.g., 'malicious') are 1.
        df_model[self.label_column] = self.df[self.label_column].apply(lambda x: 0 if x.lower() == 'benign' else 1)
        X = df_model.drop(self.label_column, axis=1)
        y = df_model[self.label_column]
        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=None, val_split=0.125):
        """
        Splits data into train, validation, and test sets.
        Test set = test_size of entire data.
        Validation set = val_split of the train+val split.
        Returns: X_train, X_val, X_test, y_train, y_val, y_test.
        """
        if random_state is None:
            random_state = self.random_state
        # First split: (train+val) and test
        X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        # Second split: from train+val, split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=val_split, random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale_data(self, X_train, X_val, X_test):
        """
        Scales the features using StandardScaler.
        Returns: Scaled versions of X_train, X_val, and X_test.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled

    def train(self, X_train_scaled, y_train, use_grid_search=False, param_grid=None, n_iter=10, cv=5):
        """
        Trains a RandomForestClassifier.
        
        :param X_train_scaled: Scaled training features.
        :param y_train: Training labels.
        :param use_grid_search: If True, use RandomizedSearchCV for hyperparameter tuning.
        :param param_grid: Dictionary of hyperparameter values.
        :param n_iter: Number of parameter settings sampled if using grid search.
        :param cv: Number of cross-validation folds.
        Returns: The trained RandomForestClassifier model.
        """
        if use_grid_search:
            # Default parameter grid if none provided
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [10, 12, 15],
                    'min_samples_leaf': [11, 12, 14],
                    'max_features': ['sqrt', 'log2', None],
                    'bootstrap': [True, False]
                }
            rf = RandomForestClassifier(random_state=self.random_state)
            random_search = RandomizedSearchCV(
                estimator=rf,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring='f1',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )
            random_search.fit(X_train_scaled, y_train)
            print("Best parameters (Randomized Search):", random_search.best_params_)
            print("Best F1 score:", random_search.best_score_)
            self.model = random_search.best_estimator_
        else:
            self.model = RandomForestClassifier(random_state=self.random_state)
            self.model.fit(X_train_scaled, y_train)
        return self.model

    def evaluate(self, X_test_scaled, y_test):
        """
        Evaluates the trained model on the test set.
        Prints the confusion matrix and classification report.
        Returns the predictions.
        """
        y_pred = self.model.predict(X_test_scaled)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        return y_pred

    def save_model(self):
        """
        Saves the trained model and the scaler.
        """
        if self.model is None:
            raise ValueError("No model trained yet.")
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path + '_scaler.pkl')
        print("Model and scaler saved to disk.")

    def run_pipeline(self, test_size=0.2, val_split=0.125, use_grid_search=False, n_iter=10, cv=5):
        """
        Executes the complete RandomForest training pipeline.
        Returns: X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, trained model.
        """
        # Data preparation
        X, y = self.prepare_data()
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y, test_size=test_size, val_split=val_split)
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_data(X_train, X_val, X_test)
        model = self.train(X_train_scaled, y_train, use_grid_search=use_grid_search, n_iter=n_iter, cv=cv)
        self.save_model()
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, model



class IsolationForestTrainer:
    def __init__(self, df, relevant_features, label_column='label', random_state=42, model_path='C:\\Users\\ASUS\\Guidewire_Hackathon\\src\\models\\trained_if_model.pkl',
    scaler_path='C:\\Users\\ASUS\\Guidewire_Hackathon\\src\\scalers\\if_model'):
        """
        Initializes the IsolationForestTrainer.

        Parameters:
            df (DataFrame): Input DataFrame containing features and (optionally) labels.
            relevant_features (list): List of feature names to use.
            label_column (str): Name of the label column (if available).
            random_state (int): Random seed for reproducibility.
            model_path (str): Path to save the trained model.
        """
        self.df = df
        self.relevant_features = relevant_features
        self.label_column = label_column
        self.random_state = random_state
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.model = None

    def prepare_data(self):
        """
        Selects the relevant features and (if label exists) converts it to binary (0: benign, 1: malicious).
        Returns:
            X (DataFrame): Feature set.
            y (Series): Labels, or None if not available.
        """
        df_model = self.df[self.relevant_features].copy()
        if self.label_column in self.df.columns:
            df_model[self.label_column] = self.df[self.label_column].apply(lambda x: 0 if x.lower() == 'benign' else 1)
            y = df_model[self.label_column]
            X = df_model.drop(self.label_column, axis=1)
        else:
            X = df_model
            y = None
        return X, y

    def split_data(self, X, y, test_size=0.2, val_split=0.125):
        """
        Splits the data into train, validation, and test sets.

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First split: train+val and test
        X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)
        # Second split: from train+val, extract validation (val_split of the full dataset)
        X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=val_split, random_state=self.random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale_data(self, X_train, X_val, X_test):
        """
        Scales the data using StandardScaler.
        
        Returns:
            X_train_scaled, X_val_scaled, X_test_scaled
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled

    def train(self, X_train_scaled, y_train, tuning_method=None, param_grid=None, n_iter=10, cv=5):
        """
        Trains an IsolationForest model with an option for hyperparameter tuning.
        
        Parameters:
            X_train_scaled (array): Scaled training features.
            y_train (Series): Training labels (used only for scoring during tuning).
            tuning_method (str): 'grid' for GridSearchCV, 'random' for RandomizedSearchCV, or None.
            param_grid (dict): Dictionary of hyperparameters to tune.
            n_iter (int): Number of iterations for randomized search.
            cv (int): Number of cross-validation folds.
            
        Returns:
            Trained IsolationForest model.
        """
        # Default hyperparameter grid if none is provided
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_samples': [0.5, 0.75, 1.0],
                'contamination': [0.1, 0.15, 0.2, 0.25],
                'max_features': [0.5, 0.75, 1.0],
                'bootstrap': [True, False]
            }
        # Initialize the IsolationForest estimator
        estimator = IsolationForest(random_state=self.random_state, n_jobs=-1)
        
        if tuning_method is not None:
            if tuning_method == 'random':
                tuner = RandomizedSearchCV(
                    estimator=estimator,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=cv,
                    scoring='f1',  # Assumes y_train is available for scoring
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=1
                )
            elif tuning_method == 'grid':
                tuner = GridSearchCV(
                    estimator=estimator,
                    param_grid=param_grid,
                    cv=cv,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=1
                )
            else:
                raise ValueError("tuning_method must be 'grid', 'random', or None.")
            
            # Fit the tuner using the training data
            tuner.fit(X_train_scaled, y_train)
            print("Best parameters:", tuner.best_params_)
            print("Best F1 score:", tuner.best_score_)
            self.model = tuner.best_estimator_
        else:
            # Train with default parameters (or manually set ones)
            self.model = IsolationForest(
                contamination=0.2,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X_train_scaled)
        return self.model

    def save_model(self):
        """
        Saves the trained model and scaler to disk.
        """
        if self.model is None:
            raise ValueError("No model has been trained yet.")
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.model_path + '_scaler.pkl')
        print("Model and scaler saved to disk.")

    def run_pipeline(self, test_size=0.2, val_split=0.125, tuning_method=None, n_iter=10, cv=5):
        """
        Runs the complete training pipeline for IsolationForest.
        
        Returns:
            X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, trained model.
        """
        X, y = self.prepare_data()
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y, test_size=test_size, val_split=val_split)
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_data(X_train, X_val, X_test)
        self.train(X_train_scaled, y_train, tuning_method=tuning_method, n_iter=n_iter, cv=cv)
        self.save_model()
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, self.model

