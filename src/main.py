from data_preprocessing import DataPreprocessor
from feature_extraction_collection import FeatureExtractor
from training import ANNTrainer
from testing import ModelTester
from evaluation import ModelEvaluator
import pandas as pd
import numpy as np


# Define the path to your training dataset file
training_file_path = r'C:\Users\ASUS\Guidewire_Hackathon\datasets\elastic_may2021_data.csv'

# Create an instance of DataPreprocessor (use appropriate delimiter)
train_preprocessor = DataPreprocessor(training_file_path, delimiter=',')

# Load and preprocess the training data
df_train = train_preprocessor.load_data()
df_train = train_preprocessor.drop_duplicates(df_train)
df_train = train_preprocessor.create_missing_flags(df_train)
df_train = train_preprocessor.convert_timestamp(df_train)
df_train = train_preprocessor.set_timestamp_index(df_train)
df_train, aggregated_train = train_preprocessor.aggregate_sessions(df_train)
# Convert IP addresses to numeric after aggregation (if needed)
df_train = train_preprocessor.convert_ips(df_train)
df_train = train_preprocessor.fill_missing(df_train)

# Feature extraction
fe = FeatureExtractor()

# Add protocol numeric mapping
df_train = fe.add_protocol_numeric(df_train)

# Plot various visualizations on the aggregated training data
fe.plot_total_bytes_over_time(aggregated_train)
fe.plot_unique_ips_over_time(aggregated_train)
fe.plot_flow_duration_distribution(df_train)
unique_ports = fe.classify_all_ports(df_train, '_source_source_port', '_source_destination_port')
print("Port classifications:", unique_ports)
fe.plot_correlation_matrix(df_train, drop_cols=['_source_flow_id', 'label', '_source_network_transport', '_source_source_ip', '_source_destination_ip', '_source_@timestamp'])

# For PCA and visualization, drop non-numeric columns
df_for_pca = df_train.drop(columns=['_source_flow_id', 'label', '_source_network_transport', '_source_source_ip', '_source_destination_ip', '_source_@timestamp'])
scaled_features_train, scaler_obj = fe.impute_and_scale_features(df_for_pca)
principal_components_train, pca_obj = fe.perform_pca(scaled_features_train)
fe.plot_pca_scatter(principal_components_train, pd.Series(df_train['label']))
explained_variance = np.var(principal_components_train, axis=0)
explained_variance_ratio = explained_variance / np.sum(explained_variance)
fe.plot_scree(explained_variance_ratio)
if principal_components_train.shape[1] >= 3:
    fe.plot_3d_pca(principal_components_train, pd.Series(df_train['label']))


# Define the list of relevant features used for training the model
relevant_features = [
    'source_ip_numeric',
    '_source_source_port',
    'destination_ip_numeric',
    '_source_destination_port',
    '_source_network_bytes',
    '_source_event_duration',
    'protocol_numeric',
    'session_bytes_total',
    'session_duration_total'
]

# Initialize the ANNTrainer with the preprocessed training DataFrame
ann_trainer = ANNTrainer(df_train, relevant_features, label_column='label', model_path='C:\\Users\\ASUS\\Guidewire_Hackathon\\src\\models\\trained_ann_model.keras')

# Run the training pipeline; this splits, scales, trains, and saves the ANN model
history, y_pred_train, y_train_actual = ann_trainer.run_pipeline()


# Define the path to your testing dataset file
testing_file_path = r'C:\Users\ASUS\Guidewire_Hackathon\datasets\elastic_may2022_data.csv'

# Create a new DataPreprocessor instance for testing (adjust delimiter as needed)
test_preprocessor = DataPreprocessor(testing_file_path, delimiter=';')

# Load and preprocess the testing data
df_test = test_preprocessor.load_data()
df_test = test_preprocessor.drop_duplicates(df_test)
df_test = test_preprocessor.create_missing_flags(df_test)
df_test = test_preprocessor.convert_timestamp(df_test)
df_test = test_preprocessor.set_timestamp_index(df_test)
df_test, aggregated_test = test_preprocessor.aggregate_sessions(df_test)
df_test = test_preprocessor.convert_ips(df_test)
df_test = test_preprocessor.fill_missing(df_test)

# Apply protocol mapping using the same feature extractor instance
df_test = fe.add_protocol_numeric(df_test)

# (Optional) You may also want to check or visualize features on test data
# e.g., fe.plot_total_bytes_over_time(aggregated_test)


# Create an instance of ModelTester using the trained ANN model and corresponding scaler
tester = ModelTester(
    model_path=ann_trainer.model_path,
    scaler_path=ann_trainer.scaler_path + '_scaler.pkl',
    relevant_features=relevant_features,
    model_type='ann',  # Change to 'rf' or 'if' if testing a different model type
    delimiter=';'      # Use the appropriate delimiter for the test CSV
)

# Here, since we have already loaded and preprocessed df_test, we can pass it directly to preprocess_data.
X_test_scaled = tester.preprocess_data(df_test)

# Predict anomalies using the ANN model
predictions, pred_probs = tester.predict(X_test_scaled, threshold=0.5)

# Save predictions to a CSV file
output_file = r'C:\Users\ASUS\Guidewire_Hackathon\datasets\outputs\anomaly_predictions(main.py).csv'
tester.save_predictions(df_test, predictions, output_file)

# If the test data contains labels, they can be extracted and used in evaluation.
if 'label' in df_test.columns:
    y_test_actual = df_test['label'].apply(lambda x: 0 if str(x).lower() == 'benign' else 1).values
else:
    y_test_actual = None


# Evaluate the model using the predictions and actual labels
if y_test_actual is not None:
    evaluator = ModelEvaluator(y_true=y_test_actual, y_pred=predictions, y_pred_proba=pred_probs)
    evaluator.compute_classification_metrics()
    evaluator.plot_roc_curve()
    evaluator.plot_precision_recall_curve()
else:
    print("Test data does not contain labels; evaluation metrics cannot be computed.")

print("Testing and evaluation pipeline completed.")
