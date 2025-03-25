import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

class FeatureExtractor:
    def __init__(self):
        pass

    def classify_port_range(self, port):
        """
        Classifies the port number into one of three categories:
          - 'Well-Known' for ports 0 to 1023,
          - 'Registered' for ports 1024 to 49151,
          - 'Ephemeral' for ports 49152 to 65535.
        If the port is missing or invalid, returns 'Unknown'.
        """
        try:
            p = int(port)
        except (ValueError, TypeError):
            return 'Unknown'
        
        if 0 <= p <= 1023:
            return 'Well-Known'
        elif 1024 <= p <= 49151:
            return 'Registered'
        elif 49152 <= p <= 65535:
            return 'Ephemeral'
        else:
            return 'Out of Range'

    def get_unique_ports(self, df, source_col, destination_col):
        """
        Extracts unique port numbers from the source and destination columns.
        """
        unique_source_ports = df[source_col].unique()
        unique_destination_ports = df[destination_col].unique()
        all_ports = np.union1d(unique_source_ports, unique_destination_ports)
        return all_ports

    def classify_all_ports(self, df, source_col, destination_col):
        """
        Classifies all unique port numbers in the DataFrame.
        Returns a dictionary mapping each port to its classification.
        """
        unique_ports = self.get_unique_ports(df, source_col, destination_col)
        port_classification = {port: self.classify_port_range(port) for port in unique_ports}
        return port_classification

    def impute_and_scale_features(self, df):
        """
        Handles missing values by imputing them with the mean and scales the features.
        Returns the scaled feature array and the fitted scaler.
        """
        imputer = SimpleImputer(strategy='mean')
        df_imputed = imputer.fit_transform(df)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df_imputed)
        return scaled_features, scaler

    def perform_pca(self, scaled_features, variance_threshold=0.95):
        """
        Applies PCA to the scaled features to retain the given variance.
        Returns the principal components and the PCA object.
        """
        pca = PCA(n_components=variance_threshold)
        principal_components = pca.fit_transform(scaled_features)
        return principal_components, pca

    def plot_pca_scatter(self, principal_components, labels):
        """
        Plots a 2D scatter plot of the first two principal components.
        
        Parameters:
            principal_components (array-like): Array of principal components.
            labels (Series or array-like): Label values. If non-numeric, they will be converted:
                'benign' -> 0 and all others -> 1.
        """
        pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])
        # Convert labels to numeric if they are not already numeric
        if not np.issubdtype(np.array(labels).dtype, np.number):
            # Assuming labels are strings; convert 'benign' to 0, everything else to 1.
            label_numeric = pd.Series(labels).apply(lambda x: 0 if str(x).lower() == 'benign' else 1)
        else:
            label_numeric = labels
        pca_df['label_numeric'] = label_numeric
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['label_numeric'], cmap='viridis', alpha=0.7)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA - Scatter Plot of Principal Components')
        plt.colorbar(scatter, label='Label')
        plt.show()

    def plot_scree(self, explained_variance_ratio):
        """
        Plots a scree plot of the explained variance ratio for each principal component.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Scree Plot')
        plt.show()

    def plot_3d_pca(self, principal_components, labels):
        """
        Plots a 3D scatter plot for the first three principal components.
        """
        pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])
        if not np.issubdtype(np.array(labels).dtype, np.number):
            label_numeric = pd.Series(labels).apply(lambda x: 0 if str(x).lower()=='benign' else 1)
        else:
            label_numeric = labels
        pca_df['label_numeric'] = label_numeric
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=pca_df['label_numeric'], cmap='viridis', alpha=0.7)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('3D Scatter Plot of PCA Components')
        fig.colorbar(scatter, label='Label')
        plt.show()

    def add_protocol_numeric(self, df):
        """
        Maps protocol strings in '_source_network_transport' to numeric values and creates a new column 'protocol_numeric'.
        
        Returns the updated DataFrame.
        """
        protocol_mapping = {'tcp': 0, 'udp': 1}
        try:
            df['protocol_numeric'] = df['_source_network_transport'].map(protocol_mapping).fillna(-1).astype(int)
        except KeyError as e:
            raise KeyError(f"Expected column '_source_network_transport' not found: {e}")
        return df

    def get_protocol_unique(self, df):
        """
        Returns the unique values from the '_source_network_transport' column.
        """
        try:
            return df['_source_network_transport'].unique()
        except KeyError as e:
            raise KeyError(f"Expected column '_source_network_transport' not found: {e}")

    def plot_total_bytes_over_time(self, aggregated_df):
        """
        Plots total bytes transferred over time using a 1-minute aggregated DataFrame.
        
        Parameters:
            aggregated_df (DataFrame): DataFrame with '_source_@timestamp' and 'total_bytes' columns.
        """
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        plt.plot(aggregated_df['_source_@timestamp'], aggregated_df['total_bytes'], marker='o', linestyle='-')
        plt.title('Total Bytes Transferred Over Time (1-Minute Intervals)')
        plt.xlabel('Timestamp')
        plt.ylabel('Total Bytes')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_unique_ips_over_time(self, aggregated_df):
        """
        Plots the count of unique source and destination IPs over time using a 1-minute aggregated DataFrame.
        
        Parameters:
            aggregated_df (DataFrame): DataFrame with '_source_@timestamp', 'unique_source_ips', and 'unique_destination_ips' columns.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(aggregated_df['_source_@timestamp'], aggregated_df['unique_source_ips'], label='Unique Source IPs', marker='o')
        plt.plot(aggregated_df['_source_@timestamp'], aggregated_df['unique_destination_ips'], label='Unique Destination IPs', marker='s')
        plt.title('Unique IP Counts Over Time (1-Minute Intervals)')
        plt.xlabel('Timestamp')
        plt.ylabel('Count of Unique IPs')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_flow_duration_distribution(self, df):
        """
        Plots the distribution of flow durations (in seconds) using the '_source_event_duration' column.
        Assumes '_source_event_duration' is in nanoseconds.
        
        Parameters:
            df (DataFrame): DataFrame containing the '_source_event_duration' column.
        """
        plt.figure(figsize=(8, 5))
        sns.histplot(df['_source_event_duration'] / 1e9, bins=50, kde=True)
        plt.title('Distribution of Flow Durations (seconds)')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self, df, drop_cols=None):
        """
        Drops non-numeric columns (or those specified) from the DataFrame, computes the correlation matrix for the numeric features,
        and plots a heatmap.
        
        Parameters:
            df (DataFrame): Input DataFrame.
            drop_cols (list): List of columns to drop before computing correlation. 
                              If None, all non-numeric columns are dropped.
        """
        if drop_cols is None:
            drop_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        df_numeric = df.drop(columns=drop_cols)
        correlation_matrix = df_numeric.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
