import pandas as pd
import numpy as np
import ipaddress

class DataPreprocessor:
    def __init__(self, file_path, delimiter=','):
        """
        Initializes the DataPreprocessor with the file path and delimiter.
        """
        self.file_path = file_path
        self.delimiter = delimiter

    def load_data(self):
        """
        Loads the CSV data into a DataFrame.
        """
        df = pd.read_csv(self.file_path, delimiter=self.delimiter)
        return df

    def drop_duplicates(self, df):
        """
        Drops duplicate rows from the DataFrame.
        """
        return df.drop_duplicates().copy()

    def create_missing_flags(self, df):
        """
        Creates boolean flag columns for missing source and destination IPs.
        """
        df.loc[:, 'missing_source_ip'] = df['_source_source_ip'].isnull() | (df['_source_source_ip'].str.strip() == "")
        df.loc[:, 'missing_destination_ip'] = df['_source_destination_ip'].isnull() | (df['_source_destination_ip'].str.strip() == "")
        return df

    def convert_timestamp(self, df):
        """
        Converts the _source_@timestamp column to datetime and drops invalid entries.
        """
        df['_source_@timestamp'] = pd.to_datetime(df['_source_@timestamp'], errors='coerce')
        return df[df['_source_@timestamp'].notnull()]

    def set_timestamp_index(self, df):
        """
        Sets the _source_@timestamp column as the DataFrame index.
        """
        return df.set_index('_source_@timestamp')

    def aggregate_sessions(self, df):
        """
        Aggregates flows into 1-minute windows and computes session-level summaries.
        Also aggregates by _source_flow_id for session total bytes and duration.
        Returns both the merged DataFrame and the 1-minute aggregated DataFrame.
        """
        # Group by 1-minute intervals for overall stats
        aggregated = df.groupby(pd.Grouper(freq='1Min')).agg({
            '_source_network_bytes': 'sum',      # Total bytes transferred
            '_source_event_duration': 'mean',      # Average event duration
            '_source_source_ip': pd.Series.nunique, # Unique source IPs count
            '_source_destination_ip': pd.Series.nunique  # Unique destination IPs count
        }).rename(columns={
            '_source_network_bytes': 'total_bytes',
            '_source_event_duration': 'avg_duration',
            '_source_source_ip': 'unique_source_ips',
            '_source_destination_ip': 'unique_destination_ips'
        }).reset_index()

        # Reset index for merging based on _source_flow_id
        df_reset = df.reset_index()

        # Aggregate per session based on _source_flow_id
        session_agg = (df_reset
            .groupby('_source_flow_id')
            .agg({
                '_source_network_bytes': 'sum',
                '_source_event_duration': 'sum',
                '_source_@timestamp': 'first'
            })
            .rename(columns={
                '_source_network_bytes': 'session_bytes_total',
                '_source_event_duration': 'session_duration_total',
                '_source_@timestamp': '_source_@timestamp_session'
            })
            .reset_index()
            .set_index('_source_flow_id')
        )

        # Merge the session-level summaries back to the original data
        df_merged = pd.merge(df_reset, session_agg, on='_source_flow_id', how='left', suffixes=('', '_session'))
        # Combine timestamps to retain the original _source_@timestamp
        df_merged['_source_@timestamp'] = df_merged['_source_@timestamp'].combine_first(df_merged['_source_@timestamp_session'])
        df_merged.drop(columns=['_source_@timestamp_session'], inplace=True)

        return df_merged, aggregated

    def ip_to_numeric(self, ip_str):
        """
        Converts an IP address string to an integer.
        Works for both IPv4 and IPv6 addresses.
        If conversion fails, returns NaN.
        """
        try:
            return int(ipaddress.ip_address(ip_str))
        except ValueError:
            return np.nan

    def convert_ips(self, df):
        """
        Converts both source and destination IP addresses to numeric format.
        """
        df['source_ip_numeric'] = df['_source_source_ip'].apply(self.ip_to_numeric)
        df['destination_ip_numeric'] = df['_source_destination_ip'].apply(self.ip_to_numeric)
        return df

    def fill_missing(self, df, fill_value=-2):
        """
        Fills missing values in the DataFrame.
        """
        df.fillna(fill_value, inplace=True)
        return df

    def preprocess(self):
        """
        Runs the complete preprocessing pipeline.
        Returns the cleaned DataFrame and aggregated session data.
        """
        df = self.load_data()
        df = self.drop_duplicates(df)
        df = self.create_missing_flags(df)
        df = self.convert_timestamp(df)
        df = self.set_timestamp_index(df)
        df, aggregated = self.aggregate_sessions(df)
        df = self.convert_ips(df)
        df = self.fill_missing(df)
        return df, aggregated
