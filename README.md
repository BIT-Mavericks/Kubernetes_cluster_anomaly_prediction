# Phase 1: AI/ML Model for Predicting Kubernetes Issues


## Overview
This project presents a novel meta-learning ensemble approach for predictive autoscaling in Kubernetes clusters. The framework integrates multiple machine learning models—namely a Graph Neural Network (GNN), a Long Short-Term Memory (LSTM) network, and a simple Artificial Neural Network (ANN) branch—combined via a meta-learner to predict and remediate anomalies such as pod failures, resource exhaustion, and network issues. By leveraging techniques like SMOTE for class imbalance, regularization (dropout, L2 regularization), and meta-learning for dynamic ensemble weighting, the system aims to overcome overfitting while maintaining high accuracy in real-time environments.
For the prediction of resource(CPU, memory) exhaustion in kubernetes cluster we have used XGBoost and LGBMClassifier. For this issue we have provided a unique solution of using Convolutional Neural Network(CNN), Logistic Regression and we have done analysis and experimentation with many other models.

## Features
- **Hierarchical Hybrid Model:** Simultaneously combines the models for each type of anomaly based on the interface.
- **Multi-Model Ensemble:** Incorporates GNN for relational data, LSTM for temporal forecasting, and ANN for additional predictive power.
- **Meta-Learner:** Dynamically weighs individual model outputs to generate a final prediction.
- **SMOTE-Based Oversampling:** Mitigates class imbalance by generating synthetic minority samples.
- **Real-Time Monitoring Integration:** Designed to eventually work with real-time metrics (e.g., from Prometheus/Kafka).

## Requirements
- Python 3.7+
- TensorFlow 2.x
- PyTorch and torch_geometric
- scikit-learn
- imbalanced-learn (imblearn)
- pandas, numpy
- Additional dependencies as listed in `requirements.txt`

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```
2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate` or use conda to create virtual environment.
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## File Structure
- **data_preprocessing.py:** Contains the code to load and preprocess the raw dataset.
- **resource.ipynb:** Contains the implementation of XGBoost and RandomForest Classifier implementation for the detection of resource(CPU, memory) exhaustion.
- **logistic.ipynb:** Contains the implementation of logistic regression for resource exhaustion(CPU, memory). 
- **feature_extraction.py:** Provides functions for feature imputation, scaling, and PCA.
- **main.py:** Implements Artifical Neural Network(ANN) for detecting the network issues in kubernetes cluster.
- **training.py:** Contains the training loop for the ensemble model, including techniques to mitigate overfitting and class imbalance.
- **evaluation.py:** Provides functions to evaluate the model’s performance on validation and test datasets.
- **testing.py:** The main execution script that ties together data preprocessing, model training, and evaluation of the ANN approach of detecting network issues.
- **README.md:** This file, providing an overview and instructions for the project.
- **retrieval.py:** Contains the code for retrieval of logs data for our project using prometheus and kubernetes API.

## Usage
1. **Preprocessing & Training:**
   - Update the dataset paths in `main.py`.
   - Run the main script:
     ```bash
     python main.py
     ```
   - The script will load the dataset, preprocess the data, train the ensemble model, and evaluate its performance on both validation and test sets.

2. **Evaluation:**
   - Model performance metrics and visualizations (such as loss and accuracy) will be printed to the console.
   - For additional testing, you can modify the evaluation module to work with your specific test data.

## Results and Future Work
The project demonstrates improved predictive accuracy and robustness through its hierarchical hybrid ensemble approach. Future work could involve:
- Integrating real-time data ingestion for live scaling decisions.
- Enhancing the meta-learner with reinforcement learning for adaptive hyperparameter tuning.
- Extending the framework to support multi-cloud and edge computing environments.

## Acknowledgements
- Datasets used in this project:
  - [Mendeley Dataset](https://data.mendeley.com/datasets/ks9vbv5pb2/1)
  - [Kaggle Dataset](https://www.kaggle.com/datasets/nickkinyae/kubernetes-resource-and-performancemetricsallocation?resource=download)
  - [4TU Dataset](https://data.4tu.nl/articles/dataset/AssureMOSS_Kubernetes_Run-time_Monitoring_Dataset/20463687)
- Additional references from research articles and the k8sgpt website.




