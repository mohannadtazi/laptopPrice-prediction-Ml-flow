# MLflow Model for Pulsar Star Classification

## Overview

This project demonstrates the use of **MLflow** for tracking machine learning models, packaging code, and deploying it in a reproducible manner. The model was trained to classify pulsar stars based on various astrophysical features. It leverages **scikit-learn** and **MLflow** to track the training process, log metrics, and store the final model.

## Why I Built It

This project was designed to:
- Showcase the integration of MLflow for model tracking and management.
- Provide a reproducible pipeline that can be easily shared and executed by others.
- Demonstrate the process of packaging machine learning code as an MLflow project, making it portable and easy to run with different datasets.

## How to Use It

### 1. Setup

Make sure you have the following dependencies installed:
- Python 3.7+
- Conda (Miniconda or Anaconda)
- MLflow
- scikit-learn
- pandas

Running the Training Script

Clone the repository and navigate to the project directory:

```bash
git clone <repo-url>
cd <project-directory>
```

Create a virtual environment and install the dependencies:

```bash
conda env create -f conda.yaml
conda activate <env-name>
```

Run the training script:

```bash
python psp/training.py --data-path ./data/pulsar_stars.csv --test-size 0.2
```
This will train the model and log metrics to MLflow.

### 3. Running as an MLflow Project
To run the project as an MLflow project, ensure you have the environment variables set for MLflow:

```bash
export MLFLOW_TRACKING_URI=<remote-mlflow-uri>
export MLFLOW_TRACKING_USERNAME="<username>"
export MLFLOW_TRACKING_PASSWORD="<password>"
```
Then, run the following command:

```bash
mlflow run . -P data_path=./data/pulsar_stars.csv
```

### 4. Exposing the Model for Inference
Once the model is trained, you can serve it using MLflow's model serving capability:

```bash
mlflow models serve -m <model-path> -p 4321
```

You can then make predictions by sending a POST request with input data to http://127.0.0.1:4321/invocations.

### Conclusion
This project provides a streamlined workflow to track, deploy, and reproduce machine learning models using MLflow. With the integration of remote MLflow tracking, model exposure, and easy-to-follow steps, the project is ready to scale for production-level MLOps tasks.