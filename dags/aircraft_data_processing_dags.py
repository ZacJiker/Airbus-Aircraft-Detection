from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Define default arguments for the DAG
default_args = {
    "owner": "airflow", 
    "depends_on_past": False,
    "start_date": datetime(2023, 8, 7),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Create a DAG instance with the given parameters
with DAG("download_and_preprocess_dag", default_args=default_args, schedule_interval="@daily", catchup=False) as dag:
    
    # Define a function to download the dataset using Kaggle API
    def download_dataset(dataset: str, input_dir: str, unzip: bool, force: bool):
        import os
        import kaggle
        kaggle.api.authenticate()
        # Create input directory if it does not exist
        os.makedirs(input_dir, exist_ok=True)
        # Download dataset files
        kaggle.api.dataset_download_files(dataset, path=input_dir, unzip=unzip, force=force)

    # Create a PythonOperator to execute the dataset download task
    downloading_dataset = PythonOperator(
        task_id="download_aircraft_dataset",
        python_callable=download_dataset,
        op_kwargs={
            "dataset": "airbusgeo/airbus-aircrafts-sample-dataset",
            "input_dir": "/home/airflow/data",
            "unzip": True,
            "force": True,
        },
    )

    # Define a function to preprocess the downloaded dataset
    def preprocess_dataset(input_path, output_path):
        import ast
        import pandas as pd
        import numpy as np
        # Read the dataset
        df = pd.read_csv(input_path)
        # Convert the geometry column from string to dictionary
        df["geometry"] = df["geometry"].apply(lambda x: ast.literal_eval(x.rstrip('\r\n')))
        # Rename the class column to lowercase 'airplane'
        df["class"] = df["class"].apply(lambda x: x.lower())
        # Define a function to convert the geometry column to bbox
        def calculate_bbox(coords): 
            coords = np.array(coords).T
            return (np.min(coords[0]), np.min(coords[1]), np.max(coords[0]), np.max(coords[1]))
        # Convert the geometry column to bbox
        df["bbox"] = df["geometry"].apply(lambda x: calculate_bbox(x))
        # Create the width, height, x, y columns
        df["w"] = df["bbox"].apply(lambda coords: coords[2] - coords[0])
        df["h"] = df["bbox"].apply(lambda coords: coords[3] - coords[1])
        df["x"] = df["bbox"].apply(lambda coords: coords[0])
        df["y"] = df["bbox"].apply(lambda coords: coords[1])
        # Drop the geometry column
        df = df.drop(columns=["geometry"])
        # Save the preprocessed dataset
        df.to_csv(output_path, index=False)

    # Create a PythonOperator to execute the dataset preprocessing task
    preprocessing_dataset = PythonOperator(
        task_id="preprocess_aircraft_dataset",
        python_callable=preprocess_dataset,
        op_kwargs={
            "input_path": "/home/airflow/data/annotations.csv",
            "output_path": "/home/airflow/data/annotations.csv",
        },
    )

    # Set the task dependencies: downloading_dataset must run before preprocessing_dataset
    downloading_dataset >> preprocessing_dataset
