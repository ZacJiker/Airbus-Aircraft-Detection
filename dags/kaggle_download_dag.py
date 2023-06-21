from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def download_dataset(dataset, output_path):
    import kaggle
    # Download dataset from Kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset, path=output_path, unzip=True)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2021, 1, 1),
}

with DAG('kaggle_download_dag', default_args=default_args, schedule='@daily', catchup=False) as dag:
    download_dataset = PythonOperator(
        task_id='download_dataset',
        python_callable=download_dataset,
        op_kwargs={
            'dataset': 'airbusgeo/airbus-aircrafts-sample-dataset',
            'output_path': '/opt/airflow/data/airbus-aircrafts-sample-dataset'
        }
    )