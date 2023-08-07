# Aircraft Detection Using Airbus Dataset

Aircraft are frequently observed at airports around the world. Earth observation satellites, such as Airbus' Pleiades twin satellites, capture images of airports regularly. Deep Learning techniques can be leveraged to automatically detect and identify aircraft in these images, providing insights into airport activity by determining the number, size, and type of aircraft present.

This documentation outlines an end-to-end machine learning pipeline for detecting Airbus aircraft using Airflow, Docker, and MLFlow. The pipeline is built upon the [Airbus Aircraft Detection Kaggle Competition](https://www.kaggle.com/datasets/airbusgeo/airbus-aircrafts-sample-dataset) dataset and comprises four key stages:

1. Data Downloading
2. Data Preprocessing
3. Model Training
4. Model Serving

Upon completion of the training stage, the resulting model is saved in the MLFlow models registry and can be used for inference. A web app is also provided for model testing, accessible at `http://localhost:8501/`.

## Important Notes

To successfully utilize this pipeline, please ensure the following:

- You possess a Kaggle account to download the required dataset. If not, you can create one [here](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2F).
- A Kaggle API token is necessary. You can generate one [here](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication).

## Getting Started (Local Deployment)

To run the pipeline locally, ensure you have the following dependencies installed:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Launching the Pipeline

1. Clone the repository:

```bash
git clone https://github.com/ZacJiker/Airbus-Aircraft-Detection
```

2. Generate the `.env` file:

```bash
bash scripts/generate_env.sh
```

3. Build the custom Airflow Docker image:

```bash
docker build -t apache/airflow:2.6.1-custom .
```

4. Start the containers:

```bash
docker-compose up -d
```

**Please allow a few minutes for initialization. You can access the Airflow UI at http://localhost:8080/ and the MLFlow UI at http://localhost:5000/. From here, you can execute the DAGs within the Airflow UI.**

## Conclusion 

This documentation provides you with a comprehensive overview of setting up an end-to-end aircraft detection pipeline using the Airbus Dataset. By leveraging Deep Learning, Airflow, Docker, and MLFlow, you can efficiently preprocess data, train a model, and deploy it for inference. Analyzing airport activity through aircraft detection becomes both automated and insightful with this powerful pipeline.