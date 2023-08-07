# Aircraft Detection (using Airbus Dataset)

Aircrafts are usually seen on airports. Earth observation satellites like Airbus' Pleiades twin satellites acquire pictures of airports all over the world on a regular basis. Deep Learning can be used to detect automatically the number, size and type of aircrafts present on the site. In turn, this can provide information about the activity of any airport.

End-to-end machine learning pipeline for Airbus aircraft detection using Airflow, Docker, and MLFlow. The pipeline is based on the [Airbus Aircraft Detection Kaggle Competition](hhttps://www.kaggle.com/datasets/airbusgeo/airbus-aircrafts-sample-dataset). The pipeline is divided into 4 stages:

1. Data Downloading
2. Data Preprocessing
3. Training
4. Serving

After the training stage, the model will be saved in the MLFlow models registry. The model can be used for inference. You can use the web app to test the model at `http://localhost:8501/`.

## Pay Attention

You need a Kaggle account to download the dataset. You can create a Kaggle account [here](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2F). And, you need to create a Kaggle API token. You can create a Kaggle API token [here](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication).

## Getting Started (From Local)

For running the pipeline, you need to install the following dependencies: 

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Launching the Pipeline

1. Clone the repository

```
git clone https://github.com/ZacJiker/Airbus-Aircraft-Detection
```

2. Build the Docker image of custom airflow

```
docker build -t apache/airflow:2.6.1-custom .
```

3. Generate the `.env` file

```
bash scripts/generate_env.sh
```

4. Start the containers

```
docker-compose up -d
```

If used docker compose in version 2.0, you can use the following command:

```
docker-compose up -d
```

**Wait for a few minutes, then you can access the Airflow UI at `http://localhost:8080/` and MLFlow UI at `http://localhost:5000/`. After, you can run the DAGs in the Airflow UI.**
