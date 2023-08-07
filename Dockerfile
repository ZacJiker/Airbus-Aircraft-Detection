# Start with the official Apache Airflow image
FROM apache/airflow:2.6.1-python3.9

# Set the user to root for system-level operations
USER root

# Accept the license agreement for msodbcsql18
ENV ACCEPT_EULA="Y"
RUN apt-get update && apt-get install -y msodbcsql18

# Install required system packages in a single RUN command
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        git \
    && rm -rf /var/lib/apt/lists/*

# Set the proper permission for airflow user
RUN chown -R airflow: /opt/airflow
RUN chown -R airflow: /home/airflow

# Switch back to the airflow user
USER airflow

# Create and set the working directory
WORKDIR /home/airflow

# Copy the requirements file and upgrade pip before installing Python packages
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt