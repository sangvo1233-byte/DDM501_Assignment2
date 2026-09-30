# =============================================================================
# Airflow image with the pipeline's dependencies
# DDM501 - Lab 2
# =============================================================================
FROM apache/airflow:2.8.4-python3.11

# 1. Install as the airflow user: as root, pip writes to a site-packages the
#    scheduler never reads.
USER airflow

# 2. Extra packages, resolved against Airflow's own constraints so pip cannot
#    upgrade something Airflow depends on.
ARG AIRFLOW_VERSION=2.8.4
ARG PYTHON_VERSION=3.11
COPY requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# 3. docker-compose.yml mounts ./pipeline at /opt/airflow/project/pipeline
ENV PYTHONPATH=/opt/airflow/project
