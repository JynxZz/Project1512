# Setup Server MLflow Docker Version
___

## DockerFile

Create your `Dockerfile_mlflow_server`
```dockerfile
# Base Image
FROM python:3.10.6-slim

# Set Working Dir
WORKDIR /mlflow_server

# Pip Install
RUN python3 -m pip install --upgrade pip
RUN pip install mlflow

# Expose Port - 5000
EXPOSE 5000

# Define Env Var
ENV MLFLOW_HOME /mlflow_server

# Run the Server
CMD mlflow server --host 0.0.0.0 --port $PORT --backend-store-uri ./mlruns --default-artifact-root ./mlartifacts
```

## Build Docker Image

Copy this commande to build your image for local

```bash
docker build -t $DOCKER_NAME -f Dockerfile_mlflow_server .
```

Create this one if you are a M-Chip user

```bash
docker build --platform linux/amd64 -t $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$ARTIFACT_REPOSITORY/$DOCKER_NAME:intel -f Dockerfile_mlflow_server .
```

## Run It Locally

Copy this image to run it locally

```bash
docker run --rm -e PORT=5000 -p 8000:5000 $DOCKER_NAME
```

You can connect on your localhost on the 8000 port to view your server.

```bash
http://localhost:8000
```

## Run It in Cloud

Create an image with the the gcp project name

```bash
docker image tag $DOCKER_NAME:intel $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$ARTIFACT_REPOSITORY/$DOCKER_NAME
```

Create an artifacts repository to store our mlflow servers

```bash
gcloud artifacts repositories create $ARTIFACT_REPOSITORY --repository-format=docker --location=$GCP_REGION  --description="Repository to store mlflow servers"
```

Then you can push it in GCP

```bash
docker push $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$ARTIFACT_REPOSITORY/$DOCKER_NAME:intel
```

Now we can deploy in Google Cloud Run

```bash
gcloud run deploy $ARTIFACT_REPOSITORY --image $GCP_REGION-docker.pkg.dev/$GCP_PROJECT/$ARTIFACT_REPOSITORY/$DOCKER_NAME:intel --platform managed --memory $MEMORY --allow-unauthenticated --region $GCP_REGION --env-vars-file .env.yaml
```

Save your URL app

```plain text
https://own-mlflow-server-fbc5geh47a-ew.a.run.app/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D
```
