# Heart Disease Prediction System
## Introduction
This project aims to predict the likelihood of heart disease occurrence in patients using machine learning techniques. The project follows the full scale of MLOps best practices, including the use of Python zenml to build the pipelines and mlflow for experiment tracking.

## Problem Statement
Heart disease is a leading cause of death globally, taking an estimated 17.9 million lives each year, representing 32% of global deaths. With the power of data, we get to investigate and determine what lifestyle habits or conditions mostly contribute to one’s likelihood of suffering from heart disease. This project seeks to leverage machine learning algorithms to predict the presence of heart disease in patients.

## Aim and Objective
The main aim of this project is to predict heart disease occurrence with the highest accuracy. To achieve this, we will test several classification algorithms. This section includes all results obtained from the study and introduces the best performer according to the accuracy metric.

## Dataset Used
The dataset used in this project is the Heart Disease Dataset, which is available on Kaggle [here](https://www.kaggle.com/datasets/amirmahdiabbootalebi/heart-disease). The dataset contains 303 rows and 14 columns, with each row representing a patient and each column representing an attribute. The attributes include age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, ST depression induced by exercise relative to rest, the slope of the peak exercise ST segment, number of major vessels coloured by fluoroscopy, thalassemia, and target. The target column indicates whether the patient has heart disease or not.

Also the system is set to fetch data from a database in real-time.

This repository aims to demonstrate how [ZenML](https://github.com/zenml-io/zenml) can help you build and deploy machine learning pipelines in a variety of ways. It offers a framework and template to base your work on, integrates with tools like [MLflow](https://mlflow.org/) for deployment, tracking, and more, and allows you to build and deploy your machine learning pipelines with ease.

Starting with ZenML 0.20.0, ZenML comes bundled with a React-based dashboard that allows you to observe your stacks, stack components, and pipeline DAGs in a dashboard interface. To access this, you need to [launch the ZenML Server and Dashboard locally](https://docs.zenml.io/user-guide/starter-guide#explore-the-dashboard). First, you must install the optional dependencies for the ZenML server:

```bash
pip install zenml["server"]
```
then 
```bash
zenml up
```

If you are running the `run_deployment.py` script, you will also need to install some integrations using ZenML: 
Note: in the process of running this a package called Daemon functionality is currently not supported on Windows.

```bash
zenml integration install mlflow -y
```

The project can only be executed with a ZenML stack that has an MLflow experiment tracker and model deployer as a component. Configuring a new stack with the two components are as follows:

```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker_heart --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack_heart -a default -o default -d mlflow -e mlflow_tracker_heart --set
```

## :thumbsup: The Solution

In order to build a real-world workflow for predicting or detecting the presence of heart disease in patients (which will help make better decisions), it is not enough to just train the model once.

Instead, I built an end-to-end pipeline for continuously predicting and deploying the machine learning model, alongside a data application that utilizes the latest deployed model.

This pipeline can be deployed to the cloud, scale up according to our needs, and ensure that we track the parameters and data that flow through every pipeline that runs. It includes raw data input, features, results, the machine learning model and model parameters, and prediction outputs. ZenML helps us to build such a pipeline in a simple, yet powerful, way.

In this Project, I give special consideration to the [MLflow integration](https://github.com/zenml-io/zenml/tree/main/examples) of ZenML. In particular, I utilize MLflow tracking to track our metrics and parameters, and MLflow deployment to deploy our model. I also use [Streamlit](https://streamlit.io/) to showcase how this model will be used in a real-world setting.

### Training Pipeline

The standard training pipeline consists of several steps:

- `ingest_data`: This step will ingest the data and create a `DataFrame`.
- `clean_data`: This step will clean the data and remove the unwanted columns.
- `train_model`: This step will train the model and save the model using [MLflow autologging](https://www.mlflow.org/docs/latest/tracking.html).
- `evaluation`: This step will evaluate the model and save the metrics -- using MLflow autologging -- into the artifact store.

### Deployment Pipeline

I have another pipeline, the `deployment_pipeline.py`, that extends the training pipeline, and implements a continuous deployment workflow. It ingests and processes input data, trains a model and then (re)deploys the prediction server that serves the model if it meets our evaluation criteria. The criteria that I have chosen is a configurable threshold on the [F1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) of the training. The first four steps of the pipeline are the same as above, but I have added the following additional ones:

- `deployment_trigger`: The step checks whether the newly trained model meets the criteria set for deployment.
- `model_deployer`: This step deploys the model as a service using MLflow (if deployment criteria is met).

In the deployment pipeline, ZenML's MLflow tracking integration is used for logging the hyperparameter values and the trained model itself and the model evaluation metrics -- as MLflow experiment tracking artifacts -- into the local MLflow backend. This pipeline also launches a local MLflow deployment server to serve the latest MLflow model if its accuracy is above a configured threshold.

The MLflow deployment server runs locally as a daemon (daemon functionality is currently not supported on Windows) process that will continue to run in the background after the example execution is complete. When a new pipeline is run which produces a model that passes the accuracy threshold validation, the pipeline automatically updates the currently running MLflow deployment server to serve the new model instead of the old one.

### Future Work
To round it off, I will deploy a Streamlit application that consumes the latest model service asynchronously from the pipeline logic. This can be done easily with ZenML within the Streamlit :

```python
service = prediction_service_loader(
   pipeline_name="continuous_deployment_pipeline",
   pipeline_step_name="mlflow_model_deployer_step",
   running=False,
)
...
service.predict(...)  
```

While this ZenML Project trains and deploys a model locally, other ZenML integrations such as the [Seldon](https://github.com/zenml-io/zenml/tree/main/examples/seldon_deployment) deployer can also be used in a similar manner to deploy the model in a more production setting (such as on a Kubernetes cluster). I use MLflow here for the convenience of its local deployment.

![training_and_deployment_pipeline](_assets/training_and_deployment_pipeline_updated.png)

## :notebook: Diving into the code

You can run two pipelines as follows:

- Training pipeline:

```bash
python run_pipeline.py
```

- The continuous deployment pipeline:

```bash
python run_deployment.py
```

## Resources & References

A blog on [Predicting how a customer will feel about a product before they even ordered it](https://blog.zenml.io/customer_satisfaction/).

You can watch a video by 
Ayush Singh [video](https://youtu.be/L3_pFTlF9EQ).

[customer-satisfaction-mlops](https://github.com/ayush714/customer-satisfaction-mlops)
