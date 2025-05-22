# Introductions
We are going to reproduce 4 types of pipeline:
- Data pipeline
- Training pipeline
- Validation pipeline
- Serving pipeline

The stack that used is Prefect.

# Validation Pipeline
The objective of validation pipeline is to validate several component.
- Data Validation
- Training Validation
- Model Validation

## Data Validation
This pipeline is to validate our data that gonna be used for create/update model machine learning.
The validation has three distinct checks on our data:
- Check for data anomalies.
- Check that the data schema hasn’t changed.
- Check that the statistics of our new datasets still align with statistics from our previous training datasets.

# About MLFlow
Currently there is an error when running mlflow project because of conflict when using mlflow.start_run().

But when not specified it, the pipeline not tracked into "pipeline run" even tho it's finished.
It could be a BUG from MLFLOW: 
http://github.com/mlflow/mlflow/issues/4830

# Inisight
There are some types of pipeline that can produce for machine learning systems. The pipeline can be categorized with two functions:
- Delivered predictions: Type of pipeline that orchestrate when user request, make predictions, and returning predictions result.
- Create/update models: Type of pipeline that create and updating existing models.

Based on the two functions above, we can define 4 types of pipelinse:
- Data pipeline: Type of pipeline that reproduce data for training models.
- Training Pipeline: Type of pipeline that retraining models.
- Validation Pipeline: Type of pipeline that validate the new models.
- Serving Pipeline: Type of pipeline that running predictions to our models.