# State
Currently there is an error when running mlflow project because of conflict when using mlflow.start_run().

But when not specified it, the pipeline not tracked into "pipeline run" even tho it's finished.
It could be a BUG from MLFLOW: 
http://github.com/mlflow/mlflow/issues/4830