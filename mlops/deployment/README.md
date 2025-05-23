# Deployment
After getting model that registered on MLFLow Registry, we can try to create a prediction by build it as an API.

## Serving Pipeline
This pipeline has responsibility to take data input, running a predictions, and getting an output.

There are two types of predictions:
- Online Predictions: Happen in realtime, typically by sending a request to an online server and returning a prediction.
- Offline Predictions: Precomputed and cached. The predictions will be running on specific time (batch) and then store it as a cache. The applications will be taking the predictions from a cache.


# TODO:
We are going to make an online predictions by building it as an API (using FastAPI)