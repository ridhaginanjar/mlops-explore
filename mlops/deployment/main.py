from fastapi import FastAPI, UploadFile, File

from model_preps import load_model, preprocess_image

app = FastAPI()

@app.get("/")
def main():
    return {"message": "API Ready to Predict!"}

@app.post("/predict")
async def prediction(file: UploadFile = File(...)):
    try:
        # Get and Prepare Data
        image_bytes = await file.read()
        image = preprocess_image(image_bytes)

        # Load Model
        return_model = load_model(run_id='189f736ce12a4e448b5ca5468c114d7d')

        prediction_prob = return_model.predict(image)
        prediction_result = (prediction_prob > 0.5).astype(int).reshape(-1)

        return {"result": prediction_result.astype(int).tolist()}
    except Exception as e:
        return {"error": str(e)}