from fastapi import FastAPI
from app.schemas import TextRequest, PredictionResponse
from app.model import predict

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Sarcasm Detection API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict_api(req: TextRequest):
    prob = predict(req.text)
    label = "Sarcastic" if prob > 0.5 else "Not Sarcastic"

    return PredictionResponse(
        probability=prob,
        label=label
    )