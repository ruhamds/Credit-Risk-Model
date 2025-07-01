from fastapi import FastAPI
from src.api.pydantic_models import InputModel, OutputModel
from src.predict import predict_function

app = FastAPI()

@app.post("/predict", response_model=OutputModel)
def predict(input_data: InputModel):
    prediction, risk_prob = predict_function(input_data)
    return OutputModel(prediction=prediction, risk_probability=risk_prob)
