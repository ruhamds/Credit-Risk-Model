from pydantic import BaseModel

class InputModel(BaseModel):
    recency: int
    frequency: int
    monetary: float

class OutputModel(BaseModel):
    prediction: int
    risk_probability: float
