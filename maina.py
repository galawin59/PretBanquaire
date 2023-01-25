import pickle
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import json
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8000",
]



pickle_in = open('./RandomForest.pkl', 'rb') 
modelRandomForest = pickle.load(pickle_in)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)

class request_body(BaseModel):
    State : str
    BankState: str
    NAICS : str
    Term : int
    NoEmp : int
    NewExist : int
    CreateJob : int
    RetainedJob: int
    FranchiseCode : int
    UrbanRural :int
    RevLineCr : int
    LowDoc:int
    DisbursementGross: float

@app.post("/predict")

def predict(data : request_body):
    
    
    pred=pd.DataFrame(dict(data),index = [0])


    class_idx = modelRandomForest.predict(pred)[0]

    return {'class':class_idx}