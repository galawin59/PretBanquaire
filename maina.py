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
    "http://localhost:8080",
]



pickle_in = open('./RandomForest.pkl', 'rb') 
modelRandomForest = pickle.load(pickle_in)

app = FastAPI()



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

@app.get("/predict")

async def predict( State : str = "CA",
    BankState: str = "CA",
    NAICS : str = "Other services (except public administration)",
    Term : int = 228,
    NoEmp : int = 6,
    NewExist : int = 1,
    CreateJob : int = 0,
    RetainedJob: int = 0,
    FranchiseCode : int = 1,
    UrbanRural :int = 0,
    RevLineCr : int = 0,
    LowDoc:int = 0,
    DisbursementGross: float = 540000.0):
    
    data = {"BankState": BankState,
    "NAICS" : NAICS,
    "Term" : Term,
    "NoEmp" : NoEmp,
    "NewExist" : NewExist,
    "CreateJob ": CreateJob,
    "RetainedJob": RetainedJob,
    "FranchiseCode" : FranchiseCode,
    "UrbanRural" :UrbanRural,
    "RevLineCr" : RevLineCr,
    "LowDoc":LowDoc,
    "DisbursementGross": DisbursementGross}
    pred=pd.DataFrame(dict(data),index = [0])


    class_idx = modelRandomForest.predict(pred)[0]

    return {'class':class_idx}