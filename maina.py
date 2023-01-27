import pickle
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware




pickle_in = open('./RandomForest.pkl', 'rb') 
modelRandomForest = pickle.load(pickle_in)

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict/")

async def predict(State : str = "CA",
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
    
    data = {"State": State,
    "BankState": BankState,
    "NAICS" : NAICS,
    "Term" : Term,
    "NoEmp" : NoEmp,
    "NewExist" : NewExist,
    "CreateJob": CreateJob,
    "RetainedJob": RetainedJob,
    "FranchiseCode" : FranchiseCode,
    "UrbanRural" :UrbanRural,
    "RevLineCr" : RevLineCr,
    "LowDoc":LowDoc,
    "DisbursementGross": DisbursementGross}
    pred=pd.DataFrame(dict(data),index = [0])


    class_pred = modelRandomForest.predict(pred)[0]
    if class_pred == "P I F":
        return {"Class":"Votre crédit à était accépté , toute mes félicitations"}
    else:
        return {"Class": "Votre crédit à était refusé , merci de revoir vos ambitions à la baisse"}
   