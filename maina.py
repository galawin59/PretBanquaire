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

async def predict(
    NAICS : str = "Other services (except public administration)",
    Term : int = 228,
   
    NewExist : int = 1,
    
    FranchiseCode : int = 1,
    UrbanRural :int = 0,
    RevLineCr : int = 0,
    LowDoc:int = 0,
    GrAppv: float = 540000.0):
    
    data = {
    "NAICS" : NAICS,
    "Term" : Term,
   
    "NewExist" : NewExist,
  
    "FranchiseCode" : FranchiseCode,
    "UrbanRural" :UrbanRural,
    "RevLineCr" : RevLineCr,
    "LowDoc":LowDoc,
    "GrAppv": GrAppv}
    pred=pd.DataFrame(dict(data),index = [0])


    class_pred = modelRandomForest.predict(pred)[0]
    if class_pred == "P I F":
        return {"Class":"Votre crédit à était accépté , toute mes félicitations"}
    else:
        return {"Class": "Votre crédit à était refusé , merci de revoir vos ambitions à la baisse"}
   