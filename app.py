# -*- coding: utf-8 -*-

# 1. Library imports
import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
import ccxt
from xgboost import XGBRegressor


# 2. Create the app object
nft_data = pd.read_csv("nft.csv")

pickle_in = open("model.pkl","rb")
model = pickle.load(pickle_in)

app = FastAPI()


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'Free tool to obtain accurate NFTs fair price'}

@app.get('/nft')
def get_name(token: int, opensea_volume: int):
    row = nft_data.loc[nft_data['token_id'] == token].values
    row[0][0] = opensea_volume
    return float(model.predict(row)[0])
    
# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload