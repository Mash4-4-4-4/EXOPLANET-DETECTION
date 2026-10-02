from fastapi import FastAPI,UploadFile, File,HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib 
import io

from ml_core import prepare_input, predict

app_data={}
@asynccontextmanager
async def lifespan(app:FastAPI):
    try:
        print("loading model")
        app_data["model_package"]=joblib.load("exo_model_v7.pkl")
        print("model loaded")
    except FileNotFoundError:
        print("model not found")
        raise HTTPException(status_code=500, detail="Model file not found.")
    yield #the code after yield will run when the app is shutting down
    
    app_data.clear()
    print("app_data cleared")
    
    
app=FastAPI(lifespan=lifespan,title="Exoplanet Prediction API", description="API for predicting exoplanets using a pre-trained model.", version="1.0.0") #this will call the lifespan function when the app starts and stops

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/predict")
async def predict_exoplanet(file:UploadFile=File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")
    try:
        contents=await file.read()
        data=pd.read_csv(io.BytesIO(contents))
        
        model_pkg=app_data["model_package"]
        df_prepared=prepare_input(data,model_pkg["features"])
        
        prediction,probability=predict(model_pkg,df_prepared)
        
        results=[]
        for i in range(len(prediction)):
            prob_dict={
            model_pkg["label_encoder"].inverse_transform([j])[0]: round(probability[i][j]*100, 2) 
            for j in range(len(probability[i]))
        }
            results.append({"row_index": i,
              "kepid": data.iloc[i]["kepid"] if "kepid" in data.columns else None,
              "prediction": prediction[i], 
              "probabilities": prob_dict})
            
            return{
                "file_name": file.filename,
                "predictions": results
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred while processing the file: {str(e)}")
