import requests #to make python http requests
nasa_api = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
params={
    "query":"select * from q1_q17_dr25_koi",
    "format":"csv"
} #we are telling nasa what we need
response=requests.get(nasa_api,params=params) #we are making a get request to the nasa api
response.raise_for_status() #this will raise an error if the request was unsuccessful

with open("data/kepler_data.csv","wb")as file:
    file.write(response.content) #we are writing the response content to a file
print("Data downloaded successfully!") #we are printing a message to the console
import pandas as pd

df = pd.read_csv("data/kepler_data.csv")

print("TOTAL COLUMNS:", len(df.columns))
print()

for i, col in enumerate(df.columns):
    print(i, col)