import numpy as np
import pandas as pd
aws=r"C:\Users\nauti\Downloads\dataset_1.data"
df=pd.read_csv(aws)

headers_data = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration",
                "num-of-doors", "body-style", "drive-wheels", "engine-location",
                "wheel-base", "length", "width", "height", "curb-weight",
                "engine-type", "num-of-cylinders", "engine-size", "fuel-system",
                "bore", "stroke", "compression-ratio", "horsepower",
                "peak-rpm", "city-mpg", "highway-mpg", "price"]
df.columns=headers_data

df.replace("?",np.nan,inplace=True)

"""" #   Column             Non-Null Count  Dtype  

 0   symboling          204 non-null    int64  
 1   normalized-losses  164 non-null    str    
 2   make               204 non-null    str    
 3   fuel-type          204 non-null    str    
 4   aspiration         204 non-null    str    
 5   num-of-doors       202 non-null    str    
 6   body-style         204 non-null    str    
 7   drive-wheels       204 non-null    str    
 8   engine-location    204 non-null    str    
 9   wheel-base         204 non-null    float64
 10  length             204 non-null    float64
 11  width              204 non-null    float64
 12  height             204 non-null    float64
 13  curb-weight        204 non-null    int64  
 14  engine-type        204 non-null    str    
 15  num-of-cylinders   204 non-null    str    
 16  engine-size        204 non-null    int64  
 17  fuel-system        204 noprint(df.info())n-null    str    
 18  bore               200 non-null    str    
 19  stroke             200 non-null    str    
 20  compression-ratio  204 non-null    float64
 21  horsepower         202 non-null    str    
 22  peak-rpm           202 non-null    str    
 23  city-mpg           204 non-null    int64  
 24  highway-mpg        204 non-null    int64  
 25  price              200 non-null    str    """
df.dropna(subset=["price"],inplace=True,axis=0)
df.reset_index(drop=True,inplace=True)
print(df.info())
from sklearn.impute import SimpleImputer as s
meanimpute=s(missing_values=np.nan,strategy="mean")
df[["normalized-losses","bore","stroke","peak-rpm","horsepower"]]=meanimpute.fit_transform(df[["normalized-losses","bore","stroke","peak-rpm","horsepower"]]).astype("object")
print(df.info())          
mostfreq=s(missing_values=np.nan,strategy="most_frequent")
df[["num-of-doors"]]=mostfreq.fit_transform(df[["num-of-doors"]]).astype("object")



df[["normalized-losses","bore","stroke","peak-rpm","horsepower","price"]]=df[["normalized-losses","bore","stroke","peak-rpm","horsepower","price"]].astype("float")        
list1=["make","fuel-type","aspiration","body-style","drive-wheels","engine-location","engine-type","fuel-system"]
for i in list1:
    enc=pd.get_dummies(df[i])
    df=pd.concat([df,enc],axis=1)
    df.drop(i,axis=1,inplace=True)

from sklearn.preprocessing import LabelEncoder as li
le=li()
l2=["num-of-doors","num-of-cylinders"]
for l in l2:
     
  df[l]=le.fit_transform(df[l])
  print(df.info())
features= list(df.columns)
features.remove("price")
X=df[features]

y=df.price
from sklearn.model_selection import train_test_split as tae
train_X,val_X,train_y,val_y=tae(X,y,random_state=1)
from sklearn.ensemble import RandomForestRegressor as r
from sklearn.metrics import mean_absolute_error
model3=r(random_state=1)
model3.fit(train_X,train_y)
k2=model3.predict(val_X)
print(mean_absolute_error(val_y,k2))
