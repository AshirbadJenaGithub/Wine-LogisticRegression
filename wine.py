from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_wine
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
wine=load_wine()
data=pd.DataFrame(wine.data,columns=wine.feature_names)
x=data.drop('ash',axis=1)
y=data['ash']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
model.fit(x_train, y_train_encoded)
prediction=model.predict(x_test)
accuracy=mean_absolute_error(prediction,y_test)
