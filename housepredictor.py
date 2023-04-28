#Code By Abubakar Asif=>abubakarasif3111@gmail.com
#AbubakarAsif FA20-BCE-013
#abubakarasif3111@gmail.com
#https://github.com/Abubakar3111
#https://www.linkedin.com/in/abubakar-asif-b3b94021a/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as LA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
data = pd.read_csv('house.csv')
x1 = data['MarlaPrice'].values
x2 = data['TotalMarla'].values
x3 = data['Room'].values
x4 = data['Floors'].values
x5 = data['Washroom'].values
x6 = data['CarPosh'].values
x7 = data['Lawn'].values
Y = data['FinalPrice'].values
m = len(x1)
x1 = x1.reshape(m)
x2 = x2.reshape(m)
x3 = x3.reshape(m)
x4 = x4.reshape(m)
x5 = x5.reshape(m)
x6 = x6.reshape(m)
x7 = x7.reshape(m)
x0=np.ones(m)
X=np.array([x0,x1,x1**2,x2,x2**2,x3,x3**2,x4,x4**2,x5,x5**2,x6,x6**2,x7,x7**2]).T
reg=LinearRegression()
reg.fit(X,Y)
h_theta=reg.predict(X)
print("Code By Abubakar Asif,FA20-BCE-013")
print("\n_______PRICE_PREDICTION_OF_HOUSES____________________________")
print("\nData Score:",reg.score(X,Y))
MarlaPrice=850000
TotalMarla=12
Room=10
Floors=2
Washroom=9
Carposh=1
Lawn=1
presal=reg.predict( [[1,MarlaPrice,MarlaPrice**2,TotalMarla,TotalMarla**2,Room,Room**2,Floors,Floors**2,Washroom,Washroom**2,Carposh,Carposh**2,Lawn,Lawn**2]])

print("\nPrediction For House having:")
print("MarlaPrice in Rs:",MarlaPrice)
print("Total_Marla:",TotalMarla)
print("Total_Rooms:",Room)
print("Total_Floors:",Floors)
print("Total_Washrooms:",Washroom)
print("Carposh:",Carposh)
print("Lawn:",Lawn)
round_off_values = np.round_(presal, decimals = -3)
print("\nPrice Predicted:")
print(round_off_values[0],"Rupees only")