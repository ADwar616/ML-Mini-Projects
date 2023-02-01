import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.Linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df=quandl.get('FRED/NROUST')

#Features->The features are the descriptive attributes
df=df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]     #Adding columns
df['HL_PCT']=(df['Adj. High']-df['Adj. Low'])/df['Adj. Close']*100     #Adding columns
df['PCT_change']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100     #Adding columns

df=df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]      #Features

forecast_column='Adj. Close'
df.fillna(-99999, inplace=True)                    #NAN value->Not a Value datapoint
                                                   #fillna->If you will want to replace missing values in a Pandas DataFrame instead of dropping it completely.
                                                          # The fillna method is designed for this.

forecast_out=int(math.ceil(0.01*len(df)))             #0.1->Predicting the data for 10 days
                                                     #0.01->Predicting the data for 1 day

#label->the label is what you're attempting to predict or forecast
df['label']=df[forecast_column].shift(-forecast_out)     #Adding columns
df.dropna(inplace=True)     #dropna-> When applied against a DataFrame, the dropna method will remove any rows that contain a NaN value.

#It is a typical standard with machine learning in code to define X (capital x), as the features, and y (lowercase y) as the label that corresponds to the features
X=np.array(df.drop(['label'],1))     #defined X (features), as our entire dataframe EXCEPT for the label column, converted to a numpy array
y=np.array(df['label'])     #we define our y variable, which is our label, as simply the label column of the dataframe, converted to a numpy array.

X=preprocessing.scale(X)
y=np.array(df['label'])

print(len(X),len(y))

