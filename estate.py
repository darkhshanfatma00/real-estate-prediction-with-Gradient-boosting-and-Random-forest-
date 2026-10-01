import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score,precision_score,mean_absolute_error,root_mean_squared_error,mean_squared_error,accuracy_score,recall_score,f1_score


# data creation

np.random.seed(42)
area_sqrt=np.random.randint(500,4000,200)
bedrooms=np.random.randint(1,6,200)
bathrooms=np.random.randint(1,5,200)
age_years=np.random.randint(0,30,200)
location_score=np.random.randint(1,10,200)
parking_spaces=np.random.randint(0,3,200)
price = (
    area_sqrt * 100
    + bedrooms * 20000
    + bathrooms * 15000
    + location_score * 30000
    + parking_spaces * 10000
    - age_years * 2000
    + np.random.randint(-20000, 20001, 200)
)



# data insertion


data=pd.DataFrame({
    "area_sqrt":area_sqrt,
    "bedrooms":bedrooms,
    "bathrooms":bathrooms,
    "age_years":age_years,
    "location_score":location_score,
    "parking_spaces":parking_spaces,
    "price":price,

    
})

shape_of_data=data.shape
print(shape_of_data)
first_rows=data.head(100)
print(first_rows)
last_rows=data.tail(100)
print(last_rows)
statistics_information=data.describe()
print(statistics_information)
data_information=data.info()
print(data_information)
missing_values=data.isna()
print(missing_values)
count_missing_values=data.isna().sum()
print(count_missing_values)




# algorithms green forest 
# classifier 

pipeline_classification=Pipeline([
    ("scaler",StandardScaler()),
    ("model1",RandomForestClassifier(n_estimators=20,max_depth=5,min_samples_leaf=5,min_samples_split=15))
])

X_classifier=data[[  "area_sqrt","bedrooms", "bathrooms","age_years","location_score","parking_spaces"]]
data["is_expensive"]=(data["price"]>=300000).astype(int)
print(data["is_expensive"].value_counts())
Y_classifier=data["is_expensive"]
X_classifier_train,X_classifier_test,Y_classifier_train,Y_classifier_test=train_test_split(X_classifier,Y_classifier,test_size=0.2,random_state=5)


pipeline_classification.fit(X_classifier_train,Y_classifier_train)
prediction_classification=pipeline_classification.predict(X_classifier_test)
print(prediction_classification)

accuracy=accuracy_score(Y_classifier_test,prediction_classification)
print(accuracy)
f1=f1_score(Y_classifier_test,prediction_classification)
print(f1)
precision=precision_score(Y_classifier_test,prediction_classification)
print(precision)
recall=recall_score(Y_classifier_test,prediction_classification)
print(recall)


# regression 

pipeline_regression=Pipeline([
    ("scaler",StandardScaler()),
    ("model1",RandomForestRegressor(n_estimators=20,max_depth=5,min_samples_leaf=5,min_samples_split=15))
])

X_regression=data[[  "area_sqrt","bedrooms", "bathrooms","age_years","location_score","parking_spaces"]]
Y_regression=data["price"]
X_regression_train,X_regression_test,Y_regression_train,Y_regression_test=train_test_split(X_regression,Y_regression,test_size=0.2,random_state=5)


pipeline_regression.fit(X_regression_train,Y_regression_train)
prediction_regression=pipeline_regression.predict(X_regression_test)
print(prediction_regression)

mse=mean_squared_error(Y_regression_test,prediction_regression)
print(mse)
mae=mean_absolute_error(Y_regression_test,prediction_regression)
print(mae)
rmse=root_mean_squared_error(Y_regression_test,prediction_regression)
print(rmse)
r2=r2_score(Y_regression_test,prediction_regression)
print(r2)