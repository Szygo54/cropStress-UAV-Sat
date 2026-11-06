import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

dane = pd.read_csv("results/tables/baza_kalibracyjna_UAV_PlanetScope_FVC.csv")
dane.info()

print(dane.head(5))

X = dane[['PS_NDVI','PS_SAVI','PS_EVI','PS_RENDVI','PS_CIre','PS_NDWI','PS_PRI','FVC']]
y = dane['UAV_NDVI']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=67)


params_rf = {
    'n_estimators': [100, 200, 300],        
    'max_depth': [4, 6, 8, 10],              
    'min_samples_leaf': [2, 5, 10],          
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestRegressor(random_state=67)

cv = GridSearchCV(estimator=rf,param_grid=params_rf,cv=5,n_jobs=-1)
cv.fit(X_train,y_train)

best_estimator = cv.best_estimator_
best_params = cv.best_params_

y_pred = best_estimator.predict(X_test)
MSE = mean_squared_error(y_test,y_pred)
RMSE = MSE ** (1/2)

print(RMSE)