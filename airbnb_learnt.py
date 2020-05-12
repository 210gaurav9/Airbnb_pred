import numpy as np 
import pandas as pd 
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import pickle
"""# Predictions"""
# Preparing dataset for predictions
data_pred = pd.read_csv('airbnb.csv')
data_pred['last_review'] = pd.to_datetime(data_pred['last_review'],infer_datetime_format=True)
data_pred.drop(['name', 'host_name'], inplace=True, axis=1)
"""Threfore all the null values in last reviews and reviews per month is due to the number of reviews"""
data_pred['reviews_per_month'] = data_pred['reviews_per_month'].fillna(0)
earliest = min(data_pred['last_review'])
data_pred['last_review'] = data_pred['last_review'].fillna(earliest)
"""Here i would call it as magic..."""
data_pred['last_review'] = data_pred['last_review'].apply(lambda x: x.toordinal() - earliest.toordinal())
"""well here i tried to convert the dates into the count of the days from the "earliest" date..."""
data_pred = data_pred[np.log1p(data_pred['price']) < 8]
data_pred = data_pred[np.log1p(data_pred['price']) > 3]
data_pred['price'] = np.log1p(data_pred['price'])
"""# Predictor distributions (Independent Variables Distribution)
## Predictors
"""
data_pred = data_pred.drop(['host_id', 'id'], axis=1)
"""## Reviews per Month
Since we found some outliers in the dataset for reviews per month..we need to remove it..
"""
data_pred['reviews_per_month'] = data_pred[data_pred['reviews_per_month'] < 17.5]['reviews_per_month']
data_pred['reviews_per_month'] = data_pred['reviews_per_month'].fillna(0)
"""## Availability 365"""
data_pred['all_year_avail'] = data_pred['availability_365']>353
data_pred['low_avail'] = data_pred['availability_365']< 12
data_pred['no_reviews'] = data_pred['reviews_per_month']==0
"""## Now lets try and see the correlation"""
data_pred.sort_values('price', ascending=True, inplace=True)
"""## Encoding categorical features"""
categorical_features = data_pred.select_dtypes(include=['object'])
categorical_features_one_hot = pd.get_dummies(categorical_features)
"""Saving the transformed dataframe"""
numerical_features =  data_pred.select_dtypes(exclude=['object'])
y = numerical_features.price
numerical_features = numerical_features.drop(['price'], axis=1)
X = np.concatenate((numerical_features, categorical_features_one_hot), axis=1)
X_df = pd.concat([numerical_features, categorical_features_one_hot], axis=1)
"""# Train test split"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""# Rescaling
Taking Robust Scaler so that the outliers are handelled if any..
"""
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
"""# Trying out different models
Apllying cross validation..
"""
# squared_loss
"""Best model"""
xbgreg_best = XGBRegressor(n_estimators=500, learning_rate=0.05, early_stopping=5, max_depth=5, min_child_weight=5 )
xbgreg_best.fit(X_train, y_train) 
y_test_xgbreg = xbgreg_best.predict(X_test)
"""However can be better if more n_estimators are used.."""
xgboost.save_model("airbnb.model")