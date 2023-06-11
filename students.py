"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
# Data wrangling
import pandas as pd
import numpy as np

# Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lazypredict.Supervised import LazyRegressor

# Set size for plot visualization
data = pd.read_csv("StudentScore.xls", delimiter=",")
# Sample data
print(data.head())
# List of all columns
print(data.info())
sns.histplot(data['math score'])
plt.title('Distribution of Math Score')
plt.savefig("student_math_score_dist.png")
# Set features and target
# x = data.drop("math score", axis=1)
x = data.drop("math score", axis=1)
y = data["math score"]
# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Preprocess data


# Scale numeric values
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

# Ordinal encode ordinal values
education_values = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree",
                    "master's degree"]
gender_values = data["gender"].unique()
lunch_values = data["lunch"].unique()
prep_values = data["test preparation course"].unique()
ord_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OrdinalEncoder(categories=[education_values, gender_values, lunch_values, prep_values],
                              handle_unknown='use_encoded_value', unknown_value=-1))])

# One-hot encode nominal values
nom_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num_features', num_transformer, ["reading score", "writing score"]),
        (
            'ord_features', ord_transformer,
            ["parental level of education", "gender", "lunch", "test preparation course"]),
        ('cat_features', nom_transformer, ["race/ethnicity"])])

# USE LINEAR REGRESSOR
# Train model
reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())])

reg.fit(x_train, y_train)
# Run prediction on test set
y_predict = reg.predict(x_test)

mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)

print("Mean absolute error {}".format(mae))
print("Mean squared error {}".format(mse))


# USE GridSearchCV (FOR SMALL DATASET)
reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())])

param_grid = {'regressor__n_estimators': [100, 200],
              'regressor__max_features': ['auto', 'sqrt'],
              'regressor__max_depth': [5, 10],
              'regressor__min_samples_split': [10, 50],
              'regressor__min_samples_leaf': [2, 5]}
grid_search = GridSearchCV(reg, param_grid=param_grid, cv=10)
grid_search.fit(x_train, y_train)
print("Best parameters: {}".format(grid_search.best_params_))
y_predict = grid_search.predict(x_test)

mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)

print("GridSearchCV: Mean absolute error {}".format(mae))
print("GridSearchCV: Mean squared error {}".format(mse))



# USE RandomizedSearchCV (FOR LARGE DATASET)
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [5, 10, 20, 30]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 50, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Random search of parameters, using 5-fold cross validation, search across 100 different combinations

random_grid = {'regressor__n_estimators': n_estimators,
               'regressor__max_features': max_features,
               'regressor__max_depth': max_depth,
               'regressor__min_samples_split': min_samples_split,
               'regressor__min_samples_leaf': min_samples_leaf}
random_search = RandomizedSearchCV(estimator=reg, param_distributions=random_grid,
                        scoring='neg_mean_squared_error', n_iter=10, cv=5,
                        verbose=1, random_state=42, n_jobs=1)
random_search.fit(x_train, y_train)
print("Best parameters: {}".format(random_search.best_params_))
y_predict = random_search.predict(x_test)

mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)

print("RandomizedSearchCV: Mean absolute error {}".format(mae))
print("RandomizedSearchCV: Mean squared error {}".format(mse))

# SEARCH FOR ALL REGRESSORS
reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(x_train, x_test, y_train, y_test)

print(models)
