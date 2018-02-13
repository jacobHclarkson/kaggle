# House Prices Submission

# data analysis and wrangling
import pandas as pd
import numpy as np

# ml
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from vecstack import stacking
from sklearn.metrics import mean_absolute_error

# visualization
import matplotlib.pyplot as plt

print()

# read data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# delete outliers
train_df = train_df.drop(train_df[train_df['Id'] == 1299].index)
train_df = train_df.drop(train_df[train_df['Id'] == 582].index)
train_df = train_df.drop(train_df[train_df['Id'] == 1191].index)
train_df = train_df.drop(train_df[train_df['Id'] == 1062].index)
train_df = train_df.drop(train_df[train_df['Id'] == 524].index)
train_df = train_df.drop(train_df[train_df['Id'] == 524].index)

# fix distributions
train_df['SalePrice'] = np.log(train_df['SalePrice'])
train_df['1stFlrSF'] = np.log(train_df['1stFlrSF'])
train_df['GrLivArea'] = np.log(train_df['GrLivArea'])
train_df['TotalBsmtSF'] = np.log1p(train_df['TotalBsmtSF'])
train_df['2ndFlrSF'] = np.log1p(train_df['2ndFlrSF'])
test_df['1stFlrSF'] = np.log(test_df['1stFlrSF'])
test_df['GrLivArea'] = np.log(test_df['GrLivArea'])
test_df['TotalBsmtSF'] = np.log1p(test_df['TotalBsmtSF'])
test_df['2ndFlrSF'] = np.log1p(test_df['2ndFlrSF'])

# combine data sets
n_train = train_df.shape[0]
n_test = test_df.shape[0]
y_train = train_df.SalePrice.values
combine_df = pd.concat((train_df, test_df)).reset_index(drop=True)
combine_df.drop(['SalePrice'], axis=1, inplace=True)

# investigate missing data
total = combine_df.isnull().sum().sort_values(ascending=False)
percent = (combine_df.isnull().sum() /
           combine_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

# dealing with missing data
# fill these with "None"
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
            'GarageCond', 'GarageType', 'BsmtCond', 'BsmtExposure', 'BsmtQual',
            'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'GarageQual',
            'GarageFinish'):
    combine_df[col] = combine_df[col].fillna('None')
# fill with 0
for col in ('GarageYrBlt', 'MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath',
            'GarageCars', 'TotalBsmtSF', 'GarageArea', 'BsmtFinSF2',
            'BsmtFinSF1', 'BsmtUnfSF'):
    combine_df[col] = combine_df[col].fillna(0)
# fill these with the most common value
for col in ('MSZoning', 'Electrical', 'Exterior2nd', 'KitchenQual',
            'Exterior1st', 'SaleType'):
    combine_df[col] = combine_df[col].fillna(combine_df[col].mode()[0])
# fill with median of neighbourhood
combine_df['LotFrontage'] = combine_df.groupby('Neighborhood')[
    'LotFrontage'].transform(lambda x: x.fillna(x.median()))
# new feature
combine_df['Functional'] = combine_df['Functional'].fillna('Typical')
# delete feature
combine_df = combine_df.drop(['Utilities'], axis=1)

combine_df = pd.get_dummies(combine_df)

# Fitting and solution
X_train = combine_df[:n_train]
Y_train = y_train
X_test = combine_df[n_train:]

# initialize base level models
models = [
    XGBRegressor(n_jobs=4),
    GradientBoostingRegressor(),
    KernelRidge()
]

# compute stacking features
S_train, S_test = stacking(
    models,
    X_train,
    y_train,
    X_test,
    regression=True,
    metric=mean_absolute_error,
    n_folds=4,
    shuffle=True,
    random_state=0,
    verbose=2)

# initialize 2nd level model
model = LinearRegression()

# fit 2nd level model
model = model.fit(S_train, y_train)

# predict
y_pred = np.expm1(model.predict(S_test))

# save result
submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": y_pred
})
submission.to_csv('house_submission.csv', index=False)

print("working")
