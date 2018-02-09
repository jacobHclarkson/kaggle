# titanic solution

import pandas as pd
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from vecstack import stacking
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
print()

# read data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# combine data sets
n_train = train_df.shape[0]
n_test = test_df.shape[0]
y_train = train_df.Survived.values
combine_df = pd.concat((train_df, test_df)).reset_index(drop=True)
combine_df.drop(['Survived'], axis=1, inplace=True)

# make new features
combine_df['Title'] = combine_df.Name.str.extract(
    '([A-Za-z]+)\.', expand=False)
combine_df['Title'] = combine_df['Title'].map({
    'Sir': 'Noble',
    'Countess': 'Noble',
    'Don': 'Noble',
    'Lady': 'Noble',
    'Ms': 'Pleb',
    'Mme': 'Pleb',
    'Mlle': 'Pleb',
    'Mrs': 'Pleb',
    'Miss': 'Pleb',
    'Master': 'Noble',
    'Col': 'Service',
    'Major': 'Service',
    'Dr': 'Service',
    'Mr': 'Pleb',
    'Jonkheer': 'Pleb',
    'Rev': 'Service',
    'Capt': 'Service'
})
combine_df['Cabin'] = train_df['Cabin'].fillna("Unknown")
combine_df['Cabin'] = train_df.Cabin.str.extract('([ABCDEFGU])', expand=False)

combine_df['FamSize'] = combine_df['SibSp'] + combine_df['Parch']
combine_df['IsAlone'] = 0

combine_df.loc[combine_df['FamSize'] == 0, 'IsAlone'] = 1
combine_df.loc[combine_df['Age'] <= 8, 'Age'] = 0
combine_df.loc[(combine_df['Age'] > 8) & (combine_df['Age'] <= 64), 'Age'] = 1
combine_df.loc[(combine_df['Age'] > 64), 'Age'] = 2

combine_df.loc[combine_df['Fare'] <= 7.91, 'Fare'] = 0
combine_df.loc[(combine_df['Fare'] > 7.91) & (combine_df['Fare'] <= 14.454),
               'Fare'] = 1
combine_df.loc[(combine_df['Fare'] > 14.454) & (combine_df['Fare'] <= 31),
               'Fare'] = 2
combine_df.loc[combine_df['Fare'] > 31, 'Fare'] = 3

# supply missing values
combine_df['Age'] = combine_df['Age'].fillna(combine_df['Age'].median())
combine_df['Embarked'] = combine_df['Embarked'].fillna(
    combine_df['Embarked'].mode()[0])
combine_df['Fare'] = combine_df['Fare'].fillna(combine_df['Fare'].median())

# convert features
combine_df['Pclass'] = combine_df['Pclass'].astype(str)
combine_df['Age'] = combine_df['Age'].astype(str)

# drop features that we aren't using
combine_df.drop(['PassengerId'], axis=1, inplace=True)
combine_df.drop(['Ticket'], axis=1, inplace=True)
combine_df.drop(['Name'], axis=1, inplace=True)

# create dummies
combine_df = pd.get_dummies(combine_df)

# prep data for learning
X_train = combine_df[:n_train]
X_test = combine_df[n_train:]

# STACKING
#---------
# initialize base level models
models = [
    LogisticRegression(),
    KNeighborsClassifier(),
    RandomForestClassifier()
]

# compute stacking features
S_train, S_test = stacking(
    models,
    X_train,
    y_train,
    X_test,
    regression=False,
    metric=accuracy_score,
    n_folds=4,
    stratified=True,
    shuffle=True,
    random_state=0,
    verbose=2)

# initialize 2nd level model
model = XGBClassifier()

# fit 2nd level model
model = model.fit(S_train, y_train)

# predict
y_pred = model.predict(S_test)

# save result
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": y_pred
})

submission.to_csv('titanic_submission.csv', index=False)

print("working")
