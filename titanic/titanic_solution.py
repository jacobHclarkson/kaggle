# titanic solution

import pandas as pd
from xgboost import XGBClassifier
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

# supply missing values
combine_df['Age'] = combine_df['Age'].fillna(combine_df['Age'].median())
combine_df['Embarked'] = combine_df['Embarked'].fillna(
    combine_df['Embarked'].mode()[0])
combine_df['Fare'] = combine_df['Fare'].fillna(combine_df['Fare'].median())

# convert features
combine_df['SibSp'] = combine_df['SibSp'].astype(str)
combine_df['Parch'] = combine_df['Parch'].astype(str)
combine_df['Pclass'] = combine_df['Pclass'].astype(str)

# drop features that we aren't using
combine_df.drop(['PassengerId'], axis=1, inplace=True)
combine_df.drop(['Cabin'], axis=1, inplace=True)
combine_df.drop(['Ticket'], axis=1, inplace=True)
combine_df.drop(['Name'], axis=1, inplace=True)

# create dummies
combine_df = pd.get_dummies(combine_df)

# train a model
X_train = combine_df[:n_train]
Y_train = y_train
X_test = combine_df[n_train:]
xgboost_regressor = XGBClassifier()
xgboost_regressor.fit(X_train, Y_train)
Y_pred = xgboost_regressor.predict(X_test)

# save result
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": Y_pred
})

submission.to_csv('titanic_submission.csv', index=False)
