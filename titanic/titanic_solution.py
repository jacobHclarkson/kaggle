# titanic solution

import pandas as pd
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
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
Y_train = y_train
X_test = combine_df[n_train:]

# xgboost
xgboost_classifier = XGBClassifier(n_estimators=1000, n_jobs=4)

xgboost_classifier.fit(X_train, Y_train)
xgboost_pred = xgboost_classifier.predict(X_test)
acc_xgboost_classifier = round(
    xgboost_classifier.score(X_train, Y_train) * 100, 2)
print(acc_xgboost_classifier)
plot_importance(xgboost_classifier, importance_type='gain')
plt.show()

# save result
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": xgboost_pred
})

submission.to_csv('titanic_submission.csv', index=False)
