# titanic solution

# wrangling
import pandas as pd

# visualization
import matplotlib.pyplot as plt

# ml
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from vecstack import stacking
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
combine_df['Age'] = combine_df['Age'].astype(str)
combine_df['Sex'] = combine_df['Sex'].map({'female': 0, 'male': 1})


# create dummies
def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name], prefix=column_name)
    df = pd.concat([df, dummies], axis=1)
    return df

for column in ["Cabin", "Title", "Embarked"]:
    combine_df = create_dummies(combine_df, column)

# drop features that we aren't using
for col in ['PassengerId', 'Ticket', 'Name', 'Cabin', 'Title', 'Embarked']:
    combine_df.drop([col], axis=1, inplace=True)

# prep data for learning
X_train = combine_df[:n_train]
X_test = combine_df[n_train:]

# STACKING
#---------
# initialize base level models
models = [
    XGBClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    ExtraTreesClassifier()
]

# compute stacking features
S_train, S_test = stacking(
    models,
    X_train,
    y_train,
    X_test,
    regression=False,
    metric=accuracy_score,
    n_folds=5,
    stratified=True,
    shuffle=True,
    random_state=0,
    verbose=2)

# initialize 2nd level model
model = LogisticRegression()

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
