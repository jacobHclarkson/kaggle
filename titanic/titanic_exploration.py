# titanic exploration

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
print()

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(train_df.info())

# pivoting categorical features
# print(train_df[['Pclass', 'Survived']].groupby(
#     ['Pclass'], as_index=False).mean().sort_values(
#         by='Survived', ascending=False))
# print(train_df[['Sex', 'Survived']].groupby(
#     ['Sex'], as_index=False).mean().sort_values(
#         by='Survived', ascending=False))
# print(train_df[['Embarked', 'Survived']].groupby(
#     ['Embarked'], as_index=False).mean().sort_values(
#         by='Survived', ascending=False))
# print(train_df[['SibSp', 'Survived']].groupby(
#     ['SibSp'], as_index=False).mean().sort_values(
#         by='Survived', ascending=False))
# print(train_df[['Parch', 'Survived']].groupby(
#     ['Parch'], as_index=False).mean().sort_values(
#         by='Survived', ascending=False))

# plot age vs average survival rate
# g = sns.FacetGrid(train_df, col='Survived')
# g.map(plt.hist, 'Age', bins=20)
# plt.show()

# plot SibSp vs average survival rate
# sns.barplot(x="SibSp", y="Survived", data=train_df)
# plt.show()

# plot Parch vs average survival rate
# sns.barplot(x="Parch", y="Survived", data=train_df)
# plt.show()

# plot FamSize vs average survival rate
# train_df['FamSize'] = train_df['Parch'] + train_df['SibSp']
# sns.barplot(x="FamSize", y="Survived", data=train_df)
# plt.show()

# make new features
# train_df['Title'] = train_df.Name.str.extract('([A-Za-z]+)\.', expand=False)
# train_df['Title'] = train_df['Title'].map({
#     'Sir': 'Nobleman',
#     'Countess': 'Noblewoman',
#     'Don': 'Nobleman',
#     'Lady': 'Noblewoman',
#     'Ms': 'Ms',
#     'Mme': 'Ms',
#     'Mlle': 'Ms',
#     'Mrs': 'Ms',
#     'Master': 'Mr',
#     'Col': 'Service',
#     'Major': 'Service',
#     'Dr': 'Service',
#     'Jonkheer': 'Mr',
#     'Rev': 'Service',
#     'Capt': 'Service'
# })
# sns.barplot(x="Title", y="Survived", data=train_df)
# plt.show()

train_df['Cabin'] = train_df['Cabin'].fillna("Unknown")
train_df['Cabin'] = train_df.Cabin.str.extract('([ABCDEFGU])', expand = False)
print(train_df['Cabin'])
sns.barplot(x="Cabin", y="Survived", data=train_df)
plt.show()
