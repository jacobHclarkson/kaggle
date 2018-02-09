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
train_df['FamSize'] = train_df['Parch'] + train_df['SibSp']
# sns.barplot(x="FamSize", y="Survived", data=train_df)
# plt.show()

# make new features
# train_df['Title'] = train_df.Name.str.extract('([A-Za-z]+)\.', expand=False)
# train_df['Title'] = train_df['Title'].map({
#     'Sir': 'Noble',
#     'Countess': 'Noble',
#     'Don': 'Noble',
#     'Lady': 'Noble',
#     'Ms': 'Pleb',
#     'Mme': 'Pleb',
#     'Mlle': 'Pleb',
#     'Mrs': 'Pleb',
#     'Miss': 'Pleb',
#     'Master': 'Noble',
#     'Col': 'Service',
#     'Major': 'Service',
#     'Dr': 'Service',
#     'Mr': 'Pleb',
#     'Jonkheer': 'Pleb',
#     'Rev': 'Service',
#     'Capt': 'Service'
# })
# sns.barplot(x="Title", y="Survived", data=train_df)
# plt.show()

# investigating cabin feature
# train_df['Cabin'] = train_df['Cabin'].fillna("Unknown")
# train_df['Cabin'] = train_df.Cabin.str.extract('([ABCDEFGU])', expand = False)
# print(train_df['Cabin'])
# sns.barplot(x="Cabin", y="Survived", data=train_df)
# plt.show()

# test is alone feature
# train_df['IsAlone'] = 0
# train_df.loc[train_df['FamSize']==0, 'IsAlone'] = 1
# sns.barplot(x="IsAlone", y="Survived", data=train_df)
# plt.show()

# test age bands
# train_df.loc[train_df['Age']<=8,'Age']=0
# train_df.loc[(train_df['Age']>8) & (train_df['Age']<=64),'Age']=1
# train_df.loc[(train_df['Age']>64),'Age']=2
# sns.barplot(x="Age", y="Survived", data=train_df)
# plt.show()

# test fare bands
# train_df['FareBand']=pd.qcut(train_df['Fare'], 4)
# print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand',ascending=True))
# sns.barplot(x="FareBand", y="Survived", data=train_df)
# plt.show()
