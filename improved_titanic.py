import pandas
titanic =pandas.read_csv("C:/Users/Sravan Apuri/Desktop/titanic/train.csv")
#print(titanic.head(5))
#print(titanic.describe())
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
#print(titanic.describe())

#print(titanic["Sex"].unique())
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
#print(titanic.head(5))

titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
#print(titanic.head(5))

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
print(scores.mean())

titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))

import re
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

titles = titanic["Name"].apply(get_title)
print(pandas.value_counts(titles))

title_mapping = {"Mr": 1, "Miss": 2, "Mrs":3, "Master":4, "Dr":5, "Rev": 6, "Major": 7, "Col": 7, 
    "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}

for k,v in title_mapping.items():

    titles[titles == k] = v


# Verify that we converted everything.

print(pandas.value_counts(titles))


# Add in the title column.

titanic["Title"] = titles

