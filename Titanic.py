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

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

alg = LinearRegression()

Kf = KFold(titanic.shape[0], n_folds = 3, random_state = 1)

predictions = []

for train, test in Kf:
    train_predictors = (titanic[predictors].iloc[train,:])
    train_target = titanic["Survived"].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)

#print(predictions)

import numpy as np
predictions = np.concatenate(predictions, axis = 0)

predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
#print(accuracy)

from sklearn.linear_model import LogisticRegression
from sklearn import  cross_validation

alg = LogisticRegression(random_state = 1)

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
#print(scores.mean())


titanic_test = pandas.read_csv("C:/Users/Sravan Apuri/Desktop/titanic/test.csv")

titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())
titanic_test.loc[titanic_test["Sex"] == "male","Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female","Sex"] = 1
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
titanic_test.loc[titanic_test["Embarked"] == "S","Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C","Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q","Embarked"] = 2
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
alg = LogisticRegression(random_state = 1)

alg.fit(titanic[predictors], titanic["Survived"])

predictions = alg.predict(titanic_test[predictors])

submission = pandas.DataFrame({"PassengerId": titanic_test["PassengerId"], "Survived": predictions})
submission.to_csv("C:/Users/Sravan Apuri/Desktop/titanic/kaggle.csv",index=False)

