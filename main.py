import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd


def main():
    lr = LogisticRegression()
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    train_data = preprocess(train, train, True)
    test_data = preprocess(test, train, False)

    mat = train_data.values
    xmat = mat[:,[1,2,3,4,5,6]]
    ymat = mat[:,0]

    tmat = test_data.values
    tmat = tmat[:,[1,2,3,4,5,6]]

    lr.fit(xmat, ymat)
    predictions = lr.predict(tmat)
    predictions = predictions.astype(int)
    submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})
    filename = 'Titanic LR Prediction.csv'
    submission.to_csv(filename, index=False)


def preprocess(data, tdata, train):

    if (train == True):
        data = data.drop(["PassengerId", "Name", "Cabin", "SibSp", "Parch", "Ticket"],
                         axis=1)
    else:
        data = data.drop([ "Name", "Cabin", "SibSp", "Parch", "Ticket"],
                          axis=1)

    #Replace Nan Embarked with most embarked location
    mostEmbarked = data["Embarked"].value_counts().idxmax()
    data["Embarked"].fillna(mostEmbarked, inplace=True)

    #Replace Nan Age with Median
    medianAge = tdata["Age"].median(skipna=True)
    data["Age"].fillna(medianAge, inplace=True)

    #Create dummy variables
    data = pd.get_dummies(data, columns=["Pclass", "Embarked"])
    data = data.drop(["Pclass_3", "Embarked_S"], axis=1)

    #Fill any remaining Nan with 0
    data = data.fillna(0)

    #change
    mapper = {'male': 1, 'female': 0}
    data.Sex = [mapper[item] for item in data.Sex]

    return data

if __name__ == '__main__':
    main()