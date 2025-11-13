import pandas as pd  # já tens em cima, mas fica aqui se correr separado

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

class AlgorithmModels:
    def __init__(self, XTrain, XTest, YTrain, randomState=42, maxDepth=10, nEstimators=200, classWeight='balanced', nJobs=-1):    
        self.XTrain = XTrain
        self.XTest = XTest
        self.YTrain = YTrain
        self.randomState = randomState
        self.maxDepth = maxDepth
        self.nEstimators = nEstimators
        self.classWeight = classWeight
        self.nJobs = nJobs
                
    def DecisonTree(self):
        model = DecisionTreeClassifier(self.randomState, self.class_weight, self.maxDepth)
        model.fit(self.XTrain, self.YTrain)
        return model.predict(self.XTest)
    
    def RandomForest(self):
        model = RandomForestClassifier(
            n_estimators=self.nEstimators,
            random_state=self.randomState,
            class_weight=self.classWeight,
            n_jobs=self.nJobs
        )
        model.fit(self.XTrain, self.YTrain)
        return model.predict(self.XTest)
    