import pandas as pd  # já tens em cima, mas fica aqui se correr separado

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Components
from Algorithm import AlgorithmModels

class Classificator:
    
    def __init__(self, file: pd.DataFrame, X:float, Y: float, testSize: float = 0.2, randomState: int = 42, stratify: str = 'y'):
        
        self.file = file
        self.X = X
        self.Y = Y
        self.testSize = testSize
        self.randomState = randomState
        self.stratify = stratify
        
    def train(self):
        return train_test_split(
            self.X, self.y, self.testSize, self.randomState, self.stratify
        )
    
    def reportDecisionTree(self):
        X_train, X_test, y_train, y_test = self.train()
        decisionTree = AlgorithmModels(X_train, X_test, y_train)
        y_pred_dt = decisionTree.DecisonTree()
        return [classification_report(y_test, y_pred_dt), accuracy_score(y_test, y_pred_dt)]
    
    def reportRandomForest(self):
        X_train, X_test, y_train, y_test = self.train()
        decisionTree = AlgorithmModels(X_train, X_test, y_train)
        y_pred_dt = decisionTree.RandomForest()
        return [classification_report(y_test, y_pred_dt), accuracy_score(y_test, y_pred_dt)]
    