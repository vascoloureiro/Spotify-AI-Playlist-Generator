import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from Classification.Algorithm import AlgorithmModels

class Classificator:
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, testSize: float = 0.2, randomState: int = 42, stratify: str = 'y'):
        self.X = X
        self.y = y                  
        self.testSize = testSize
        self.randomState = randomState
        self.stratify = stratify
        
    def train(self):
        return train_test_split(
            self.X,
            self.y,
            test_size=self.testSize,
            random_state=self.randomState,
            stratify=self.y if self.stratify == 'y' else None
        )
    
    def reportDecisionTree(self):
        X_train, X_test, y_train, y_test = self.train()
        dt_model = AlgorithmModels(X_train, X_test, y_train)
        y_pred, _ = dt_model.DecisonTree()
        return classification_report(y_test, y_pred), accuracy_score(y_test, y_pred)
    
    def reportRandomForest(self):
        X_train, X_test, y_train, y_test = self.train()
        rf_model = AlgorithmModels(X_train, X_test, y_train)
        y_pred, _ = rf_model.RandomForest()
        return classification_report(y_test, y_pred), accuracy_score(y_test, y_pred)
