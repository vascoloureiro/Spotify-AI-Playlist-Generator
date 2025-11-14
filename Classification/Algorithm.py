from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

class AlgorithmModels:
    def __init__(self, XTrain, XTest, YTrain, randomState=42):
        self.XTrain = XTrain
        self.XTest = XTest
        self.YTrain = YTrain
        self.randomState = randomState

    def DecisonTree(self):
        model = DecisionTreeClassifier(random_state=self.randomState)
        model.fit(self.XTrain, self.YTrain)
        return model.predict(self.XTest), model
    
    def RandomForest(self):
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=self.randomState,
            class_weight='balanced',
            n_jobs=-1
        )
        model.fit(self.XTrain, self.YTrain)
        return model.predict(self.XTest), model

    def RandomForest_tuned(self, X_test, y_test):
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "class_weight": ["balanced"]
        }

        grid = GridSearchCV(
            RandomForestClassifier(random_state=self.randomState, n_jobs=-1),
            param_grid,
            cv=3,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1
        )

        grid.fit(self.XTrain, self.YTrain)
        best = grid.best_estimator_
        preds = best.predict(X_test)

        return preds, best, grid.best_params_
