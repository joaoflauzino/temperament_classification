# Bibliotecas para modelagem:
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

class classification_models(object):

    def __init__(self, X_train, Y_train, X_valid, Y_valid, models):

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        self.models = models
        self.results = {}
        
    def k_fold(self):
        pass
    
    def hist_validation(self):
        pass

    def apply_model(self):

        for name, model in self.models:
            create_model = model.fit(self.X_train, self.Y_train)
            predictions = create_model.predict(self.X_valid)
            
            self.results[name] = {
                                    "acc": accuracy_score(self.Y_valid , predictions),
                                    "Precisão": precision_score(self.Y_valid , predictions),
                                    "Recall": recall_score(self.Y_valid , predictions),
                                    "F1_score": f1_score(self.Y_valid , predictions, average='macro'),
                                    "roc_auc_score": roc_auc_score(self.Y_valid , predictions, multi_class='ovr')
                                  }
        
        return self.results