# Bibliotecas para modelagem:
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedKFold

class classification_models(object):

    def __init__(self, X, Y, models):

        self.X = X
        self.Y = Y
        self.models = models
        self.X_train = None # k-fold result
        self.Y_train = None # k-fold result
        self.X_valid = None # k-fold result
        self.Y_valid = None # k-fold result
        self.results = {} # Model results

    def create_struct(self):
        for name, model in self.models:
            self.results[name] = {}
            self.results[name]['acc'] = []
            self.results[name]['precision'] = []
            self.results[name]['recall'] = []
            self.results[name]['f1_score'] = []
            self.results[name]['auc'] = []
        
    def k_fold(self):
        kf = RepeatedKFold(n_splits=5, n_repeats=30, random_state=20)
        for linhas_treino, linhas_valid in kf.split(self.X):
            self.X_train, self.X_valid = self.X[linhas_treino], self.X[linhas_valid]
            self.Y_train, self.Y_valid = self.Y.iloc[linhas_treino], self.Y.iloc[linhas_valid]
            self.apply_model()
            
    def hist_validation(self):
        pass

    def apply_model(self):
    
        self.create_struct()
        for name, model in self.models:
            create_model = model.fit(self.X_train, self.Y_train)
            predictions = create_model.predict(self.Y_valid)
            self.results[name]['acc'].append(accuracy_score(self.Y_valid , predictions))
            self.results[name]['precision'].append(precision_score(self.Y_valid , predictions))
            self.results[name]['recall'].append(recall_score(self.Y_valid , predictions))
            self.results[name]['f1_score'].append(f1_score(self.Y_valid , predictions))
            self.results[name]['auc'].append(roc_auc_score(self.Y_valid , predictions))
            
        return self.results