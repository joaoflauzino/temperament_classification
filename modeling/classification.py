# Bibliotecas para modelagem:
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedKFold
import seaborn as sns
import matplotlib.pyplot as plt

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
            for medidas in ['acc', 'precision', 'recall', 'f1_score', 'auc']:
                self.results[name][medidas] = []
        
    def k_fold(self):
        self.create_struct()
        kf = RepeatedKFold(n_splits=5, n_repeats=30, random_state=20)
        for linhas_treino, linhas_valid in kf.split(self.X):
            self.X_train, self.X_valid = self.X[linhas_treino], self.X[linhas_valid]
            self.Y_train, self.Y_valid = self.Y.iloc[linhas_treino], self.Y.iloc[linhas_valid]
            self.apply_model()
            
    def hist(self, metrics):
        
        modelos = [x[0] for x in self.models]
        linhas = 2
        colunas = 3
        fig, axs = plt.subplots(linhas, colunas, figsize=(15, 15))
        ind = []
        for i,v in enumerate(axs):
            for j,k in enumerate(v):
                ind.append([i,j])
        
        for i in range(len(modelos)):
            sns.histplot(data=self.results[modelos[i]], x=metrics, kde=True, color="skyblue", ax=axs[ind[i][0], ind[i][1]]).set_title(f'{modelos[i]}_{metrics}')
        #plt.show()
        plt.savefig(f'results/{metrics}.png', dpi=300)
                
    def reports(self):
        for i in ['acc', 'precision', 'recall', 'f1_score', 'auc']:
            self.hist(i)
            
    def apply_model(self):
    
        for name, model in self.models:
            model_ = model.fit(self.X_train, self.Y_train)
            predictions = model_.predict(self.X_valid)
            self.results[name]['acc'].append(accuracy_score(self.Y_valid , predictions))
            self.results[name]['precision'].append(precision_score(self.Y_valid , predictions))
            self.results[name]['recall'].append(recall_score(self.Y_valid , predictions))
            self.results[name]['f1_score'].append(f1_score(self.Y_valid , predictions))
            self.results[name]['auc'].append(roc_auc_score(self.Y_valid , predictions))
            
        return self.results