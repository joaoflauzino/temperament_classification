# Bibliotecas para modelagem:
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.utils import np_utils
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import accuracy_score


class classification_models(object):

    def __init__(self, X_train, Y_train, X_valid, Y_valid, models):

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        self.models = models
        self.results = {}

    def apply_model(self):

        # for name, model in self.models:
        #     create_model = model.fit(self.X_train, self.Y_train)
        #     predictions = create_model.predict(self.X_valid)
        #     self.results[name] = accuracy_score(self.Y_valid , predictions)
        
        return self.models