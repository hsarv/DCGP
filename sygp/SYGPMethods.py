from .SimpleSum import SimpleSum
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression

from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingClassifier, HistGradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sygp.SYGP import SYGP
from sygp.PrepareDataset import PrepareDataset
from sygp.MahalanobisDistanceClassifier import MahalanobisDistanceClassifier
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn import tree, neighbors
import pandas as pd
import numpy as np
from Argumentssygp import *


class SYGPMethods:
    use_residual = False
    classification = True
    gsgp_standard = False
    ML = False
    ML_default = False
    one_pop = False
    Random_pop = False
    sygp_flag = False
    gsgp_flag = False
    modelName = ""
    methodName = ""

    def root_mean_squared_error(self,y_true, y_pred):
        from sklearn.metrics import mean_squared_error
        import numpy as np
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def __init__(self, sygp_flag,FITNESS_TYPE,classification, ML, tor1,cross, ksplitTrainingset,base1,base2):
        self.sygp_flag = sygp_flag
        self.classification = classification
        self.ML = ML
        self.stgp_cross = cross
        self.tor1 = tor1
        self.FITNESS_TYPE = FITNESS_TYPE
        self.sygp_k_split_trainingset = ksplitTrainingset

        self.cross=cross
        self.base1=base1
        self.base2=base2


    def _calculate_classification_metrics(self, y_true, y_pred, clf=None, X_test=None, y_proba1=None):
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred, average="macro")
        metrics['precision'] = precision_score(y_true, y_pred, average="macro")
        metrics['recall'] = recall_score(y_true, y_pred, average="macro")

        try:
            # Determine probabilities for AUC
            if clf is not None and X_test is not None:
                if hasattr(clf, "predict_proba"):
                    y_proba = clf.predict_proba(X_test)
                    if len(np.unique(y_true)) == 2:  # Binary classification
                        if y_proba.shape[1] > 1:
                            y_proba_binary = y_proba[:, 1]
                        else:
                            y_proba_binary = y_proba.ravel()
                        metrics['auc'] = roc_auc_score(y_true, y_proba_binary)
                    else:  # Multiclass classification
                        metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class="ovr")
                else:
                    metrics['auc'] = np.nan

            elif y_proba1 is not None:
                metrics['auc'] = roc_auc_score(y_true, y_proba1)
            else:
                metrics['auc'] = np.nan
        except Exception as e:
            print("AUC calculation failed:", e)
            metrics['auc'] = np.nan

        return metrics

    def set_data_set(self, X_train, y_train, X_test, y_test, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val

    def set_method_name(self, model_name):
        self.modelName = model_name

    def call_sygp(self, pop_size, which, r, train_end, test_end, fitness_end, result_train,
                  result_test, result_fitness, dim, result_eval, MAX_NICHE_COUNT2,
                  cross_rate, niche_size,
                  f1_train=None, f1_test=None,
                  precision_train=None, precision_test=None,
                  recall_train=None, recall_test=None,
                  auc_test=None):

        sygp = SYGP( OPERATORS, MAX_DEPTH,
                    pop_size, MAX_GENERATION, MAX_Evaluation, 5, ELITISM_SIZE, LIMIT_DEPTH,
                    DIM_MIN, DIM_MAX, THREADS, r, UPDATEALL, self.cross, VALIDATION, KFOLD, VERBOSE,
                    self.modelName, self.FITNESS_TYPE[0], which,
                    niche_size, MAX_NICHE_COUNT2, self.methodName,[0], self.classification,
                    self.sygp_k_split_trainingset, self.use_residual, self.base1, self.base2, cross_rate)

        sygp.fit(self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val)
        result_train += [sygp.trainingAccuracyOverTime]
        result_test += [sygp.testAccuracyOverTime]
        result_fitness += [sygp.fitnessOverTime]

        fitness_end += [sygp.fitnessOverTime[-1]]
        train_end += [sygp.msfvector.getTrainingMeasure(self.X_train, self.y_train)]
        test_end += [sygp.msfvector.getTestMeasure(self.X_test, self.y_test)]
        dim += [sygp.msfvector.getNumberOfDimensions()]

        if self.classification:
            metrics = self._calculate_classification_metrics(self.y_test, sygp.msfvector.predict(self.X_test),
                                                             sygp.msfvector, self.X_test,sygp.msfvector.predict_proba(self.X_test))
            if f1_test is not None: f1_test += [metrics['f1']]
            if precision_test is not None: precision_test += [metrics['precision']]
            if recall_test is not None: recall_test += [metrics['recall']]
            if auc_test is not None: auc_test += [metrics['auc']]

        result_eval += []
        return train_end, test_end, fitness_end, result_train, result_test, result_fitness, dim, result_eval, \
               self.methodName, self.cross, f1_train, f1_test, precision_train, precision_test, recall_train, recall_test, auc_test

    def call_ML_Method(self, method, train_end, test_end,
                       f1_train=None, f1_test=None,
                       precision_train=None, precision_test=None,
                       recall_train=None, recall_test=None,
                       auc_test=None):

        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test

        # Select the appropriate model
        clf = None

        if method == "SimpleSum":
            clf = SimpleSum()
        elif method == "Lir":
            clf = LinearRegression()
        elif method == 'SVR':
            clf = SVR()
        elif method == 'KNeighborsRegressor':
            clf = neighbors.KNeighborsRegressor()
        elif method == 'MLPRegressor':
            clf = MLPRegressor(random_state=1, max_iter=500)
        elif method == 'DecisionTreeRegressor':
            clf = tree.DecisionTreeRegressor()
        elif method == 'RandomForestRegressor':
            clf = RandomForestRegressor(n_estimators=50, random_state=0, oob_score=True)
        elif method == 'ExtraTreesRegressor':
            clf = ExtraTreesRegressor(n_estimators=200, n_jobs=-1)
        elif method == 'AdaBoostRegressor':
            clf = AdaBoostRegressor(n_estimators=200)
        elif method == 'GBDT':
            clf = GradientBoostingRegressor(n_estimators=200)
        elif method == 'DART':
            clf = LGBMRegressor(n_jobs=1, n_estimators=200, boosting_type='dart', xgboost_dart_mode=True)
        elif method == 'XGBoost':
            clf = XGBRegressor(n_jobs=1, n_estimators=200)
        elif method == 'LightGBM':
            clf = LGBMRegressor(n_jobs=1, n_estimators=200)
        elif method == 'CatBoost':
            clf = CatBoostRegressor(n_estimators=200, thread_count=1, verbose=False, allow_writing_files=False)
        elif method == 'Ridge':
            clf = Ridge()
        # Add classification models
        elif method == 'KNN':
            clf = KNeighborsClassifier()
        elif method == 'LR':
            clf = LogisticRegression()
        elif method == '1NN':
            clf = KNeighborsClassifier(n_neighbors=1)
        elif method == 'DT':
            clf = tree.DecisionTreeClassifier(random_state=42, max_depth=6)
        elif method == 'RF':
            clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        elif method == 'MLP':
            clf = MLPClassifier(alpha=1, max_iter=1000)
        elif method == 'NC':
            clf = MahalanobisDistanceClassifier()
        elif method == 'SVM':
            clf = SVC(kernel="linear", C=1, max_iter=500)
        elif method == 'RS':
            clf = BaggingClassifier(bootstrap=False, max_features=0.5)
        elif method == 'LDA':
            clf = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
        elif method == 'XGB':
            clf = HistGradientBoostingClassifier(min_samples_leaf=10, max_iter=50)

        # Train model
        clf.fit(X_train, y_train)

        # Collect metrics
        if self.classification:
            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)

            train_end += [accuracy_score(y_train, y_pred_train)]
            test_end += [accuracy_score(y_test, y_pred_test)]

            metrics = self._calculate_classification_metrics(y_test, y_pred_test, clf, X_test)
            if f1_train is not None: f1_train += [metrics.get('f1_train', None)]
            if f1_test is not None: f1_test += [metrics.get('f1', None)]
            if precision_train is not None: precision_train += [metrics.get('precision_train', None)]
            if precision_test is not None: precision_test += [metrics.get('precision', None)]
            if recall_train is not None: recall_train += [metrics.get('recall_train', None)]
            if recall_test is not None: recall_test += [metrics.get('recall', None)]
            if auc_test is not None: auc_test += [metrics.get('auc', None)]
        else:
            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)

            train_end += [np.sqrt(mean_squared_error(y_train, y_pred_train))]
            test_end += [np.sqrt(mean_squared_error(y_test, y_pred_test))]

        return train_end, test_end, f1_train, f1_test, precision_train, precision_test, recall_train, recall_test, auc_test

