from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from Argumentssygp import *
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from collections import Counter
import pandas as pd
import csv
import os
class PrepareDataset:
    use_residual = False
    classification = True
    gsgp_standard = False
    ML = False
    ML_default = False
    one_pop = False
    Random_pop = False
    sygp = False
    gsgp = False
    Normalization = True

    def __init__(self,sygp, classification, ML, Normalization,cross=7):
        self.sygp = sygp
        self.classification = classification
        self.ML = ML
        self.Normalization = Normalization
        self.cross=cross

    def masking(self,y, k):
        y_prim = y.copy()
        for index in range(len(y)):
            if y_prim[index] == k:
                y_prim[index] = 1
            else:
                y_prim[index] = -1
        return y_prim

    def get_dataset(self,DATASETS_DIR,which,r):
        df_X = pd.read_csv(DATASETS_DIR + which + str(r) + 'train.csv')
        X_train = df_X.iloc[:, :-1]
        y_train = df_X['Y']

        df_Y = pd.read_csv(DATASETS_DIR + which + str(r) + 'test.csv')
        X_test = df_Y.iloc[:, :-1]
        y_test = df_Y['Y']

        df_Y = pd.read_csv(DATASETS_DIR + which + str(r) + 'val.csv')
        X_val = df_Y.iloc[:, :-1]
        y_val = df_Y['Y']
        if self.Normalization:
            # scaler = MinMaxScaler()
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_val_scaled = scaler.transform(X_val)
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
            X_val = pd.DataFrame(X_val_scaled, columns=X_test.columns)
        if self.use_residual:
            num_class = len(np.unique(y_train))
            if num_class == 2:
                y_train =self.masking(y_train, 1)
                y_test =self.masking(y_test, 1)
                y_val =self.masking(y_val, 1)
        return X_train, y_train, X_test, y_test, X_val, y_val

    def save_result(self, OUTPUT_DIR1, which, dim, test_end, train_end, result_eval, result_fitness, fitness_end, result_train,
                    result_test, method,base1, base2, MAX_NICHE_COUNT, cross_rate, niche_size, functions,
                    f1_train=None, f1_test=None, precision_train=None, precision_test=None,
                    recall_train=None, recall_test=None, auc_test=None,
                    runtime=None):
        """
        Save all results and metrics (including runtime) into CSV files.
        """

        # =============================
        # Determine output directory and filename
        # =============================
        if self.sygp :
            OUTPUT_DIR = os.path.join(OUTPUT_DIR1, str(which))
            outputfile = f'_{which}{MAX_Evaluation}'
            end_of_path = (f"{method}pop{POPULATION_SIZE}_type_{self.cross}{outputfile}"
                           f"{FITNESS_TYPE[0]}{VALIDATION}{base1}{base2}{MAX_NICHE_COUNT}"
                           f"rate{cross_rate}N_S{niche_size}{functions}")

        else:
            OUTPUT_DIR = OUTPUT_DIR1
            outputfile = f'_{which}'
            end_of_path = f"{method}{outputfile}"

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # =============================
        # Helper to save lists to CSV
        # =============================
        def _save_csv(filename, data):
            path = os.path.join(OUTPUT_DIR, f"{filename}{end_of_path}.csv")
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(data)

        # =============================
        # Save standard metrics
        # =============================
        _save_csv("test", test_end)
        _save_csv("train", train_end)
        _save_csv("dim", dim)
        _save_csv("result_eval", result_eval)
        _save_csv("result_fitness", result_fitness)
        _save_csv("fitness_end", fitness_end)
        _save_csv("result_train", result_train)
        _save_csv("result_test", result_test)

        # =============================
        # Save classification metrics if available
        # =============================
        if f1_train is not None:       _save_csv("f1_train", f1_train)
        if f1_test is not None:        _save_csv("f1_test", f1_test)
        if precision_train is not None: _save_csv("precision_train", precision_train)
        if precision_test is not None: _save_csv("precision_test", precision_test)
        if recall_train is not None:   _save_csv("recall_train", recall_train)
        if recall_test is not None:    _save_csv("recall_test", recall_test)
        if auc_test is not None:       _save_csv("auc_test", auc_test)

        # =============================
        # Save runtime if available
        # =============================
        if runtime is not None:
            if not isinstance(runtime, (list, tuple, np.ndarray)):
                runtime = [runtime]  # wrap float into a list
            _save_csv("runtime", runtime)
        print(f"Results saved to {OUTPUT_DIR}")

