import time
import numpy as np
import pandas as pd

from sygp.MethodsConfig import Methodsconfig
from sygp.SYGPConfig import SYGPConfig
from sygp.PrepareDataset import PrepareDataset
from sygp.SYGPMethods import SYGPMethods
from Argumentssygp import *
import os

########################################################
DATASETS_DIR = 'datasetbreak2/'
cross_rate=0.75
cross_over_type = 0
method = ['sygp_classification']
functions = "simple"
methods_config = Methodsconfig(method[0])

my_list = methods_config.get_parameter()
sygp = my_list[0]
classification = my_list[1]
ML = my_list[2]
tor1 = my_list[3]
cross = my_list[4]
k_split_trainingset = my_list[5]
Normalization = my_list[6]
datasetName = my_list[7]
methodNames = my_list[8]
pop_size = my_list[9]
OUTPUT_DIR = my_list[10]
FITNESS_TYPE = my_list[11]


########################################################
sygp_config = SYGPConfig(sygp)
PrepDataset = PrepareDataset(sygp, classification, ML, cross)
base1 = 1
base2 = 1
niche_size = 300

sygpmethods = SYGPMethods(sygp, FITNESS_TYPE, classification, ML, tor1, cross, k_split_trainingset, base1, base2)

for which in datasetName:
    print('dataset:' + which)

    for method in methodNames:
        result_test = []
        result_train = []
        result_fitness = []
        fitness_end = []
        train_end = []
        f1_train = []
        f1_test = []
        precision_train = []
        precision_test = []
        recall_train = []
        recall_test = []
        auc_test = []
        test_end = []
        dim = []
        result_eval = []
        accuracy = []
        runtime_list = []
        modelName = sygp_config.get_model_name(method)
        sygpmethods.set_method_name(modelName)
        start = 0
        end = 29
        RUNS = range(start, end + 1)
        model_name = ""

        for r in RUNS:
            run_start = time.time()  # start timer

            X_train, y_train, X_test, y_test, X_val, y_val = PrepDataset.get_dataset(DATASETS_DIR, which, r)
            X_train = pd.concat([X_train, X_val], axis=0)
            y_train = pd.concat([y_train, y_val], axis=0)
            X_val, y_val = [], []
            sygpmethods.set_data_set(X_train, y_train, X_test, y_test, X_val, y_val)
            Num_class = len(np.unique((y_train)))


            (train_end, test_end, fitness_end, result_train, result_test, result_fitness, dim,
             result_eval, method_name, cross_over_type,
             f1_train, f1_test, precision_train, precision_test,
             recall_train, recall_test, auc_test) = sygpmethods.call_sygp(
                pop_size, which, r, train_end, test_end, fitness_end, result_train, result_test,
                result_fitness, dim, result_eval, MAX_NICHE_COUNT,
                cross_rate, niche_size, f1_train, f1_test, precision_train,
                precision_test, recall_train, recall_test, auc_test
            )

            # compute runtime
            run_end = time.time()
            runtime = run_end - run_start
            print(f"Run {r} finished in {runtime:.2f} seconds")
            runtime_list.append(runtime)
            # save results immediately after each run
            PrepDataset.save_result(
                OUTPUT_DIR, which, dim, test_end, train_end, result_eval, result_fitness, fitness_end, result_train,
                result_test, modelName,
                base1, base2, MAX_NICHE_COUNT, cross_rate, niche_size,
                functions, f1_train, f1_test, precision_train,
                precision_test, recall_train, recall_test, auc_test,
                runtime=runtime_list  # <-- NEW
            )
