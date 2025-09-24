
class SYGPConfig:

    def __init__(self, sygp):
        self.sygp = sygp


    def get_model_name(self,method):
        modelName = ''
        if method == 'KNN':
            modelName = "Nearest Neighbors"
        elif method == 'SimpleSum':
            modelName = "SimpleSum"
        elif method == 'LR':
            modelName = "LogisticRegression"
        elif method == '1NN':
            modelName = "1NN"
        elif method == 'DT':
            modelName = "DecisionTree"
        elif method == 'RF':
            modelName = "RandomForestClassifier"
        elif method == 'MLP':
            modelName = "Neural Net"
        elif method == 'NC':
            modelName = "MahalanobisDistanceClassifier"
        elif method == 'SVM':
            modelName = "Linear SVM"
        elif method == 'RS':
            modelName = "RS"
        elif method == 'LDA':
            modelName = "LDA"
        elif method == 'XGB':
            modelName = "xgb"
        elif method == "Lir":
            modelName = "LinearRegression"
        elif method == "LDA":
            modelName = "LDA"
        elif method == "RFreg":
            modelName = "RFreg"
        else:
            modelName = method
        return modelName

