class Methodsconfig:

    sygp = False
    classification = False
    one_pop = False
    ML = False
    k_split_trainingset = False
    Normalization = False
    tor1 = False
    cross = 0
    pop_size = 20
    Random_pop = False
    sygp_hierarchical = False

    def __init__(self,method1):
        self.sygp = False
        self.classification = False
        self.one_pop = False
        self.ML = False
        self.k_split_trainingset = False
        self.Normalization = False
        self.tor1 = False
        self.cross = 7
        self.pop_size = 20
        self.Random_pop = False
        self.FITNESS_TYPE=['Accuracy']

        if method1 == 'ML_classification':
            self.OUTPUT_DIR = 'ML_classification/'
            self.classification = True
            self.ML = True
            self.methodNames = ['LDA', 'RF', 'LR', 'KNN', 'DT', 'MLP', 'NC', 'SVM']

            self.FITNESS_TYPE = ["Accuracy"]
            self.datasetName = ['heart', 'yeast', 'segment', 'wav', 'movement_libras', 'vowel', 'mcd3', 'mcd10',
                                'bupa', 'haberman', 'ionosphere', 'pimaindiansdiabetes', 'wdbc', "sonar",
                                'balance_scale', 'iris', 'wine']
        elif method1 == 'sygp_regression':
            self.sygp = True
            self.OUTPUT_DIR = 'sygp_regression/'
            self.methodNames = ['RFreg','KNeighborsRegressor', 'SVR', 'MLPRegressor','GBDT', 'DecisionTreeRegressor', 'Ridge',
                                'LinearRegression']
            self.FITNESS_TYPE = ["MSE"]
            self.datasetName = [ 'bm4', 'Pollen', 'keijzer5', 'Toxicity', 'Concrete', 'Real_estate_valuation', 'RatPol2D', 'X05Y1Z15']
            # 'ET', 'AdaBoost','GBDT' ,'DART','XGBoost','KNeighborsRegressor','SVR', 'MLPRegressor','DecisionTreeRegressor','Ridge','LinearRegression'
        # ---------------------------------------
        elif method1 == 'sygp_classification':
            self.classification = True
            self.sygp = True
            self.OUTPUT_DIR = 'sygp_classification/'
            self.methodNames = ['LR','LDA', 'KNN', 'DT',  'NC', 'SVM', 'RF', 'MLP']
            self.Normalization = False
            self.FITNESS_TYPE = ["Accuracy"]
            self.cross = 7
            self.datasetName = ['bupa']
            # ['yeast', 'segment', 'wav', 'movement_libras', 'vowel', 'mcd10', 'bupa'
            #   'haberman', 'ionosphere', 'pimaindiansdiabetes', 'wdbc', "sonar", 'balance_scale', 'iris', 'wine','heart', 'mcd3']


    def get_parameter(self):

        self.OUTPUT_DIR = "all_results/" + self.OUTPUT_DIR
        my_list=[self.sygp,self.classification, self.ML, self.tor1,self.cross, self.k_split_trainingset,self.Normalization, self.datasetName,self.methodNames,self.pop_size,self.OUTPUT_DIR,self.FITNESS_TYPE]
        return my_list

