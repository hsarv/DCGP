from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from .Individual import Individual
from .MulticlassLinearReg import MulticlassLinearReg
from .Node import Node
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge


def convert_last(msf, X):
	ret = pd.DataFrame()
	for i in range(len(msf)):
		a = msf[i].calculate(X)
		ret["#" + str(i)] = np.transpose(a[0])
	return ret
def calculate_correlation(individual, feature_set, Tr_X, Tr_Y):
	feature_set_Output = convert_last(feature_set, Tr_X)
	output = convert_indiv(individual, Tr_X)
	y1 = feature_set_Output.values
	y2 = output.values

	mutual = False
	corr = False

	if mutual:
		minfo = mutual_info_regression(y1, y2, discrete_features=False)
		corr = np.max(minfo)
		thr = 0.4

	elif corr:
		correlations = np.corrcoef(y2, y1, rowvar=False)[0, 1:]
		corr = np.max(np.abs(correlations))
		thr = 0.6
	else:
		# Add a constant column for the intercept term
		X = add_constant(feature_set_Output)
		vif_data1 = pd.DataFrame()
		vif_data1["Feature"] = X.columns
		vif_data1["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
		max_corr1 = np.max(vif_data1["VIF"].iloc[1:])
		# if max_corr1 > 2:
		# 	print(max_corr1)

		X['new'] = y2
		# Calculate VIF for each feature
		vif_data = pd.DataFrame()
		vif_data["Feature"] = X.columns
		vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

		corr = vif_data["VIF"].iloc[-1]
		thr = 4

	return corr, thr


def tournament(rng, population,n):
	candidates=[]
	if len(population) < n:
			return population[0],0

	for i in range(n):

		if isinstance(rng, np.random._generator.Generator):
			value = rng.integers(0, len(population) - 1)  # Use 'integers' for the new RNG
		else:
			value = rng.randint(0, len(population) - 1)

		candidates.append(value)
	return population[min(candidates)],min(candidates)


def convert1DClass(y):
	Y = np.zeros((len(y), 1))
	Y = np.argmax(np.max(y, axis=0))
	return Y


def GetRep(population):
	for i in range(len(population)):
		if population[i].isRep:
			candidates = i
	return population[candidates], candidates


def getOffspring(msf,pool_population,rng, allpopulation, i,numGeneration, tournament_size, Tr_X,Tr_Y,crossover_type,idd,operators, terminals, max_depth, model_name, fitnessType,residual,cross_rate):
	if numGeneration == 0:
		isCross = 0
	else:
		isCross = rng.random() <cross_rate
	desc = None
	availableXO = [0, 1, 2,3]
	availableMT = [0,1,2]
	if isCross:
		if crossover_type == 0:
			whichXO = 0
		elif crossover_type == 1:
			whichXO = 1
		elif crossover_type==2:
			whichXO = 2
		elif crossover_type==3:
			whichXO = 3
		elif crossover_type == 4:
			whichXO = availableXO[rng.randint(0, len(availableXO)-1)]

		elif crossover_type == 5:
			whichXO =5
		elif crossover_type == 6:
			whichXO =6
		elif crossover_type == 7:
			whichXO =7
		elif crossover_type == 8:
			whichXO = 7
		elif crossover_type == 9:
			# availableXO = [7, 9]
			# whichXO = availableXO[rng.randint(0, len(availableXO) - 1)]
			whichXO = 9
		elif crossover_type == 10:
			whichXO =10

		elif crossover_type == 11:
			whichXO =11
		elif crossover_type == 20:
			whichXO =20
		else:
			availableXO = [0, 1]
			whichXO = availableXO[rng.randint(0, len(availableXO)-1)]

		if whichXO == 0:
			desc, id1, id2 = STXO(rng, allpopulation[i], tournament_size)
			p1 = i
			p2 = i
		elif whichXO == 1:
			j = idd
			desc, id1, id2 = STXOMix(rng, allpopulation[i], allpopulation[j], tournament_size,Tr_X,Tr_Y,operators, terminals, max_depth, model_name, fitnessType)
			# desc, id1, id2 = betwXO(rng, allpopulation[i], allpopulation[j], tournament_size)
			p1 = i
			p2 = j
		elif whichXO == 2:
			j = idd
			desc, id1, id2 = RepXO(rng, allpopulation[i], allpopulation[j], tournament_size)
			p1 = i
			p2 = j
		elif whichXO == 3:
			j = rng.randint(0, len(allpopulation) - 1)
			if not len(allpopulation) == 1:
				while j == i:
					j = rng.randint(0, len(allpopulation) - 1)
			desc, id1, id2 = TwoRepXO(rng, allpopulation[i], allpopulation[j], 1)
			p1 = i
			p2 = j
		elif whichXO == 5:
			desc, id1, id2 = RandomXO(tournament_size, rng, allpopulation[0],operators, terminals, max_depth, model_name, fitnessType)
			p1 = i
			p2 = i
		elif whichXO == 6:
			desc, id1, id2 = binaryXO(rng, allpopulation[0],operators, terminals, max_depth, model_name, fitnessType)
			p1 = i
			p2 = i
		elif whichXO == 7:
			desc, id1, id2 = RandomXO(tournament_size,rng, allpopulation[i], operators, terminals, max_depth, model_name, fitnessType)
			p1 = i
			p2 = i
		elif whichXO == 8:
			desc, id1, id2 = binary_linear_XO(tournament_size, rng, allpopulation[i], operators, terminals, max_depth,
									  model_name, fitnessType)
			p1 = i
			p2 = i
		elif whichXO == 9:
			desc, id1, id2 = linear_XO(pool_population,tournament_size, rng, allpopulation[i], operators, terminals, max_depth,
									   model_name, fitnessType, residual, Tr_X, Tr_Y)
			p1 = i
			p2 = i
		elif whichXO == 10:
			desc, id1, id2 = linear_XO_ru(pool_population,tournament_size, rng, allpopulation[i], operators, terminals, max_depth,
									   model_name, fitnessType, residual, Tr_X, Tr_Y)
			p1 = i
			p2 = i
		elif whichXO == 11:
			desc, id1, id2 = linear_XO_Lasso(pool_population,tournament_size, rng, allpopulation[i], operators, terminals, max_depth, model_name, fitnessType, residual, Tr_X, Tr_Y)
			p1 = i
			p2 = i

		elif whichXO == 20:
			desc, id1, id2 = STXO_Corr(rng,  allpopulation[i], tournament_size, msf, Tr_X, Tr_Y,operators, terminals, max_depth, model_name, fitnessType, correlation_threshold=0.5, num_candidates=5)
			p1 = i
			p2 = i

	else:
		#whichMut = availableMT[  rng.randint(0,len(availableMT)-1 ) ]
		whichMut = 0
		if whichMut == 0:
			desc, id1 = STMUT(rng, allpopulation[i], tournament_size)
			p1 = i
			p2 = i
		elif whichMut == 1:
			desc, id1 = RepMUT(rng, allpopulation[i], tournament_size)
			p1 = i
			p2 = i
		id2 = 0
	return desc, id1, id2, isCross, p1, p2


def discardDeep(population, limit):
	ret = []
	for ind in population:
		if ind.getDepth() <= limit:
			ret.append(ind)
	return ret

def selectFirstParent(rng, population, tor_size):
	ind1, id1 = tournament(rng, population, 2)
	sub_tree=ind1
	return sub_tree
def selectsecondParent(rng, population, tor_size):
	ind1, id1 = tournament(rng, population, 2)
	sub_tree=ind1
	return sub_tree
def tournament_ru(rng, population,n,residual,Tr_X,Tr_Y):
	first_sub, bext_ru=best_first_sub_tree(rng, population, n, residual,Tr_X)
	second_sub_tree, bext_ru2=best_second_sub_tree(rng, population, n, residual,Tr_X,first_sub,bext_ru)
	return first_sub,second_sub_tree

def convert_indiv(indiv, X):
	'''
	Returns the converted input space.
	'''
	ret = pd.DataFrame()
	for i in range(len(indiv.dimensions)):
		a = indiv.dimensions[i].calculate(X)
		ret["#" + str(i)] = a
	return ret

def convert(n1, X):
	ret = pd.DataFrame()
	a = n1.calculate(X)
	ret["#" + str(0)] = a
	return ret
def best_first_sub_tree(rng, population, t_size, residual,Tr_X):
	res = pd.DataFrame()
	res["#" + str(0)] = residual
	bext_ru = -1000
	r_ind1 = rng.randint(0, len(population) - 1)
	ind1 = population[r_ind1]
	d1 = ind1.getDimensions()
	r1 = rng.randint(0, len(d1) - 1)
	n1_best = d1[r1].getRandomNode(rng)
	for i in range(t_size):
		r_ind1 = rng.randint(0, len(population) - 1)
		ind1 = population[r_ind1]
		d1 = ind1.getDimensions()
		r1 = rng.randint(0, len(d1) - 1)
		n1 = d1[r1].getRandomNode(rng)
		hyper_x = convert(n1, Tr_X)
		ru = res.corrwith(hyper_x)
		if ru[0]> bext_ru:
			bext_ru = ru[0]
			n1_best = n1
	return n1_best,bext_ru

def cal_second_measure_ru(residual,hyper_x1,hyper_x2,ru1y):
	ru2y = residual.corrwith(hyper_x2)
	ru12 = hyper_x1.corrwith(hyper_x2)
	F= ((ru2y-(ru1y*ru12))*(ru2y-(ru1y*ru12)))/(1-(ru12*ru12))
	return F

def best_second_sub_tree(rng, population, t_size, residual,Tr_X, first_sub_tree,r1y):
	hyper_x1 = convert(first_sub_tree, Tr_X)
	res = pd.DataFrame()
	res["#" + str(0)] = residual
	bext_ru2 = -1000
	r_ind = rng.randint(0, len(population) - 1)
	ind = population[r_ind]
	d = ind.getDimensions()
	r = rng.randint(0, len(d) - 1)
	n2_best = d[r].getRandomNode(rng)
	for i in range(t_size):
		r_ind = rng.randint(0, len(population) - 1)
		ind = population[r_ind]
		d = ind.getDimensions()
		r = rng.randint(0, len(d) - 1)
		n = d[r].getRandomNode(rng)
		hyper_x2 = convert(n, Tr_X)
		ru_second_measure =cal_second_measure_ru(res,hyper_x1,hyper_x2,r1y)
		if ru_second_measure[0] > bext_ru2:
			bext_ru2 = ru_second_measure[0]
			n2_best = n

	return n2_best,bext_ru2

def linear_XO(pool_population,tournament_size,rng, population,operators, terminals, max_depth, model_name, fitnessType, residual, Tr_X, Tr_Y):

	n1, n2 = tournament_ru(rng, pool_population, 10, residual, Tr_X, Tr_Y)
	model = LinearRegression(fit_intercept=True)


	hyper_X_1 = convert(n1,Tr_X)
	hyper_X_2 = convert(n2, Tr_X)
	hyper_X= pd.concat([hyper_X_1, hyper_X_2], axis=1)

	# model = Ridge()

	# # Perform cross-validation
	# cv_scores = cross_val_score(model, hyper_X, residual, cv=5, scoring='neg_mean_squared_error')
	# print(f"Mean CV MSE: {-cv_scores.mean()}")

	# Hyperparameter tuning with Grid Search
	# param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
	# grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
	# grid_search.fit(hyper_X, residual)
	#
	# # Best model after Grid Search
	# best_model = grid_search.best_estimator_
	# #print( grid_search.best_estimator_.alpha)
	model.fit(hyper_X, residual)
	# pr=model.predict(hyper_X)
	#
	# mse = 1 * mean_squared_error(pr, residual)
	# #print(mse)
	a = model.coef_[0]
	b = model.coef_[1]
	c= model.intercept_

	ind = population[0]
	#
	n_ax1 = Node()
	op, n_args = ind.operators[2]
	n_ax1.setValue(op)
	n_ax1.setBranche(n1, 0)

	n_a = Node()
	n_a.setValue(a)
	n_ax1.setBranche(n_a, 1)
	#______________________________
	n_bx2 = Node()
	n_bx2.setValue(op)
	n_bx2.setBranche(n2, 0)

	n_b = Node()
	n_b.setValue(b)
	n_bx2.setBranche(n_b, 1)
#-----------------------------------
	op, n_args = ind.operators[0]
	n_ax1_bx2 = Node()
	n_ax1_bx2 .setValue(op)
	n_ax1_bx2.setBranche(n_ax1, 0)
	n_ax1_bx2.setBranche(n_bx2, 1)
#________________
	n_c = Node()
	n_c.setValue(c)
#_______________
	n_end = Node()
	op, n_args = ind.operators[0]
	n_end .setValue(op)
	n_end.setBranche(n_ax1_bx2, 0)
	n_end.setBranche(n_c, 1)

	ind_end = Individual(ind.operators, ind.terminals, ind.max_depth, ind.model_name, ind.fitnessType)
	ind_end.createNode(n_end)
	ret = []
	ret.append(ind_end)

	return ret,0,0
def linear_XO_Lasso(pool_population,tournament_size,rng, population,operators, terminals, max_depth, model_name, fitnessType, residual, Tr_X, Tr_Y):
	n1, id1 = tournament(rng, population, 2)
	n2, id2 = tournament(rng, population, 2)


	hyper_X_1 = convert_indiv(n1,Tr_X)
	hyper_X_2 = convert_indiv(n2, Tr_X)
	hyper_X= pd.concat([hyper_X_1, hyper_X_2], axis=1)

	#model = Ridge()
	model=MulticlassLinearReg()
	# # Perform cross-validation
	# cv_scores = cross_val_score(model, hyper_X, residual, cv=5, scoring='neg_mean_squared_error')
	# print(f"Mean CV MSE: {-cv_scores.mean()}")

	# Hyperparameter tuning with Grid Search
	# param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
	# grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
	# grid_search.fit(hyper_X, residual)
	#
	# # Best model after Grid Search
	# best_model = grid_search.best_estimator_
	# #print( grid_search.best_estimator_.alpha)
	model.fit(hyper_X, residual)
	# pr=model.predict(hyper_X)
	#
	# mse = 1 * mean_squared_error(pr, residual)
	# #print(mse)
	a = model.coef_[0]
	b = model.coef_[1]


	ind = population[0]
	#
	n_ax1 = Node()
	op, n_args = ind.operators[2]
	n_ax1.setValue(op)
	n_ax1.setBranche(n1, 0)

	n_a = Node()
	n_a.setValue(a)
	n_ax1.setBranche(n_a, 1)
	#______________________________
	n_bx2 = Node()
	n_bx2.setValue(op)
	n_bx2.setBranche(n2, 0)

	n_b = Node()
	n_b.setValue(b)
	n_bx2.setBranche(n_b, 1)
#-----------------------------------
	op, n_args = ind.operators[0]
	n_ax1_bx2 = Node()
	n_ax1_bx2 .setValue(op)
	n_ax1_bx2.setBranche(n_ax1, 0)
	n_ax1_bx2.setBranche(n_bx2, 1)


	ind_end = Individual(ind.operators, ind.terminals, ind.max_depth, ind.model_name, ind.fitnessType)
	ind_end.createNode(n_ax1_bx2)
	ret = []
	ret.append(ind_end)

	return ret,0,0

def linear_XO_ru(pool_population,tournament_size,rng, population,operators, terminals, max_depth, model_name, fitnessType, residual, Tr_X, Tr_Y):

	first_sub, bext_ru=best_first_sub_tree(rng, pool_population, tournament_size, residual,Tr_X)
	n1=first_sub
	ind = population[0]
	ind_end = Individual(ind.operators, ind.terminals, ind.max_depth, ind.model_name, ind.fitnessType)
	ind_end.createNode(n1)
	ret = []
	ret.append(ind_end)

	return ret,0,0


def binary_linear_XO(rng, population,operators, terminals, max_depth, model_name, fitnessType):
	ind1, id1 = tournament(rng, population, 2)
	ind2, id2 = tournament(rng, population, 2)

	d1 = ind1.getDimensions()
	d2 = ind2.getDimensions()

	r1 = rng.randint(0, len(d1)-1)
	r2 = rng.randint(0, len(d2)-1)

	n1 = d1[r1].getRandomNode(rng)
	n2 = d2[r2].getRandomNode(rng)


	n = Node()
	operatorlist=ind1.operators.copy()
	op, n_args = ind1.operators[rng.randint(0, len(operatorlist) - 1)]
	n.setValue(op)
	n.setBranche(n1,0)
	n.setBranche(n2,1)
	indexes = [y[0] for y in operatorlist].index(op)
	del operatorlist[indexes]
	nn = Node()
	op, n_args = operatorlist[rng.randint(0, len(operatorlist) - 1)]
	nn.setValue(op)
	nn.setBranche(n1, 0)
	nn.setBranche(n2, 1)

	ind_end = Individual(ind2.operators, ind1.terminals, ind1.max_depth, ind1.model_name, ind1.fitnessType)
	ind_end.createNode(n)

	ind2_end = Individual(ind2.operators, ind2.terminals, ind2.max_depth, ind2.model_name, ind2.fitnessType)
	ind2_end.createNode(nn)

	ret=[]
	ret.append(ind_end)
	ret.append(ind2_end)

	return ret, id1, 0


def boosting_XO(tournament_size,rng, population,operators, terminals, max_depth, model_name, fitnessType):
	ind1, id1 = tournament(rng, population, 2)
	ind2, id2 = tournament(rng, population, 2)

	d1 = ind1.getDimensions()
	d2 = ind2.getDimensions()

	r1 = rng.randint(0, len(d1)-1)
	r2 = rng.randint(0, len(d2)-1)

	n1 = d1[r1].getRandomNode(rng)
	n2 = d2[r2].getRandomNode(rng)


	n = Node()
	operatorlist=ind1.operators.copy()
	op, n_args = ind1.operators[rng.randint(0, len(operatorlist) - 1)]
	n.setValue(op)
	n.setBranche(n1,0)
	n.setBranche(n2,1)
	indexes = [y[0] for y in operatorlist].index(op)
	del operatorlist[indexes]
	nn = Node()
	op, n_args = operatorlist[rng.randint(0, len(operatorlist) - 1)]
	nn.setValue(op)
	nn.setBranche(n1, 0)
	nn.setBranche(n2, 1)

	ind_end = Individual(ind2.operators, ind1.terminals, ind1.max_depth, ind1.model_name, ind1.fitnessType)
	ind_end.createNode(n)

	ind2_end = Individual(ind2.operators, ind2.terminals, ind2.max_depth, ind2.model_name, ind2.fitnessType)
	ind2_end.createNode(nn)

	ret=[]
	ret.append(ind_end)
	ret.append(ind2_end)

	return ret, id1, 0

def binaryXO(rng, population,operators, terminals, max_depth, model_name, fitnessType):
	ind1, id1 = tournament(rng, population, 2)
	ind2, id2 = tournament(rng, population, 2)

	d1 = ind1.getDimensions()
	d2 = ind2.getDimensions()

	r1 = rng.randint(0, len(d1)-1)
	r2 = rng.randint(0, len(d2)-1)

	n1 = d1[r1].getRandomNode(rng)
	n2 = d2[r2].getRandomNode(rng)


	n = Node()
	operatorlist=ind1.operators.copy()
	op, n_args = ind1.operators[rng.randint(0, len(operatorlist) - 1)]
	n.setValue(op)
	n.setBranche(n1,0)
	n.setBranche(n2,1)
	indexes = [y[0] for y in operatorlist].index(op)
	del operatorlist[indexes]
	nn = Node()
	op, n_args = operatorlist[rng.randint(0, len(operatorlist) - 1)]
	nn.setValue(op)
	nn.setBranche(n1, 0)
	nn.setBranche(n2, 1)

	ind_end = Individual(ind2.operators, ind1.terminals, ind1.max_depth, ind1.model_name, ind1.fitnessType)
	ind_end.createNode(n)

	ind2_end = Individual(ind2.operators, ind2.terminals, ind2.max_depth, ind2.model_name, ind2.fitnessType)
	ind2_end.createNode(nn)

	ret=[]
	ret.append(ind_end)
	ret.append(ind2_end)

	return ret, id1, 0

def RandomXO(tournament_size,rng, population,operators, terminals, max_depth, model_name, fitnessType):
	'''
	Randomly selects one node from each of two individuals; swaps the node and
	sub-nodes; and returns the two new Individuals as the offspring.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	ind1, id1 = tournament(rng, population, tournament_size)

	ind2 = Individual(operators, terminals, max_depth, model_name, fitnessType)
	ind2.create(rng, n_dims=1)

	d1 = ind1.getDimensions()
	d2 = ind2.getDimensions()

	r1 = rng.randint(0, len(d1)-1)
	r2 = rng.randint(0, len(d2)-1)

	n1 = d1[r1].getRandomNode(rng)
	n2 = d2[r2].getRandomNode(rng)

	n1.swap(n2)

	ret = []
	for d in [d1, d2]:
		i = Individual(ind1.operators, ind1.terminals, ind1.max_depth, ind1.model_name, ind1.fitnessType)
		i.copy(d)
		ret.append(i)
	return ret, id1, 0
def STXO(rng, population, tournament_size):
	'''
	Randomly selects one node from each of two individuals; swaps the node and
	sub-nodes; and returns the two new Individuals as the offspring.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	ind1, id1 = tournament(rng, population, tournament_size)
	ind2, id2 = tournament(rng, population, tournament_size)

	d1 = ind1.getDimensions()
	d2 = ind2.getDimensions()

	r1 = rng.randint(0, len(d1)-1)
	r2 = rng.randint(0, len(d2)-1)

	n1 = d1[r1].getRandomNode(rng)
	n2 = d2[r2].getRandomNode(rng)

	n1.swap(n2)

	ret = []
	for d in [d1, d2]:
		i = Individual(ind1.operators, ind1.terminals, ind1.max_depth, ind1.model_name, ind1.fitnessType)
		i.copy(d)
		ret.append(i)
	return ret, id1, id2


def STXO_Corr(rng1, population, tournament_size, feature_set, Tr_X, Tr_Y, operators, terminals, max_depth, model_name,
			  fitnessType, correlation_threshold=0.5, num_candidates=5):
	trial = 0
	children = []

	while len(children) < 2:

		ind2 = Individual(operators, terminals, max_depth, model_name, fitnessType)
		ind2.create(rng1, n_dims=1)
		seed_sequence = np.random.SeedSequence()
		seed = seed_sequence.generate_state(1)[0]  # Generate a new seed each time
		rng = np.random.default_rng(seed)

		ind1, id1 = tournament(rng, population, tournament_size)

		d1 = ind1.getDimensions()[0]
		d2 = ind2.getDimensions()[0]
		seed_sequence = np.random.SeedSequence()
		seed = seed_sequence.generate_state(1)[0]
		rng = np.random.default_rng(seed)

		n1 = d1.getRandomNode(rng)
		n2 = d2.getRandomNode(rng)

		n1.swap(n2)

		child1 = Individual(ind1.operators, ind1.terminals, ind1.max_depth, ind1.model_name, ind1.fitnessType)
		child1.copy([d1])

		child2 = Individual(ind2.operators, ind2.terminals, ind2.max_depth, ind2.model_name, ind2.fitnessType)
		child2.copy([d2])
		corr1,thr1 = calculate_correlation(child1, feature_set, Tr_X, Tr_Y)

		if corr1 <thr1:
			children.append(child1)

		corr2 ,thr2 = calculate_correlation(child2, feature_set, Tr_X, Tr_Y)

		if corr2 < thr2:
			children.append(child2)

		if len(children) >1:
			break
		trial=trial+1
	if trial>5:
		print(trial)
	best_child1 = children[0]
	best_child2 = children[1]
	return (best_child1, best_child2), id1, id1


def STXO_Corr2(rng, population, tournament_size, feature_set, Tr_X, Tr_Y, operators, terminals, max_depth, model_name, fitnessType, correlation_threshold=0.5, num_candidates=5):

    ind2 = Individual(operators, terminals, max_depth, model_name, fitnessType)
    ind2.create(rng, n_dims=1)

    seed_sequence = np.random.SeedSequence()
    seed = seed_sequence.generate_state(1)[0]  # Generate a new seed each time
    rng = np.random.default_rng(seed)

    # Perform tournament selection to pick two parents
    ind1, id1 = tournament(rng, population, tournament_size)


    # Generate all children independently
    children = []
    for _ in range(num_candidates):
        d1 = ind1.getDimensions()[0]
        d2 = ind2.getDimensions()[0]
        # Use different seeds for each candidate to ensure diversity
        seed_sequence = np.random.SeedSequence()
        seed = seed_sequence.generate_state(1)[0]  # Generate a new seed each time
        rng = np.random.default_rng(seed)  # Create a new RNG with each seed

        # Select random nodes from both parents
        n1 = d1.getRandomNode(rng)
        n2 = d2.getRandomNode(rng)
        n3 = d1.getRandomNode(rng)
        n4 = d2.getRandomNode(rng)
        n1.swap(n2)
        # n3.swap(n4)  # You might want to swap nodes for multiple crossovers

        # Create a deep copy of both parents' dimensions for the offspring
        child1 = Individual(ind1.operators, ind1.terminals, ind1.max_depth, ind1.model_name, ind1.fitnessType)
        child1.copy([d1])  # Copy the first parent's dimension into the child

        child2 = Individual(ind2.operators, ind2.terminals, ind2.max_depth, ind2.model_name, ind2.fitnessType)
        child2.copy([d2])  # Copy the second parent's dimension into the second child

        # Add both children independently to the candidates list
        children.append(child1)
        children.append(child2)


    # List to store the correlation values and corresponding children
    correlation_results = []

    # Evaluate correlations and store the correlation values
    for child in children:
        corr = calculate_correlation(child, feature_set, Tr_X, Tr_Y)
        # Store the correlation value with the child
        correlation_results.append((corr, child))

    # Sort the children by correlation value
    correlation_results.sort(key=lambda x: x[0])  # Sort by correlation, ascending order

    # Select the two children with the least correlation
    best_corr1, best_child1 = correlation_results[0]  # Child with the least correlation
    best_corr2, best_child2 = correlation_results[1]  # Second child with the least correlation

    # If the sum of correlations exceeds the threshold, fall back to parents
    if best_corr1 + best_corr2 > correlation_threshold:
        return [ind1, ind2], id1, id1

    return (best_child1, best_child2), id1, id1

def STXO_Corr1(rng, population, tournament_size, feature_set, Tr_X, Tr_Y,operators, terminals, max_depth, model_name, fitnessType, correlation_threshold=0.5, num_candidates=5):
	'''
	Performs subtree crossover by testing multiple crossover points and selecting the best.

	Parameters:
	rng (Random): Random number generator instance.
	population (list): A list of Individuals, sorted from best to worse.
	tournament_size (int): Number of participants in the tournament selection.
	feature_set (numpy.array or pandas.DataFrame): A feature set to calculate correlation.
	correlation_threshold (float): Maximum allowed correlation between offspring and feature_set.
	num_candidates (int): Number of candidate crossover points to evaluate.

	Returns:
	list: Two new offspring (Individuals).
	int: ID of the first parent.
	int: ID of the second parent.
	'''
	ind2 = Individual(operators, terminals, max_depth, model_name, fitnessType)
	ind2.create(rng, n_dims=1)

	seed_sequence = np.random.SeedSequence()
	seed = seed_sequence.generate_state(1)[0]  # Generate a new seed each time
	rng = np.random.default_rng(seed)

	# Perform tournament selection to pick two parents
	ind1, id1 = tournament(rng, population, tournament_size)
	seed_sequence = np.random.SeedSequence()
	seed = seed_sequence.generate_state(1)[0]  # Generate a new seed each time
	rng = np.random.default_rng(seed)
	# ind2, id2 = tournament(rng, population, tournament_size)


	# Generate candidates by trying multiple crossover points
	candidates = []
	for _ in range(num_candidates):
		d1 = ind1.getDimensions()[0]
		d2 = ind2.getDimensions()[0]
		# Use different seeds for each candidate to ensure diversity
		seed_sequence = np.random.SeedSequence()
		seed = seed_sequence.generate_state(1)[0]  # Generate a new seed each time
		rng = np.random.default_rng(seed)  # Create a new RNG with each seed

		# Select random nodes from both parents
		n1 = d1.getRandomNode(rng)
		n2 = d2.getRandomNode(rng)
		n3 = d1.getRandomNode(rng)
		n4 = d2.getRandomNode(rng)
		n1.swap(n2)
		# n3.swap(n4)
		# Create a deep copy of both parents' dimensions for the offspring
		child1 = Individual(ind1.operators, ind1.terminals, ind1.max_depth, ind1.model_name, ind1.fitnessType)
		child1.copy([d1])  # Copy the first parent's dimension into the child

		child2 = Individual(ind2.operators, ind2.terminals, ind2.max_depth, ind2.model_name, ind2.fitnessType)
		child2.copy([d2])  # Copy the second parent's dimension into the second child

		# Revert the swap for the next iteration
		# Store the candidate pair of children
		candidates.append((child1, child2))

	# Evaluate correlations and select the best pair
	best_pair = None
	best_correlation = float('inf')
	for child1, child2 in candidates:
		corr1 = calculate_correlation(child1, feature_set, Tr_X, Tr_Y,)
		corr2 = calculate_correlation(child2, feature_set, Tr_X, Tr_Y,)
		max_corr = max(corr1, corr2)

		if max_corr < best_correlation:
			best_correlation = max_corr
			best_pair = (child1, child2)

	# If the best correlation exceeds the threshold, fall back to parents
	if best_correlation > 1:
		return [ind1, ind2], id1, id1

	return best_pair, id1, id1

def STXOMix(rng, populationi, populationj, tournament_size, Tr_x, Tr_y,operators, terminals, max_depth, model_name, fitnessType):
	'''
	Randomly selects one node from each of two individuals; swaps the node and
	sub-nodes; and returns the two new Individuals as the offspring.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	ind1, id1 = tournament(rng, populationi, tournament_size)
	# ind2, id2 = tournament(rng, populationj, tournament_size)

	ind2 = Individual(operators, terminals, max_depth, model_name, fitnessType)
	ind2.create(rng, n_dims=1)


	d1 = ind1.getDimensions()
	d2 = ind2.getDimensions()

	r1 = rng.randint(0, len(d1)-1)
	r2 = rng.randint(0, len(d2)-1)
	numofcheck = 3
	numberOfOsubtree=min(numofcheck, ind1.getSize()-3)
	best_mi_y = 0
	n1_best = d1[0]
	hyper_best = ind1.convert(Tr_x)
	sizelist = list(range(0, ind1.getSize()-3))
	for k in range(numberOfOsubtree):
		val = sizelist[rng.randint(0, len(sizelist) - 1)]
		n1 = d1[r1].getRandomNode(rng, val)
		indexes = [y for y in sizelist].index(val)
		del sizelist[indexes]
		ii = Individual(ind1.operators, ind1.terminals, ind1.max_depth, ind1.model_name, ind1.fitnessType)
		ii.createNode(n1)
		hyper_x = ii.convert(Tr_x)
		mi_y1 = mutual_info_classif(hyper_x, Tr_y)
		if mi_y1[0] > best_mi_y:
			best_mi_y = mi_y1[0]
			n1_best = n1
			hyper_best=hyper_x
	best_mi_y2 = 0
	numberOfOsubtree2 = min(numofcheck, ind2.getSize()-3)
	n2_best=d2[0]
	mbetwnbest=100
	sizelist=list(range(0, ind2.getSize()-3))
	for k in range(numberOfOsubtree2):
		val=sizelist[rng.randint(0, len(sizelist) - 1)]
		n2 = d2[r2].getRandomNode(rng,val)
		indexes = [y for y in sizelist].index(val)
		del sizelist[indexes]
		ii=Individual(ind2.operators, ind2.terminals, ind2.max_depth, ind2.model_name, ind2.fitnessType)
		ii.createNode(n2)
		hyper_x = ii.convert(Tr_x)
		mi_y2 = mutual_info_classif(hyper_x, Tr_y)
		if mi_y2[0] > best_mi_y2:
			# mbetwn=mutual_info_regression(hyper_x, hyper_best, discrete_features=False)
			# if mbetwn<mbetwnbest:
			best_mi_y2 = mi_y2[0]
			n2_best = n2
				# mbetwnbest=mbetwn

	n = Node()
	operatorlist=ind1.operators.copy()
	op, n_args = ind1.operators[rng.randint(0, len(operatorlist) - 1)]
	n.setValue(op)
	n.setBranche(n1_best,0)
	n.setBranche(n2_best,1)
	indexes = [y[0] for y in operatorlist].index(op)
	del operatorlist[indexes]
	nn = Node()
	op, n_args = operatorlist[rng.randint(0, len(operatorlist) - 1)]
	nn.setValue(op)
	nn.setBranche(n1_best, 0)
	nn.setBranche(n2_best, 1)

	ind_end = Individual(ind2.operators, ind1.terminals, ind1.max_depth, ind1.model_name, ind1.fitnessType)
	ind_end.createNode(n)

	ind2_end = Individual(ind2.operators, ind2.terminals, ind2.max_depth, ind2.model_name, ind2.fitnessType)
	ind2_end.createNode(nn)

	ret=[]
	ret.append(ind_end)
	ret.append(ind2_end)
	return ret, id1, 0


def M3XO(rng, population, tournament_size):
	'''
	Randomly selects one dimension from each of two individuals; swaps the 
	dimensions; and returns the two new Individuals as the offspring.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''

	ind1, id1 = tournament(rng, population, tournament_size)
	ind2, id1 = tournament(rng, population, tournament_size)

	d1 = ind1.getDimensions()
	d2 = ind2.getDimensions()

	r1 = rng.randint(0,len(d1)-1)
	r2 = rng.randint(0,len(d2)-1)

	d1.append(d2[r2])
	d2.append(d1[r1])
	d1.pop(r1)
	d2.pop(r2)

	ret = []
	for d in [d1,d2]:
		i = Individual(ind1.operators, ind1.terminals, ind1.max_depth, ind1.model_name, ind1.fitnessType)
		i.copy(d)
		ret.append(i)
	return ret

def betwXO(rng,populationi,populationj, tournament_size):
	'''
	Randomly selects one node from each of two individuals; swaps the node and
	sub-nodes; and returns the two new Individuals as the offspring.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	ind1, id1 = tournament(rng, populationi, tournament_size)
	ind2, id2 = tournament(rng, populationj, tournament_size)

	d1 = ind1.getDimensions()
	d2 = ind2.getDimensions()

	r1 = rng.randint(0, len(d1)-1)
	r2 = rng.randint(0, len(d2)-1)

	n1 = d1[r1].getRandomNode(rng)
	n2 = d2[r2].getRandomNode(rng)

	n1.swap(n2)

	ret = []
	for d in [d1,d2]:
		i = Individual(ind1.operators, ind1.terminals, ind1.max_depth, ind1.model_name, ind1.fitnessType)
		i.copy(d)
		ret.append(i)
	return ret, id1, id2

def RepXO(rng,populationi,populationj, tournament_size):
	'''
	Randomly selects one node from each of two individuals; swaps the node and
	sub-nodes; and returns the two new Individuals as the offspring.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''

	ind1, id1 = GetRep(populationi)
	ind2, id2 = tournament(rng, populationj, tournament_size)

	d1 = ind1.getDimensions()
	d2 = ind2.getDimensions()

	r1 = rng.randint(0, len(d1)-1)
	r2 = rng.randint(0, len(d2)-1)

	n1 = d1[r1].getRandomNode(rng)
	n2 = d2[r2].getRandomNode(rng)

	n1.swap(n2)

	ret = []
	for d in [d1, d2]:
		i = Individual(ind1.operators, ind1.terminals, ind1.max_depth, ind1.model_name, ind1.fitnessType)
		i.copy(d)
		ret.append(i)
	return ret, id1, id2
def TwoRepXO(rng,populationi,populationj, tournament_size):
	'''
	Randomly selects one node from each of two individuals; swaps the node and
	sub-nodes; and returns the two new Individuals as the offspring.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''

	ind1, id1 = GetRep(populationi)
	ind2, id2 = GetRep(populationj)

	d1 = ind1.getDimensions()
	d2 = ind2.getDimensions()

	r1 = rng.randint(0, len(d1)-1)
	r2 = rng.randint(0, len(d2)-1)

	n1 = d1[r1].getRandomNode(rng)
	n2 = d2[r2].getRandomNode(rng)

	n1.swap(n2)

	ret = []
	for d in [d1, d2]:
		i = Individual(ind1.operators, ind1.terminals, ind1.max_depth, ind1.model_name, ind1.fitnessType)
		i.copy(d)
		ret.append(i)
	return ret, id1, id2
def STMUT(rng, population, tournament_size):
	'''
	Randomly selects one node from a single individual; swaps the node with a 
	new, node generated using Grow; and returns the new Individual as the offspring.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	ind1, id1= tournament(rng, population, tournament_size)
	d1 = ind1.getDimensions()
	r1 = rng.randint(0,len(d1)-1)
	n1 = d1[r1].getRandomNode(rng)
	n = Node()
	n.create(rng, ind1.operators, ind1.terminals, ind1.max_depth)
	n1.swap(n)


	ret = []
	i = Individual(ind1.operators, ind1.terminals, ind1.max_depth, ind1.model_name, ind1.fitnessType)
	i.copy(d1)
	ret.append(i)
	return ret, id1
def RepMUT(rng, population, tournament_size):
	'''
	Randomly selects one node from a single individual; swaps the node with a
	new, node generated using Grow; and returns the new Individual as the offspring.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	ind1, id1 = GetRep(population)
	d1 = ind1.getDimensions()
	r1 = rng.randint(0,len(d1)-1)
	n1 = d1[r1].getRandomNode(rng)
	n = Node()
	n.create(rng, ind1.operators, ind1.terminals, ind1.max_depth)
	n1.swap(n)


	ret = []
	i = Individual(ind1.operators, ind1.terminals, ind1.max_depth, ind1.model_name, ind1.fitnessType)
	i.copy(d1)
	ret.append(i)
	return ret, id1

def M3ADD(rng, population, tournament_size, dim_max):
	'''
	Randomly generates a new node using Grow; this node is added to the list of
	dimensions; the new Individual is returned as the offspring.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	ind1, id1 = tournament(rng, population, tournament_size)
	ret = []

	if ind1.getNumberOfDimensions() < dim_max:
		d1 = ind1.getDimensions()
		n = Node()
		n.create(rng, ind1.operators, ind1.terminals, ind1.max_depth)
		d1.append(n)

		i = Individual(ind1.operators, ind1.terminals, ind1.max_depth, ind1.model_name, ind1.fitnessType)
		i.copy(d1)
		ret.append(i)

	return ret

def M3REM(rng, population, tournament_size, dim_min):
	'''
	Randomly selects one dimensions from a single individual; that dimensions is
	removed; the new Individual is returned as the offspring.

	Parameters:
	population (list): A list of Individuals, sorted from best to worse.
	'''
	ind1,id1 = tournament(rng, population, tournament_size)
	ret = []

	if ind1.getNumberOfDimensions() > dim_min:
		d1 = ind1.getDimensions()
		r1 = rng.randint(0,len(d1)-1)
		d1.pop(r1)
		
		i = Individual(ind1.operators, ind1.terminals, ind1.max_depth, ind1.model_name, ind1.fitnessType)
		i.copy(d1)
		ret.append(i)
	
	return ret
