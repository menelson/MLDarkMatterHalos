'''
	A script to pik up the h5 data using pandas,
	create separate training, testing, and validation sets,
	and finally study some of the correlations between
	the features and the labels
'''

import os
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import df_scaler as dfs
from math import sqrt

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb

def main(correlations=False,inputData="galaxies.Z80.h5", outputPath="./Plots/"):
	df = pd.read_hdf(inputData, key='Galaxies', mode='r', start = 0, stop = 624645)

	# Halo properties should be features; galaxu properties should be labels
	features = ['Halo_mass', 'Halo_growth_rate', 'Halo_mass_peak', 
			'Scale_peak_mass', 'Scale_half_mass', 'Halo_radius', 
			'Concentration', 'Halo_spin', 'Intra_cluster_mass', 
			'X_pos', 'Y_pos', 'Z_pos', 'X_vel', 'Y_vel', 'Z_vel', 'Type']

	labels = ['Stellar_mass', 'SFR', 'Stellar_mass_obs', 'SFR_obs']
	
	# Correlations
	correlation = df.corr()
	mask = np.zeros(correlation.shape, dtype=bool)
	mask[np.triu_indices(len(mask))] = True
	sns.heatmap(correlation, vmin = -1, vmax = 1, center = 0, cmap = 'coolwarm', annot = False, mask = mask)
	for label in labels:
		sns.pairplot(df, x_vars=features, y_vars=label)
	if correlations:
		plt.show()
	
	# Relevant features, following initial correlation studies
	irrel_features = ['Scale_peak_mass', 'Scale_half_mass', 'Concentration', 'Halo_spin', 'Intra_cluster_mass', 
			'X_pos', 'Y_pos', 'Z_pos', 'X_vel', 'Y_vel', 'Z_vel', 'Type']

	# Make separate dataframes for relevant features and labels
	df_X = df.copy()
	df_X.drop(labels, inplace=True, axis=1)
	df_X.drop(irrel_features, inplace=True, axis=1)
	df_y = df.copy()
	df_y.drop(features, inplace=True, axis=1)
	
	# Use the df_scaler class to perform a scaling (standard is the default)
	scale = dfs.df_scaler()
	df_X_scaled = scale.fit_transform(df_X)
	
	print df_X.head(10)
	print '_'*40
	print df_X_scaled.head(10)

	# Let's also apply some dimensionality reduction with PCA
	pca = PCA(n_components = 2)
	df_X_scaled = pca.fit_transform(df_X_scaled)
	print df_X_scaled
	print '_'*40	

	# Make the separate training, validation, and testing sets
	X_train, X_test, y_train, y_test = train_test_split(df_X_scaled, df_y, test_size=0.3, random_state=1)
 	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=1)

	'''
		Before any hyperparameter tuning let's load a pipeline of a few models and 
		investigate their out-of-the-box performance
	'''
	
	models = [
	  ('nn MLP', MLPRegressor(hidden_layer_sizes=(50, ), activation='relu', solver='adam', alpha=0.001)),
          ('forest', RandomForestRegressor(n_estimators=200)),
          ('kneig', KNeighborsRegressor()),
	  ('lasso', Lasso(alpha=0.01)), 
	  ('ridge', Ridge()),  
          ('xtree', ExtraTreesRegressor(n_estimators=200))
	]

	mod_name = []
	rmse_validate = []
	mae_validate = []
	r2_score_validate = []

	for model in models:
    		train = X_train.copy()
    		val = X_val.copy()
    		print model[0]
    		mod_name.append(model[0])
    		model_pipe = Pipeline([model])  
    		print 'X shape: ', train.shape, ' ... ', 'y shape: ', y_train.shape
		model_pipe.fit(train, y_train)  
    		trainPreds = model_pipe.predict(train) 
		preds = model_pipe.predict(val) # Predict using the validation set
    		
		if mod_name == 'forest': print 'The random forest feature importances are: ', model_pipe.feature_importances_
    		rmse_validate.append(mean_squared_error(y_val, preds))
		mae_validate.append(mean_absolute_error(y_val, preds))
		r2_score_validate.append(r2_score(y_val, preds))    

    		print 'Training set RMSE: {0}'.format(round(sqrt(mean_squared_error(y_train, trainPreds)), 4))
    		print 'Training set MAE: {0}'.format(round(mean_absolute_error(y_train, trainPreds), 2))
		print 'Training set r2 score: {0}'.format(round(r2_score(y_train, trainPreds), 2))    		
    		print 'Validation set RMSE: {0}'.format(round(sqrt(mean_squared_error(y_val, preds)), 4))
    		print 'Validation set MAE: {0}'.format(round(mean_absolute_error(y_val, preds), 2))
		print 'Validation set r2 score: {0}'.format(round(r2_score(y_val, preds), 2))    		

    		print '_'*40
    		print '\n'
    
		results = pd.DataFrame({'model_name': mod_name, 
                        'rmse_validate': rmse_validate,
                        'mae_validate': mae_validate})	
	

	'''
		Perform hyperparameter tuning using a grid search, model-by-model
	'''

	# Make a dictionary of some relevant parameters for the models in the grid search
	grid_search_params = { 
				'lasso': {
					'alpha': [0.002, 0.001, 0.0009, 0.0008], 
                			'tol': [0.005, 0.001, 0.0005, 0.0001]
				},
				'ridge': {
					'alpha': [2, 1.7, 1.5, 1.3, 1, 0.9], 
                			'tol': [0.005, 0.001, 0.0005, 0.0001]
				},
				'forest': {
					'max_depth': [10, 20, 30, None],
                 			'max_features': ['auto', 'sqrt', 'log2'], 
                 			'min_samples_leaf': [1, 3, 5, 10],
                 			'min_samples_split': [2, 4, 6, 8]
				},
				'xtree': {
					'max_depth': [10, 20, 30, None],
                 			'max_features': ['auto', 'sqrt', 'log2'], 
                 			'min_samples_leaf': [1, 3, 5, 10],
                 			'min_samples_split': [2, 4, 6, 8],
		 			'n_estimators': [100, 200, 300, 400, 500]
				},
				'knn': {
					'n_neighbors': [2, 5, 7, 10, 15],
		 			'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
		 			'leaf_size': [5, 10, 20, 30, 40, 50]
				},
				'MLP': {
					'alpha': [0.001, 0.01, 0.1],
                 			'hidden_layer_sizes': [50, 100, 200, 300, 400], 
                 			'activation': ['identity', 'logistic', 'tanh', 'relu'],
                 			'solver': ['lbfgs', 'sgd', 'adam'],
		 			'learning_rate': ['constant', 'invscaling', 'adaptive']
				}
	}

	# Define a number of K-folds to use in the training
	folds = KFold(5, shuffle=True, random_state=541)
	
	gs_rmse_validate = []
	gs_mae_validate = []
	gs_r2_score_validate = []
	
	mod_index = 0
	for params in sorted(grid_search_params.keys()):
		print grid_search_params[params]
		model = models[mod_index][1]
		result = GridSearchCV(model, param_grid=grid_search_params[params], cv=folds, scoring='neg_mean_squared_error')
		result.fit(X_train, y_train)
		best = result.best_estimator_
		print 'Best estimator for ', model, ': ', best
		print 'Train using best estimator'
		print '-'*40
    		trainPreds = best.predict(X_train) # Now we use the best estimator for each model from the grid search 
		preds = best.predict(X_val) # Predict using the validation set
    
    		gs_rmse_validate.append(mean_squared_error(y_val, preds))
		gs_mae_validate.append(mean_absolute_error(y_val, preds))
		gs_r2_score_validate.append(r2_score(y_val, preds))    

    		print 'Training set RMSE: {0}'.format(round(sqrt(mean_squared_error(y_train, trainPreds)), 4))
    		print 'Training set MAE: {0}'.format(round(mean_absolute_error(y_train, trainPreds), 2))
		print 'Training set r2 score: {0}'.format(round(r2_score(y_train, trainPreds), 2))    		
    		print 'Validation set RMSE: {0}'.format(round(sqrt(mean_squared_error(y_val, preds)), 4))
    		print 'Validation set MAE: {0}'.format(round(mean_absolute_error(y_val, preds), 2))
		print 'Validation set r2 score: {0}'.format(round(r2_score(y_val, preds), 2))    		
		
		mod_index += 1

def cv_score(df_train, y_train, kfolds, pipeline, imp_coef=False):
    oof = np.zeros(len(df_train))
    train = df_train.copy()
    
    feat_df = pd.DataFrame()
    
    for n_fold, (train_index, test_index) in enumerate(kfolds.split(train.values)):
            
        trn_data = train.iloc[train_index][:]
        val_data = train.iloc[test_index][:]
        
        trn_target = y_train.iloc[train_index].values.ravel()
        val_target = y_train.iloc[test_index].values.ravel()
        
        pipeline.fit(trn_data, trn_target)

        oof[test_index] = pipeline.predict(val_data).ravel()

        if imp_coef:
            try:
                fold_df = get_coef(pipeline)
            except AttributeError:
                fold_df = get_feature_importance(pipeline)
                
            fold_df['fold'] = n_fold + 1
            feat_df = pd.concat([feat_df, fold_df], axis=0)
       
    if imp_coef:
        feat_df = feat_df.groupby('feat')['score'].agg(['mean', 'std'])
        feat_df['abs_sco'] = (abs(feat_df['mean']))
        feat_df = feat_df.sort_values(by=['abs_sco'],ascending=False)
        del feat_df['abs_sco']
        return oof, feat_df
    else:    
        return oof

if __name__ == "__main__":
	from argparse import ArgumentParser
    	parser = ArgumentParser()
    	parser.add_argument("-i", "--inputData", help="Path to the input data.",default="galaxies.Z80.h5")
	parser.add_argument("-o", "--outputPath", help="Path to the output directory.",default="./Plots/") 
    	parser.add_argument("-c", "--correlations", help="", action="store_true", default=False)
	options = parser.parse_args()

    	# Input and output directories
    	inData = options.inputData
    	outDir = options.outputPath
    	if not os.path.exists(outDir):
        	os.makedirs(outDir)
           	print "The output directory did not exist, I have just created one: ", outDir   


    	# Defining dictionary to be passed to the main function
    	option_dict = dict( (k, v) for k, v in vars(options).iteritems() if v is not None)
    	print option_dict
    	main(**option_dict)

