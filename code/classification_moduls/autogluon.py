from base_files import opto_io as oio
from base_files import opto_dataset_creation as odc
from autogluon.tabular import TabularPredictor
from pathlib import Path
import logging
import sklearn.metrics as metrics

homedir=str(Path( __file__).parents[1]) 
logging.basicConfig(filename=f'{homedir}/logs/debuging.log', level=logging.WARNING, format='%(asctime)s %(levelname)s %(name)s %(message)s')


class autogluon_classification():
    """ classification with AutoGluon
    """
    def __init__(self):
        self.wavelength = ('2','1')
        self.normalization = ('relative_change', 'zscore')
        self.feature_representation = 'op'
        self.path = "./AutogluonModels"
        self.hyperparameters = {
        'NN_TORCH': {},
        'GBM': [{'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}}, {}, 'GBMLarge'],
        'CAT': {},
        'XGB': {},
        'FASTAI': {},
        'RF': [{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}],
        'XT': [{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}],
        'KNN': [{'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}}, {'weights': 'distance', 'ag_args': {'name_suffix': 'Dist'}}],
        "LR":{'penalty': 'L2',},
        }

    def load_data(self,pruning=False,ppath_left="",ppath_right=""):
        """ loads the data with optional pruning

            Paramaeters: 
                pruning -- if set True apllys pruning with pruning masks.
                ppath_left -- sets the path for the pruning mask of the left breast.
                ppath_right -- sets the path for the pruning mask of the right breast.
            
            Returns:
                X(np.array): The data matrix of the model selection split
                y(np.array): The Validation list of the model selection split
                X_eval(np.array): The data matrix of the testing split
                y_eval(np.array): The Validation list of the testing split
        """

        logger=logging.getLogger(f"classification load_data()")
        try:

            if pruning:
                data,tumor_map,original_names = oio.get_data(allPatients=True, preprocessing=False,pruning=pruning,pruning_path_left=ppath_left,pruning_path_right=ppath_right)
            else:
                data,tumor_map,original_names = oio.get_data(allPatients=True, preprocessing=False)
            names , eval_names = odc.gen_eval_split(tumor_map,original_names)
            X_eval,y_eval=odc.getDistanceDataMatrix(data, eval_names, tumor_map, wave_length=self.wavelength,norm_methods=self.normalization,preprocessing=self.feature_representation,preprocess=True)
            X,y = odc.getDistanceDataMatrix(data, names , tumor_map, wave_length=self.wavelength,norm_methods=self.normalization,preprocessing=self.feature_representation,preprocess=True)
            return X,y,X_eval,y_eval
        except Exception as err:
            logger.error(err)
            
    def get_roc_autog(self,df,dfy):
        """ calculates the AUC score for the best model autogluon trained 

            Paramaeters: 
                df -- The Data Table including lable colum
                dfy -- The Validation list 
            
            Returns:
                fpr(list): Increasing false positive rates
                tpr(list): Increasing true positive rates
                roc_auc(float): the AUC score
                thresholds(list): Decreasing thresholds on the decision function
        """

        logger=logging.getLogger(f"autogluon get_roc_autog()")
        try:
            predictor = TabularPredictor.load(self.path)
            name = predictor.get_model_best()
            pred = predictor.predict_proba(df, name)
            print(len(pred[1]))
            fpr, tpr, thresholds = metrics.roc_curve(dfy, pred[1],drop_intermediate=False)
            roc_auc = metrics.auc(fpr, tpr) 
            return fpr,tpr,roc_auc,thresholds
        except Exception as err:
            logger.error(err)

    def train(self,X,y):
        """ trains the Autogluon models

            Paramaeters: 
                X -- List of the data matrixs from the model selection split 
                y -- Validation list from the model selection split 
        """

        logger=logging.getLogger(f"autogluon train()")
        try:
            df,dfy=odc.prep_data_autog(X,y)
            TabularPredictor(label="y",eval_metric="roc_auc",path=self.path,).fit(df,hyperparameters=self.hyperparameters)
        except Exception as err:
            logger.error(err)

    def eval(self,X_eval,y_eval):
        """ evaluats the Autogluon models
            Paramaeters: 
                X_eval -- List of the data matrixs from the testing split 
                y_eval -- Validation list from the testing split 
        """

        logger=logging.getLogger(f"autogluon eval()")
        try:
            df_eval,dfy_eval=odc.prep_data_autog(X_eval,y_eval)
            fpr,tpr,roc_auc,thresholds=self.get_roc_autog(df_eval,dfy_eval)
            return(roc_auc)
        except Exception as err:
            logger.error(err)

# script can be used standalone in the consol to test 
if __name__ == "__main__":
    classifier = autogluon_classification()
    X,y,X_eval,y_eval = classifier.load_data()
    classifier.train(X,y)
    print(classifier.eval(X_eval,y_eval))

