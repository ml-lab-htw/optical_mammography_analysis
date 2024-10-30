from base_files import opto_io as oio
from base_files import opto_dataset_creation as odc
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, f_classif
from pathlib import Path
import logging
import sklearn.metrics as metrics

homedir=str(Path( __file__).parents[1]) 
logging.basicConfig(filename=f'{homedir}/logs/debuging.log', level=logging.WARNING, format='%(asctime)s %(levelname)s %(name)s %(message)s')


class classificiation (): 
    """ classification with SVM

        Paramaeters: 
           mode -- determens if bilateral (bi) or unilateral (uni) should be used 
    """
    def __init__(self, mode):
        if mode == "bi":
            self.mode=mode
            self.wavelength = ('1', '2')
            self.normalization = ('relative_change', 'zero_centering')
            self.feature_representation = 'op'
            self.n_feature_selection = 3
            self.c = 0.001
        elif mode == "uni":
            self.mode=mode
            self.wavelength = ('1',)
            self.normalization = ('relative_change', 'robust')
            self.feature_representation = 'sf'
            self.c = 599.4842503189409 # Found via gridsearch cv and log scaled list of values 
        else: 
            print("please use valid mode")

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
            if self.mode=="bi":
                if pruning:
                    data,tumor_map,original_names = oio.get_data(preprocessing=False,pruning=pruning,pruning_path_left=ppath_left,pruning_path_right=ppath_right)
                else:
                    data,tumor_map,original_names = oio.get_data(preprocessing=False)
                names , eval_names = odc.gen_eval_split(tumor_map,original_names)
                X_eval,y_eval=odc.getDistanceDataMatrix(data, eval_names, tumor_map, wave_length=self.wavelength,norm_methods=self.normalization,preprocessing=self.feature_representation,preprocess=True)
                X,y = odc.getDistanceDataMatrix(data, names , tumor_map, wave_length=self.wavelength,norm_methods=self.normalization,preprocessing=self.feature_representation,preprocess=True)
                return X,y,X_eval,y_eval
            elif self.mode=="uni":
                if pruning:
                    data,tumor_map,original_names = oio.get_data(preprocessing=True,six_features=True,pruning=pruning,pruning_path_left=ppath_left,pruning_path_right=ppath_right)
                else:
                    data,tumor_map,original_names = oio.get_data(preprocessing=True,six_features=True)
                names , eval_names = odc.gen_eval_split(tumor_map,original_names)
                X_eval,y_eval= odc.getBreastBasedMatrix(data, eval_names, tumor_map, wave_length=self.wavelength,norm_methods=self.normalization,preprocessing=self.feature_representation)
                X,y = odc.getBreastBasedMatrix(data, names , tumor_map, wave_length=self.wavelength,norm_methods=self.normalization,preprocessing=self.feature_representation)
                return X,y,X_eval,y_eval
        except Exception as err:
            logger.error(err)

        
    def classify(self,X,y,X_eval,y_eval):
        """ loads the data with optional pruning

            Paramaeters: 
                X -- The data matrix of the model selection split
                y -- The Validation list of the model selection split
                X_eval -- The data matrix of the testing split
                y_eval -- The Validation list of the testing split
            
            Returns:
                roc_auc(float): the auc score 
        """

        logger=logging.getLogger(f"classification classify()")
        try:
            if self.mode == "bi":
                X = SelectKBest(f_classif,k=self.n_feature_selection).fit_transform(X,y)
                X_eval = SelectKBest(f_classif,k=self.n_feature_selection).fit_transform(X_eval,y_eval)
            clf=LinearSVC(C=self.c,dual=False)
            clf.fit(X,y)
            pred = clf.decision_function(X_eval)
            fpr, tpr, thresholds = metrics.roc_curve(y_eval, pred, pos_label=1,drop_intermediate=False)
            roc_auc = metrics.auc(fpr, tpr)
            return(roc_auc)
        except Exception as err:
            logger.error(err)

# script can be used standalone in the consol to test set mode to "uni" or "bi"
if __name__ == "__main__":
    mode="bi"
    classifier = classificiation(mode)
    X,y,X_eval,y_eval = classifier.load_data()
    print(classifier.classify(X,y,X_eval,y_eval))

