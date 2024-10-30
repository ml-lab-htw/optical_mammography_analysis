from sparsification_simulation import generat_sparsification_masks as gsm
from classification_moduls import classification as cl
from tqdm import tqdm
from pathlib import Path
import logging
import os
import re
import csv

homedir=str(Path( __file__).parents[1])  
logging.basicConfig(filename=f'{homedir}/logs/debuging.log', level=logging.WARNING, format='%(asctime)s %(levelname)s %(name)s %(message)s')


class sparsification_simulation():
    """ runns the spars set up simulation

    """
    def __init__(self):
        self.path=f"{homedir}/sparsification_simulation/masks"

    def prep_masks(self):
        """ Checks if pruning masks exist, if not generates them 

        """
        logger=logging.getLogger(f"spars_simulation prep_masks()")
        try:
            if not os.path.exists(self.path):
                os.mkdir(self.path)
                mask_generator = gsm.sparsification_masks()
                for s in ["left" ,"right"]:
                    mask_generator.pickel_dm(s)
        except Exception as err:
            logger.error(err)

    def sort_key(self,filename):
        """ provides a sort_key for sorting the filenames in to a specific format for correct loading
            
            Paramaeters: 
                filename -- filename that shall be sorted
                
        """
        logger=logging.getLogger(f"g_s_masks pickel_dm()")
        try:
            match = re.match(r"del_mask_(left|right)_(\d+)x\d+_(\d+).pickle", filename)
            if match:
                side = 0 if match.group(1) == 'right' else 1
                first_number_of_size = int(match.group(2))
                index = int(match.group(3))
                return (side, first_number_of_size, index)
            return filename
        except Exception as err:
            logger.error(err)

    def prep_path_list(self):
        """ generates a list of the filenames of all pruning masks 
            
            Reurns: 
                paths(list): sorted list of paths of all pruning masks  
                
        """
        logger=logging.getLogger(f"g_s_masks pickel_dm()")
        try:
            onlyfiles = [f for f in os.listdir(self.path) if 
            os.path.isfile(os.path.join(self.path, f))]
            paths=[]
            files=sorted(onlyfiles,key=self.sort_key,reverse=True)
            for id,e in enumerate(files):
                if e.split("_")[2] == "right":
                    continue
                else: 
                    paths.append([e,files[14+id]])
            return paths    
        except Exception as err:
            logger.error(err)

    def sparsification(self,mode="bi"):
        """ simulats a spars set up  
            
            Paramaeters: 
                mode -- defirientiats betwen bilateral (bi) and unilateral (uni) simulation
                
        """
        logger=logging.getLogger(f"g_s_masks pickel_dm()")
        try:
            classifier=cl.classificiation(mode)
            with open(f'res_{mode}.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                for file in tqdm(self.prep_path_list()): 
                    path_left= f"{self.path}/{file[0]}"
                    path_right= f"{self.path}/{file[1]}"
                    X,y,X_eval,y_eval = classifier.load_data(pruning=True,ppath_left=path_left,ppath_right=path_right)
                    test_score=classifier.classify(X,y,X_eval,y_eval)
                    writer.writerow([file[0].split("_")[3]]+[test_score]) 
        except Exception as err:
            logger.error(err)
            
# script can be used standalone in the consol to test set mode to "uni" or "bi"
if __name__ == "__main__":
    sim=sparsification_simulation()
    sim.prep_masks()
    sim.sparsification("uni")
