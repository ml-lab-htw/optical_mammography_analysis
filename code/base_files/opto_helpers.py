from itertools import combinations
from tqdm import tqdm
from pathlib import Path
import logging
import pickle

homedir=str(Path( __file__).parents[1])  

def process_file_names(name):
    logger=logging.getLogger(f"helpers process_file_names()")
    try:
        pato = name.split("-")[0]
        # 1 = tumor left or both; -1 = tumor right; if healthy still has val 1
        if pato == "left":
            return 1
        elif pato == "right":
            return -1 
        elif pato == "non":
            return 0
        else: 
            print("Maybe there is a non valid file in the folder")
    except Exception as err:
        logger.error(err)

def loadall(filename):
    """Iterativly loads lines of pickel file. 

        Paramaeters: 
           filename -- name of the pickel file to load
        Yield:
            line from pickel file
    """
    logger=logging.getLogger(f"helpers loadall()")
    try:
        with open(filename, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break
    except Exception as err:
        logger.error(err)

