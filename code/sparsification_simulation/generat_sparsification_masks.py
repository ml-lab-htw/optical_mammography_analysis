from pathlib import Path
import logging
import numpy as np
import pickle

homedir=str(Path.cwd().parents[0]) 
logging.basicConfig(filename=f'{homedir}/logs/debuging.log', level=logging.WARNING, format='%(asctime)s %(levelname)s %(name)s %(message)s')

class sparsification_masks():
    """ generats the pruning mask for the preset spars configurations

    """
    def __init__(self):
        test_det=np.array(range(0,64))
        test_sen=np.array(range(0,64)[::2])

        dm_32_1_det=np.array([4,2,8,6,12,10,16,14,20,18,24,22,28,30,26,32,36,34,40,38,44,42,48,46,52,50,56,54,60,58,64,62])-1
        dm_32_1_sen=np.array([1,5,3,9,7,15,13,11,23,21,19,17,25,27,29,31,33,35,39,37,41,43,47,45,51,49,55,53,59,57,61,63])-1

        dm_32_2_det=np.array([1,5,3,9,7,15,13,11,23,21,19,17,25,27,29,31,33,35,39,37,41,43,47,45,51,49,55,53,59,57,61,63])-1
        dm_32_2_sen=np.array([1,5,3,9,7,15,13,11,23,21,19,17,25,27,29,31,33,35,39,37,41,43,47,45,51,49,55,53,59,57,61,63])-1

        dm_16_1_det=np.array([2,12,8,16,20,24,28,32,40,38,48,50,46,52,58,60])-1
        dm_16_1_sen=np.array([1,3,13,19,27,31,7,23,33,35,41,43,53,55,61,63])-1


        dm_16_2_det=np.array([4,12,20,28,26,8,16,24,34,36,44,42,54,56,62,64])-1
        dm_16_2_sen=np.array([9,11,17,29,5,15,21,25,39,37,47,45,51,49,59,57])-1

        dm_16_3_det=np.array([2,6,14,22,32,7,23,35,37,43,45,51,53,59,61,18])-1
        dm_16_3_sen=np.array([1,9,29,31,13,19,5,25,33,39,41,47,49,55,57,63])-1


        dm_8_1_det=np.array([5,7,23,25,36,44,56,64])-1
        dm_8_1_sen=np.array([1,11,17,31,33,41,53,61])-1

        dm_8_2_det=np.array([11,17,5,25,35,41,55,63])-1
        dm_8_2_sen=np.array([9,29,7,23,33,43,53,61])-1
        
        dm_8_3_det=np.array([1,13,19,31,37,57,44,56])-1
        dm_8_3_sen=np.array([5,7,23,25,35,45,49,63])-1

        dm_8_4_det=np.array([1,11,17,31,40,48,52,60])-1
        dm_8_4_sen=np.array([3,15,21,27,35,41,53,63])-1

        dm_4_1_det=np.array([15,21,63,35])-1
        dm_4_1_sen=np.array([3,27,43,55])-1

        dm_4_2_det=np.array([5,25,44,56])-1
        dm_4_2_sen=np.array([13,19,57,37])-1

        dm_4_3_det=np.array([1,31,42,54])-1
        dm_4_3_sen=np.array([13,19,35,63])-1

        dm_4_4_det=np.array([8,24,40,60])-1
        dm_4_4_sen=np.array([1,31,41,53])-1
        
        self.detectors=[test_det,dm_32_1_det,dm_32_2_det,dm_16_1_det,dm_16_2_det,dm_16_3_det,dm_8_1_det,dm_8_2_det,dm_8_3_det,dm_8_4_det,dm_4_1_det,dm_4_2_det,dm_4_3_det,dm_4_4_det]
        self.emitters=[test_sen,dm_32_1_sen,dm_32_2_sen,dm_16_1_sen,dm_16_2_sen,dm_16_3_sen,dm_8_1_sen,dm_8_2_sen,dm_8_3_sen,dm_8_4_sen,dm_4_1_sen,dm_4_2_sen,dm_4_3_sen,dm_4_4_sen]

    def pickel_dm(self,side):
        """ creats the 14 pruning masks as pickl files and saves them 
            
            Paramaeters: 
                side -- marks the breast side in the file name 
                
        """
        logger=logging.getLogger(f"g_s_masks pickel_dm()")
        try:
            detector_NAMES=["test_det","dm_32_1_det","dm_32_2_det","dm_16_1_det","dm_16_2_det","dm_16_3_det","dm_8_1_det","dm_8_2_det","dm_8_3_det","dm_8_4_det","dm_4_1_det","dm_4_2_det","dm_4_3_det","dm_4_4_det"]
            i=0            
            for detectors_to_keep,emitters_to_keep,dnames in zip(self.detectors,self.emitters,detector_NAMES): 
                det=sorted(detectors_to_keep)
                list_emi=sorted(emitters_to_keep)
                emi =[]
                emi_map=[1,0,2,0,3,0,4,0,5,0,6,0,7,0,8,0,9,0,10,0,11,0,12,0,13,0,14,0,15,0,16,0,17,0,18,0,19,0,20,0,21,0,22,0,23,0,24,0,25,0,26,0,27,0,28,0,29,0,30,0,31,0,32,0]
                for index in list_emi:
                    emi.append(emi_map[index]-1)
                temp=[]
                for os in np.round(np.linspace(0,2048,32,endpoint=False).astype(int)):
                    temp.append(os+det)
                det_total=np.array(temp).astype(int).flatten()
                dm_det = np.setdiff1d(np.arange(2048), det_total) 
                zws=[]
                for emitter in emi: 
                    row=[]
                    for os in range(64):
                        row.append(emitter*64+os)
                    zws.append(row)
                emi_total=np.array(zws).astype(int).flatten()
                dm_emi = np.setdiff1d(np.arange(2048), emi_total) 
                combined_array = np.concatenate((dm_emi, dm_det))
                indices_to_delete = np.unique(combined_array)
                with open(f"./masks/del_mask_{side}_{len(emitters_to_keep)}x{len(detectors_to_keep)}_{i}.pickle","wb") as f:
                    pickle.dump(indices_to_delete,f)
                i+=1
        except Exception as err:
            logger.error(err)



