import scipy.io as scio
import numpy as np
import os
from glob import glob


#time_path='./data'
#MS_path='./data/data/MS_N_4_30.mat'
#OTDOA_DATA_path='./data/data/OTDOA_DATA_N_4_30.mat'

#time_path='./data'
#MS_path='./train_step2AWGN3/output_2_10.mat'
#OTDOA_DATA_path='./train_step2AWGN3/input2_2_10.mat'
MS_path='./train_step1/output.mat'
OTDOA_DATA_path='./train_step1/input.mat'


#time_path='./data'
#MS_path='./EVA5_4_25/aug1/distance_1.mat'
#OTDOA_DATA_path='./EVA5_4_25/aug1/signal_all_ri_phase_1.mat'
#OTDOA_DATA_path='./EVA5_1km_15/signal_AWGN_all_abs_1.mat'

def get_OTADA_DATA(BSN):
    file_mame_t1=OTDOA_DATA_path

    t=[]
    t=scio.loadmat(file_mame_t1)
    OTADA_DATA=[]
    OTADA_DATA.extend(t['corr'])
    OTADA_DATA=np.array(OTADA_DATA)
    OTADA_DATA = np.reshape(OTADA_DATA, (84000,BSN-1))
    OTADA_DATA = np.expand_dims(OTADA_DATA,axis=1)

    #OTADA_DATA=(OTADA_DATA-OTADA_DATA.min())/(OTADA_DATA.max()-OTADA_DATA.min())
    return OTADA_DATA


def get_OTADA_DATA_X(BSN):
    file_mame_t1=OTDOA_DATA_path

    t=[]
    t=scio.loadmat(file_mame_t1)
    #print(t)
    OTADA_DATA=[]
    OTADA_DATA.extend(t['input'])
    OTADA_DATA=np.array(OTADA_DATA)
    print(OTADA_DATA.shape)
    OTADA_DATA = np.reshape(OTADA_DATA, (80000,14,60,1))
    #OTADA_DATA = np.expand_dims(OTADA_DATA,axis=1)
    OTADA_DATA=OTADA_DATA[0:80000,:,0:30,:]
    # print(np.shape(OTADA_DATA))
    # OTADA_DATA=np.reshape(OTADA_DATA,[10000,120,3])
    return OTADA_DATA

def get_MS( ):
    file_mame_MS=MS_path
    t=[]
    t=scio.loadmat(file_mame_MS)
    #print(t)
    MS=[]
    MS.extend(t['output'])
    MS =np.array(MS)
    MS=np.reshape(MS,(80000,1))
    MS=MS[0:80000,:]
    #MS=np.append(MS,MS,axis=0)
    #MS=MS-156.25
    #MS = (MS - MS.min()) / (MS.max() - MS.min())
    return MS

