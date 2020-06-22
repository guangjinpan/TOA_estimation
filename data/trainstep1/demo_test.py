from keras.layers import Input
from keras.models import Model
import model_art
import scipy.io as scio
import numpy as np
import pandas as pd
import os
import time
from glob import glob


MS_path='./test2X/output_EVA-5Hz.mat'
OTDOA_DATA_path='./test2X/input_EVA-5Hz.mat'


os.environ['CUDA_VISIBLE_DEVICES']='2'
start = time.clock()
BSN=7
NUE=1400
def get_OTADA_DATA(BSN):
    file_mame_t1=OTDOA_DATA_path

    t=[]
    t=scio.loadmat(file_mame_t1)
    OTADA_DATA=[]
    OTADA_DATA.extend(t['input'])
    OTADA_DATA=np.array(OTADA_DATA)
    print(np.shape(OTADA_DATA))
    OTADA_DATA=OTADA_DATA[0:NUE,:,0:30]
    OTADA_DATA = np.reshape(OTADA_DATA, (NUE,14,30,1))
    #OTADA_DATA = np.expand_dims(OTADA_DATA,axis=1)

    #print(OTADA_DATA[1])
    return OTADA_DATA

def get_MS( ):
    file_mame_MS=MS_path
    t=[]
    t=scio.loadmat(file_mame_MS)
    #print(t)
    MS=[]
    MS.extend(t['output'])
    MS =np.array(MS)
    MS=np.reshape(MS,(NUE,1))
    print(MS.shape)
    print(MS)
    return MS

OTDOA_DATA = get_OTADA_DATA(BSN)
MS = get_MS()
print(3333)
input = Input(shape=(14, 30,1))
predict = model_art.pred2(input)
model = Model(inputs=input, outputs=predict)
model.load_weights('./savemodel/64train2-EVA5HZ+1000.hdf5')   #DNN4 *12*60
result = []
count = 0
print(3333)
for i in range(NUE):
    j = np.expand_dims(OTDOA_DATA[i],axis=0)
    result.append(model.predict(j))
result = np.array(result)
result = np.reshape(result, (NUE, 1))
result=result
print (result.shape)
MSx = [x[0] for x in MS]
resultx = [x[0] for x in result]


end = time.clock()
print(end-start)
df = pd.DataFrame(np.vstack([MSx, resultx]).T)
                  #columns=['MSx', 'MSy', 'resultx', 'resulty', 'error_distance'])

df.to_csv('./64eva51000X.csv', index=False)

