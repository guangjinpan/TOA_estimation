import scipy.io as scio
import data_pro
import numpy as np
import os
from glob import glob

BSN=7
def load_data():
    OTDOA_DATA =data_pro.get_OTADA_DATA_X(BSN)
    MS =data_pro.get_MS()
    return OTDOA_DATA,MS
load_data()
