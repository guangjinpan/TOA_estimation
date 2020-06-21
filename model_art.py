from keras.layers import Dense, Flatten, Conv2D,Conv1D, Conv3D, BatchNormalization, Activation,MaxPooling2D, Add, Reshape
from keras.layers import Input, Dropout
from tensorflow.keras import Model
def pred(input):


    x = Conv2D(2,kernel_size=(3,3),strides=(2,2),padding='same',name='conv1')(input)
    x = Activation('relu')(x)
    x = Conv2D(2,kernel_size=(3,3),strides=(2,2),padding='same',name='conv2')(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    #x = Dense(1024, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    #x = Dense(8, activation='relu')(x)
    #x = Dense(8, activation='softmax')(x)
    #x = Dropout(0.5)(x)
    #x = Dense(8, activation='relu')(x)
    x = Dense(1, activation='relu')(x)
    #x = Activation('softmax')(x)
    return x






def pred_step1(input):


    x = Conv2D(2,kernel_size=(3,3),strides=(2,2),padding='same',name='conv1')(input)
    x = Activation('relu')(x)
    x = Conv2D(2,kernel_size=(3,3),strides=(2,2),padding='same',name='conv2')(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    x = Dense(1, activation='relu')(x)

    return x

def pred_freezen(input):


    x = Conv2D(2,kernel_size=(3,3),strides=(2,2),padding='same',name='conv1',trainable=False)(input)
    x = Activation('relu')(x)
    x = Conv2D(2,kernel_size=(3,3),strides=(2,2),padding='same',name='conv2',trainable=False)(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    x = Dense(1, activation='linear')(x)
    return x

