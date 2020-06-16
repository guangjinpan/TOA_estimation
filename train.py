import init_data
import model_art
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam,SGD,RMSprop,Adadelta,Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import tensorflow as tf
import argparse
import os
import sys


from keras import backend as K
import numpy as np
FLAGS = None

def smooth_L1_loss(y_true, y_pred):
    THRESHOLD = K.variable(1.0)
    mae = K.abs(y_true-y_pred)
    flag = K.greater(mae, THRESHOLD)
    loss = K.mean(K.switch(flag, (mae - 0.5), K.pow(mae, 2)), axis=-1)
    return loss


def mse1(y_true, y_pred):
    THRESHOLD = K.variable(0.005)
    mse =  K.square((y_true)-(y_pred))
    flag = K.greater(mse, THRESHOLD)
    loss = K.mean(K.switch(flag, mse-THRESHOLD,  K.square((y_true)-(y_true))), axis=-1)
    return loss


def accuracy1(y_true, y_pred):
    acc = K.mean(K.square(K.round(y_true)-K.round(y_pred)),axis=-1)
    return acc



def main(_):
    BSN=7
    learn_rate = 0.005
    batchsize = 256
    epoch = 1000
    opt = Adam(lr=learn_rate)#, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False    
    opt1 = SGD(lr=learn_rate)#,decay=learn_rate / epoch)
    opt2=RMSprop(lr=learn_rate, rho=0.9, epsilon=None, decay=0.0)
    opt3=Adadelta(lr=learn_rate, rho=0.95, epsilon=None, decay=0.0)
    opt4=Nadam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.00004)

    trainX,trainY = init_data.load_data()
    print(np.shape(trainX))
    input = Input(shape=(14,30,1))
    #input = Input(shape=(6,5,60,1))
    predict = model_art.pred2_x(input)
    model = Model(inputs=input, outputs=predict)
    model.compile(optimizer=opt, loss=mse1, metrics=['accuracy',accuracy1])
    #model.load_weights('savemodel/64train1+1000.hdf5')
    model.summary()
    checkpoint = ModelCheckpoint('savemodel/xxx+{epoch:02d}.hdf5',monitor='val_loss',verbose=1,
                                 save_weights_only=True, save_best_only=False, period=50)
    # checkpoint = ModelCheckpoint('savemodel/cnnt+{epoch:02d}.hdf5',monitor='val_loss',verbose=1,
    #                              save_weights_only=True, save_best_only=False, period=10)
    earlystop = EarlyStopping(patience=10, verbose=1)
    tensorboard = TensorBoard(write_graph=True)

    res = model.fit(trainX, trainY, steps_per_epoch=10000//5*4 //batchsize,epochs =epoch,callbacks=[checkpoint],validation_split=0.2,validation_steps=10000//5 //batchsize)
    loss_h=res.history['loss']
    np_loss=np.array(loss_h)
    np.savetxt('txt220.txt',np_loss)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu',
        type=int,
        default=1,
        help='which gpu to use')
    FLAGS, unparsed = parser.parse_known_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    set_session(tf.InteractiveSession(config=config))

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
