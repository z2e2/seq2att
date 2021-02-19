import tensorflow as tf
import keras
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv1D, TimeDistributed, CuDNNLSTM, LSTM
from keras.layers import Dense, Dot, Dropout, Bidirectional
from keras.layers import Activation, BatchNormalization, Add, Flatten
from keras.optimizers import Adam

## ===== to be removed ===== ##
np.random.seed(1)
tf.set_random_seed(2)
## ===== to be removed ===== ##

def conv_net_block(X, n_cnn_filters=256, cnn_window=9, block_name='convblock_0'):
    '''
    convolutional block with a 1D convolutional layer, a batch norm layer followed by a relu activation.
    parameters:
    n_cnn_filters: number of output channels
    cnn_window: window size of the 1D convolutional layer
    '''
    X = Conv1D(n_cnn_filters, cnn_window, strides=1, padding='same', name='{}_cnv0'.format(block_name))(X)
    X = BatchNormalization(axis=-1, name='{}_bn0'.format(block_name))(X)
    X = Activation('relu', name='{}_a0'.format(block_name))(X)
    return X

def res_net_block(X, n_cnn_filters=256, cnn_window=9, block_name='resblock_0'):
    '''
    residual net block accomplished by a few convolutional blocks.
    parameters:
    n_cnn_filters: number of output channels
    cnn_window: window size of the 1D convolutional layer
    '''
    X_identity = X
    # cnn0
    X = Conv1D(n_cnn_filters, cnn_window, strides=1, padding='same', name='{}_cnv0'.format(block_name))(X)
    X = BatchNormalization(axis=-1, name='{}_bn0'.format(block_name))(X)
    X = Activation('relu', name='{}_a0'.format(block_name))(X)
    # cnn1
    X = Conv1D(n_cnn_filters, cnn_window, strides=1, padding='same', name='{}_cnv1'.format(block_name))(X)
    X = BatchNormalization(axis=-1, name='{}_bn1'.format(block_name))(X)
    X = Activation('relu', name='{}_a1'.format(block_name))(X)
    # cnn2
    X = Conv1D(n_cnn_filters, cnn_window, strides=1, padding='same', name='{}_cnv2'.format(block_name))(X)
    X = BatchNormalization(axis=-1, name='{}_bn2'.format(block_name))(X)
    X = Add()([X, X_identity])
    X = Activation('relu', name='{}_a2'.format(block_name))(X)
    return X

def attention_layer(H_lstm, n_layer, n_node, block_name='att'):
    '''
    feedforward attention layer accomplished by time distributed dense layers.
    parameters:
    n_layer: number of hidden layers
    n_node: number of hidden nodes
    '''
    H_emb = H_lstm
    for i in range(n_layer):
        H_lstm = TimeDistributed(Dense(n_node, activation="tanh"), name='{}_dense_{}'.format(block_name, i))(H_lstm)
    M = TimeDistributed(Dense(1, activation="linear"), name='{}_dense_final'.format(block_name))(H_lstm)
    alpha = keras.layers.Softmax(axis=1, name='{}_weights'.format(block_name))(M)
    r_emb = Dot(axes = 1, name='{}_mul'.format(block_name))([alpha, H_emb])
    r_emb = Flatten(name='{}_emb'.format(block_name))(r_emb)
    return r_emb

def fully_connected(r_emb, n_layer, n_node, drop_out_rate, block_name='fc'):
    '''
    fully_connected layer consists of a few dense layers.
    parameters:
    n_layer: number of hidden layers
    n_node: number of hidden nodes
    drop_out_rate: dropout rate to prevent the model from overfitting
    '''
    for i in range(n_layer):
        r_emb = Dense(n_node, activation="relu", name='{}_dense_{}'.format(block_name, i))(r_emb)
    r_emb = Dropout(drop_out_rate, name='dropout')(r_emb) 
    return r_emb

def sequence_attention_model(opt):
    '''
    implementation of sequence attention (Read2Phenotype) model 
    '''
    # Define model
    X = Input(shape=(opt.SEQLEN, opt.BASENUM))
    
    ## CONV Layers
    # no cnn
    if opt.if_cnn == 0:
        X_cnn = X
    # cnn + res_net
    else:
        X_cnn = X
        # cnn
        for i in range(opt.n_cnn_layer):
            X_cnn = conv_net_block(X_cnn, opt.n_cnn_filters, opt.cnn_window, 'convblock_{}'.format(str(i)))
        # res_net
        for i in range(opt.n_cnn_layer):
            X_cnn = res_net_block(X_cnn, opt.n_cnn_filters, opt.cnn_window, 'resblock_{}'.format(str(i)))

    ## RNN Layers
    if opt.if_lstm == 0:
        H_lstm = X_cnn
    elif opt.if_lstm == 1:
        if opt.device == "gpu":
            H_lstm = CuDNNLSTM(opt.n_lstm_node, return_sequences=True, name='LSTM')(X_cnn)
        else:
            H_lstm = LSTM(opt.n_lstm_node, return_sequences=True, name='LSTM')(X_cnn)
    else:
        if opt.device == "gpu":
            H_lstm = Bidirectional(CuDNNLSTM(opt.n_lstm_node, return_sequences=True, activation='tanh',recurrent_activation='sigmoid'), merge_mode='sum', name='LSTM')(X_cnn)
        else:
            H_lstm = Bidirectional(LSTM(opt.n_lstm_node, return_sequences=True, activation='tanh',recurrent_activation='sigmoid'), merge_mode='sum', name='LSTM')(X_cnn)
        H_lstm = Activation('tanh')(H_lstm)
        
    ## ATT Layers
    r_emb = attention_layer(H_lstm, opt.att_n_layer, opt.att_n_node, block_name = 'att')
    
    # additional fully connected
    r_emb = fully_connected(r_emb, opt.fc_n_layer, opt.fc_n_node, opt.drop_out_rate, block_name = 'fc')
    
    if opt.Ty == 2:
        out = Dense(1, activation='sigmoid', name='final_dense')(r_emb)
        model = Model(inputs = X, outputs = out)
        # Compile model
        model.compile(optimizer=Adam(lr = opt.opt_lr, beta_1=0.9, beta_2=0.999, decay=opt.opt_decay),
                        metrics=['accuracy'],
                        loss='binary_crossentropy')
    else:
        out = Dense(opt.Ty, activation='softmax', name='final_dense')(r_emb)
        model = Model(inputs = X, outputs = out)
        # Compile model
        model.compile(optimizer=Adam(lr = opt.opt_lr, beta_1=0.9, beta_2=0.999, decay=opt.opt_decay),
                        metrics=['accuracy'],
                        loss='categorical_crossentropy')
    return model