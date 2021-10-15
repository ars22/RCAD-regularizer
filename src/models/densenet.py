
import numpy as np
from keras.layers import Dense, Dropout, Flatten, Input, GlobalAveragePooling2D, merge, Activation, ZeroPadding2D, Conv2D, MaxPooling2D, BatchNormalization, Concatenate, GlobalMaxPooling2D
from keras.regularizers import l2
from keras.models import Model, Sequential
import tensorflow as tf

COMPRESSION = 1.0
CHANNEL = 3
NUM_FILTER = 128
DROPOUT_RATE = 0.
N_LAYERS = 3

# Dense Block
def add_denseblock(input, num_filter = 12, dropout_rate = 0.2):
  temp = input
  for _ in range(N_LAYERS):
      BatchNorm = BatchNormalization(epsilon=1.1e-5)(temp)
      relu = Activation('relu')(BatchNorm)
      # kernel_regularizer to regularze kernel weights
      # l2 for penallizing weights with large magnitudes
      Conv2D_3_3 = Conv2D(int(num_filter*COMPRESSION), (3,3), use_bias=False, padding='same', kernel_regularizer=l2(0.0002))(relu) 
      if dropout_rate>0:
        Conv2D_3_3 = Dropout(dropout_rate)(Conv2D_3_3)
      concat = Concatenate(axis=-1)([temp,Conv2D_3_3])
      temp = concat
  return temp


def add_transition(input, num_filter = 12, dropout_rate = 0.2):
  BatchNorm = BatchNormalization(epsilon=1.1e-5)(input)
  relu = Activation('relu')(BatchNorm)
  # kernel_regularizer to regularize kernel weights
  # l2 for penallizing weights with large magnitudes
  Conv2D_BottleNeck = Conv2D(int(num_filter*COMPRESSION), (1,1), use_bias=False ,padding='same',kernel_regularizer=l2(0.0002))(relu)
  if dropout_rate>0:
    Conv2D_BottleNeck = Dropout(dropout_rate)(Conv2D_BottleNeck)
  avg = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(Conv2D_BottleNeck)
  return avg


def output_layer(input, num_classes, softmax):
  BatchNorm = BatchNormalization()(input)
  relu = Activation('relu')(BatchNorm)
  conv2D = Conv2D(num_classes, (1,1), use_bias=False ,kernel_regularizer=l2(0.0002))(relu)
  BatchNorm = BatchNormalization()(conv2D)
  relu = Activation('relu')(BatchNorm)
  GAP = GlobalAveragePooling2D()(relu)
  if softmax:
    return Activation('softmax')(GAP)    
  return GAP


def DenseNet(num_classes, softmax):
  input = Input(shape=(None, None, CHANNEL,))
  First_Conv2D = Conv2D(NUM_FILTER, (3, 3), use_bias=False, padding='same')(input)
  First_Block = add_denseblock(First_Conv2D, NUM_FILTER, DROPOUT_RATE)
  First_Transition = add_transition(First_Block, num_filter=256, dropout_rate=DROPOUT_RATE)
  Second_Block = add_denseblock(First_Transition, NUM_FILTER, DROPOUT_RATE)
  Second_Transition = add_transition(Second_Block, num_filter=320,dropout_rate=DROPOUT_RATE)
  Third_Block = add_denseblock(Second_Transition, NUM_FILTER, DROPOUT_RATE)
  Third_Transition = add_transition(Third_Block, num_filter=384, dropout_rate=DROPOUT_RATE)
  Fourth_Block = add_denseblock(Third_Transition, NUM_FILTER, DROPOUT_RATE)
  Fourth_Transition = add_transition(Fourth_Block, num_filter=512, dropout_rate=DROPOUT_RATE)
  Fifth_Block = add_denseblock(Fourth_Transition, NUM_FILTER, DROPOUT_RATE)
  output = output_layer(Fifth_Block, num_classes, softmax)
  model = Model(inputs=[input], outputs=[output])
  return model


if __name__ == '__main__':
  model = DenseNet(num_classes=200)
  print(model.summary())