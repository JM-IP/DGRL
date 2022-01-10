#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-07-10 20:18
# @Author  : Jianming Ip
# @Site    : 
# @File    : multibin_layer.py
# @Company : VMC Lab in Peking University

import keras
import tensorflow as tf
import larq as lq
import numpy as np
class Multibin(tf.keras.layers.Layer):

  def __init__(self, out_channel, kernel_size, M=1, **kwargs):
    self.out_channel = out_channel
    self.kernel_size = kernel_size
    self.M=M
    super(Multibin, self).__init__(**kwargs)


  def build(self, **kwargs):
    super(Multibin, self).build(**kwargs)
    # Create a trainable weight variable for this layer.
    self.layers=keras.Sequential()
    for i in range(self.M):
        self.layers.add(lq.layers.QuantConv2D(self.out_channel, self.kernel_size, padding="same", **kwargs))
    # Be sure to call this at the end


  def call(self, inputs):
    outputs=self.layers[0](inputs)
    for i in range(1, self.M):
        outputs = outputs+self.layers[i](inputs)
    return outputs

  # def compute_output_shape(self, input_shape):
  #   shape = tf.TensorShape(input_shape).as_list()
  #   shape[-1] = self.out_channel
  #   return tf.TensorShape(shape)

  def get_config(self):
    base_config = super(Multibin, self).get_config()
    base_config['output_dim'] = self.output_dim

  @classmethod
  def from_config(cls, config):
    return cls(**config)


# dataset
data = np.random.random((1000, 32 , 32, 3))
labels = np.random.random((1000, 10))
# Create a model using the custom layer
inputs = keras.Input(shape=(32,))
Multibin1 = Multibin()
out1 = Multibin1(10,3)(inputs)
out2 = tf.Dense(10)(out1)
final = keras.layers.Activation('softmax')(out2)

# The compile step specifies the training configuration
final.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy', out1, out2])

final.fit(data, labels, epochs=10, batch_size=32)