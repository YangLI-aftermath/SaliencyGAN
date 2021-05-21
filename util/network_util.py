# -*- coding=utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf


########################
# Conv.
########################

def weight_variable(shape):
    # initial = tf.truncated_normal(shape, stddev=0.01)
    # return tf.Variable(initial)
    initial = tf.contrib.layers.xavier_initializer_conv2d() # random innitializer
    return tf.Variable(initial(shape=shape))

def diated_conv2d(input, in_features, out_features, kernel_size, dilated_rate,with_bias=False):
    W = weight_variable([ kernel_size, kernel_size, in_features, out_features ])
    conv = tf.nn.atrous_conv2d(input,W,dilated_rate,padding='SAME')
    if with_bias:
        return conv + bias_variable([ out_features ])
    return conv

def conv2d(input, in_features, out_features, kernel_size,stride, with_bias=False):
    W = weight_variable([kernel_size, kernel_size, in_features, out_features])
    conv = tf.nn.conv2d(input, W, [1, stride, stride, 1], padding='SAME')
    if with_bias:
        return conv + bias_variable([out_features])
    return conv
    
def batchnorm(x,is_training,center=True,scale =True,epsilon = 0.001,decay=0.95):

    shape = x.get_shape().as_list()
    mean, var = tf.nn.moments(x,
                    axes=range(len(shape)-1)# [0] is batch dimension
                )

    ema = tf.train.ExponentialMovingAverage(decay=decay)  # decay of exponential moving average

    def mean_var_with_update():
        ema_apply_op = ema.apply([mean, var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(mean), tf.identity(var)

    #mean, var = mean_var_with_update()
    mean, var = tf.cond(is_training,  # is_training value True/False
                        mean_var_with_update, # if isn't training , use the mean & value from the input and update the shadow variables simultaneously
                        lambda: (
                            ema.average(mean),
                            ema.average(var)
                        )
                        )
    if center == True:
        shift_v = tf.Variable(tf.zeros(shape[-1]))
    else:
        shift_v = None

    if scale == True:
        scale_v = tf.Variable(tf.ones(shape[-1]))
    else:
        scale_v = None


    output = tf.nn.batch_normalization(x, mean, var, shift_v, scale_v, epsilon)
    return output

def diated_conv2d(input, in_features, out_features, kernel_size, dilated_rate,with_bias=False):
    W = weight_variable([ kernel_size, kernel_size, in_features, out_features ])
    conv = tf.nn.atrous_conv2d(input,W,dilated_rate,padding='SAME')
    if with_bias:
        return conv + bias_variable([ out_features ])
    return conv

def batch_activ_conv(x,
                             n_in_features,
                             n_out_features,
                             keep_prob,
                             kernel_size,
                             is_training,
                             scope,
                             stride =1):
    with tf.variable_scope(scope,reuse=True)
        x = batchnorm(x,is_training=is_training)
        x = tf.nn.relu(x)
        x = conv2d(x, in_features, out_features, kernel_size,stride)
        x = tf.nn.dropout(x, keep_prob)
        return x

def batch_activ_conv_trans(scale1_fused,
                                        n_in_features,
                                        n_out_features,
                                        kernel_size,
                                        is_training,
                                        keep_prob,
                                        name,
                                        stride =1
                                        ):
    with tf.variable_scope(scope,reuse=True)
        x = batchnorm(x,is_training=is_training)
        x = tf.nn.relu(x)
        x = conv2d(x, in_features, out_features, kernel_size,stride)
        x = tf.nn.dropout(x, keep_prob)
        return x   
        

def batch_activ_conv_dilated(current, 
                             in_features,
                             out_features, 
                             kernel_size,dilated_rate, 
                             is_training, 
                             keep_prob):
        current = batchnorm(current,is_training=is_training)
        current = tf.nn.relu(current)
        current = diated_conv2d(current, in_features, out_features, kernel_size, dilated_rate)
        current = tf.nn.dropout(current, keep_prob)
        return current


def dense_block(input, 
                             layers, 
                             in_features, 
                             growth,
                             dilated_rate, 
                             is_training, 
                             keep_prob
                             name):
    with tf.variable_scope(name, reuse=True)    
        current = input 
        features = in_features
        for idx in range(layers):
            tmp = batch_activ_conv_dilated(current, features, growth, 3, dilated_rate,is_training, keep_prob) # the number of feature maps produced is growth rate
            current = tf.concat((current, tmp), 3)  # the concantenation of input layer and all preceding layers
            features += growth      # the lth layer has input + (l-1) * growth rate input feature maps
        return current, features #return the output(current) and n_channels

def avg_pool(input, s, padding):
    return tf.nn.avg_pool(input, [1, s, s, 1], [1, s, s, 1], padding=padding)

########################
# ppm
########################

def get_kernel_size(factor):
    """
        Find the kernel size given the desired factor of upsampling.
        """
    return 2 * factor - factor % 2

def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

def bilinear_upsample_weights(factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """

    filter_size = get_kernel_size(factor)

    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)

    upsample_kernel = upsample_filt(filter_size)
    #print ("upsample_kernel=",upsample_kernel)

    for i in xrange(number_of_classes):
        weights[:, :, i, i] = upsample_kernel

    return weights

def upsample(input,factor,channel=1):
    # upsample_weight
    upsample_filter_np = bilinear_upsample_weights(factor,
                                                   channel)
    # Convert to a Tensor type
    upsample_filter_tensor = tf.constant(upsample_filter_np)
    down_shape = tf.shape(input)
    # Calculate the ouput size of the upsampled tensor here only has a shape
    up_shape = tf.stack([
        down_shape[0],
        down_shape[1] * factor,
        down_shape[2] * factor,
        down_shape[3]
    ])
    # Perform the upsampling
    upsampled_input = tf.nn.conv2d_transpose(input, upsample_filter_tensor,
                                           output_shape=up_shape,
                                           strides=[1, factor, factor, 1])
    upsampled_input = tf.reshape(upsampled_input, [-1, up_shape[1], up_shape[2], channel])

    return upsampled_input

#ppm for 64X64
def pyramid_pooling_64(tensor,features_channel,pyramid_feature_chnanels = 1):
    pyramid_layer1 = avg_pool(tensor, 64)
    pyramid_layer1_compressed = conv2d(pyramid_layer1, features_channel, pyramid_feature_chnanels, 1)
    pyramid_layer1_upsampled = upsample(pyramid_layer1_compressed, 64, pyramid_feature_chnanels)

    pyramid_layer2 = avg_pool(tensor,32)
    pyramid_layer2_compressed = conv2d(pyramid_layer2, features_channel, pyramid_feature_chnanels, 1)
    pyramid_layer2_upsampled = upsample(pyramid_layer2_compressed,32,pyramid_feature_chnanels)

    pyramid_layer3 = avg_pool(tensor, 21)
    print("pyramid_layer3",pyramid_layer3)
    pyramid_layer3_compressed = conv2d(pyramid_layer3, features_channel, pyramid_feature_chnanels, 1)
    pyramid_layer3_upsampled = upsample(pyramid_layer3_compressed, 21, pyramid_feature_chnanels)
    pyramid_layer3_upsampled = tf.pad(pyramid_layer3_upsampled,[[0,0],[0,1],[0,1],[0,0]],mode="SYMMETRIC")
    print("pyramid_layer3_upsampled",pyramid_layer3_upsampled)

    pyramid_layer4 = avg_pool(tensor, 10)
    print("pyramid_layer4", pyramid_layer4)
    pyramid_layer4_compressed = conv2d(pyramid_layer4, features_channel, pyramid_feature_chnanels, 1)
    pyramid_layer4_upsampled = upsample(pyramid_layer4_compressed, 10, pyramid_feature_chnanels)
    pyramid_layer4_upsampled = tf.pad(pyramid_layer4_upsampled, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="SYMMETRIC")
    print("pyramid_layer4_upsampled", pyramid_layer4_upsampled)


    pyramid = tf.concat((pyramid_layer1_upsampled,
                         pyramid_layer2_upsampled,
                         pyramid_layer3_upsampled,
                         pyramid_layer4_upsampled),
                        3)  

    pyramid = tf.reshape(pyramid,[-1,64,64,4*pyramid_feature_chnanels])

    return pyramid,pyramid_feature_chnanels*4

########################
# Deconv
########################

def conv2d_trans(input, 
                 in_features, 
                 out_features, 
                 kernel_size,
                 name,
                 stride,
                 ):
    with tf.variable_scope(name,reuse=True)
        _,ht,wd,ch = input.get_shape().as_list()
        W = weight_variable([kernel_size, kernel_size, in_features, out_features])
        conv = tf.nn.conv2d_transpose(input, W, output_shape=stride*ht,[1, 1, 1, 1], padding='SAME')
    return conv 

def batch_activ_conv_trans(scale1_fused,
                                        n_in_features,
                                        n_out_features,
                                        kernel_size,
                                        is_training,
                                        keep_prob,
                                        name,
                                        stride =1
                                        ):
    with tf.variable_scope(scope,reuse=True)
        x = batchnorm(x,is_training=is_training)
        x = tf.nn.relu(x)
        x = conv2d_trans(x, in_features, out_features, kernel_size, name, stride)
        x = tf.nn.dropout(x, keep_prob)
    return x   
 