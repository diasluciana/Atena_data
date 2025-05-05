#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install tensorflow-quantum')


# In[2]:


#get_ipython().system('pip install -U tensorflow==2.3.0')


# In[3]:


#Importing tensorflow
import tensorflow as tf
#import tensorflow_quantum as tfq
import time
from tensorflow.keras import datasets, layers, models
from keras.layers import Conv2D, MaxPooling2D
#Importing some tensorflow quantum stuff
#import cirq
import sympy
import pandas as pd
#For data manipulation
import numpy as np
from sklearn import metrics
import collections
import pickle
#to plot stuff
import matplotlib.pyplot as plt
import os
import seaborn as sns
import h5py
import numpy as np
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split
from random import sample
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
#import pennylane as qml
#from pennylane import numpy as np
#from pennylane.templates import RandomLayers

def split_dataArray(inputs, outputs, test_split=0.05, val_split=0.05):
    N = inputs.shape[0]
    ids = np.random.permutation(N)
    nval = int((1 - test_split) * N)
    ntrain = int((1 - val_split) * N)
    XTrain = inputs[ids[:ntrain],:,:,:]
    YTrain = outputs[ids[:ntrain]]
    XVal = inputs[ids[ntrain:ntrain+nval],:,:,:]
    YVal = outputs[ids[ntrain:ntrain+nval]]
    XTest = inputs[ids[ntrain+nval:],:,:,:]
    YTest = outputs[ids[ntrain+nval:]]
    print("########")
    print(XTrain.shape)
    print(YTrain.shape)
    print(XVal.shape)
    print(YVal.shape)

    return XTrain, YTrain, XVal, YVal, XTest, YTest, {'training': ids[:ntrain],
            'validation': ids[ntrain:ntrain+nval], 'testing': ids[ntrain+nval:]}



#Defining the whole thing!!!
class QConv(tf.keras.layers.Layer):
  #initializaiton of the QCNN layer
  def __init__(self, filter_size, depth, activation = None, name = None, kernel_regularizer=None, **kwangs):
    #Standard notation thingy
    super(QConv, self).__init__(name=name, **kwangs)

    #Defining of the varaibles
    self.filter_size = filter_size
    self.depth = depth
    self.learning_params = []
    self.QCNN_layer_gen()
    self.activation = tf.keras.layers.Activation(activation)
    self.kernel_regularizer = kernel_regularizer
     
  def get_config(self):
    config = super(QConv, self).get_config()
    config.update({"filter_size": self.filter_size})
    config.update({"depth": self.depth})
    config.update({"activation": self.activation})
    config.update({"kernel_regularizer": self.kernel_regularizer})
    #config.update({"padding": "same"})
    return config

  #Initialize parameters for the quantum gates
  def _get_new_param(self):
    #Literally just generates a string "p0"... Instead of 0 it's just a number that
    new_param = sympy.symbols('p'+str(len(self.learning_params)))
    #Increase the size of the list (thus the numbers keep increasing (so there's no duplicates))
    self.learning_params.append(new_param)
    return new_param
  
  #This just defines 2 parameterized qubit gates and places them
  def _QConv(self, step, target, qubits):
    import cirq
    #First defining a Z and an X gate. First part = the rotation value (where we place our parameter), second part = where we place our gates
    yield cirq.CZPowGate(exponent=self._get_new_param())(qubits[target], qubits[target+step])
    yield cirq.CXPowGate(exponent=self._get_new_param())(qubits[target], qubits[target+step])
  
  def QCNN_layer_gen(self):
    #Pixels = the area which the filter will cover
    pixels = self.filter_size**2
    import cirq
    #So we're going to take our kernal and map it to qubits
    cirq_qubits = cirq.GridQubit.rect(self.filter_size, self.filter_size)

    #How you define the start of a quantum circuit
    input_circuit = cirq.Circuit()

    #There's another set of parameterized gates here. And we've got to define it's parameters
    input_params = [sympy.symbols('a%d' %i) for i in range(pixels)]

    #Now we apply those initial RX gates at the beginning for each qubit
    for i, qubit in enumerate(cirq_qubits):
      input_circuit.append(cirq.rx(np.pi*input_params[i])(qubit))
    
    #We're going to start antoher part, this time it's the kernal part
    QCNN_circuit = cirq.Circuit()

    #Basically something to help with the architechture of the kernal part (to help with the placement of the X and Z gates)
    step_size = [2**i for i in range(np.log2(pixels).astype(np.int32))]
    
    #This is the appending of said X and Z gates
    for step in step_size:
      for target in range(0, pixels, 2*step):
        QCNN_circuit.append(self._QConv(step,target,cirq_qubits))
    
    #now take the 2 parts of the quantum circuit to merge them all together
    full_circuit = cirq.Circuit()
    full_circuit.append(input_circuit)
    full_circuit.append(QCNN_circuit)

    #save it to use it later
    self.circuit = full_circuit

    #Save the parameters to use later
    self.params = input_params + self.learning_params

    #Save the operators (for the output) for later use
    self.op = cirq.Z(cirq_qubits[0])
  
  #Intializes everything... It creates the layer (with weights and stuff)
  def build(self, input_shape):
    #What's the input (image) width? Height? Number of channels?
    self.width = input_shape[1]
    self.height = input_shape[2]
    self.channel = input_shape[3]

    # The number of times which the kernal will pass on the image
    self.num_x = self.width - self.filter_size + 1
    self.num_y = self.height - self.filter_size + 1

    #Initializing the kernal! name, (how many (if there are 8, then it'll be a rectangular prism, but ostensibly 8 different kernals), channels, number of parameters each)
    #Then we initialzie the parameters, plus slap on a regularator if we wanted to
    self.kernel = self.add_weight(name = 'kernal',
                                 shape = [self.depth, self.channel, len(self.learning_params)],
                                 initializer = tf.keras.initializers.he_normal(),
                                 regularizer = self.kernel_regularizer)
    
    import tensorflow_quantum as tfq
    #We take our thing and convert it to a (quantum?) tensor
    self.circuit_tensor = tfq.convert_to_tensor([self.circuit] * self.num_x * self.num_y * self.channel)
  
  #Where the computation happens
  def call(self, inputs):
    #This is generating a giant stack of all the segements of the inputs which we're going to pass over the kernal
    #Also: It's just adding the slice to the whole stack each time. (It works like. Which cord on the map, then take a bit out of that with the size) 
    stack_set = None
    for i in range(self.num_x):
      for j in range(self.num_y):
        slice_part = tf.slice(inputs, [0, i, j, 0], [-1, self.filter_size, self.filter_size, -1])
        slice_part = tf.reshape(slice_part, shape=[-1, 1, self.filter_size, self.filter_size, self.channel])
        if stack_set == None:
          stack_set = slice_part
        else:
          stack_set = tf.concat([stack_set, slice_part], 1)
    #Then we just reformat it
    stack_set = tf.transpose(stack_set, perm=[0, 1, 4, 2, 3])
    stack_set = tf.reshape(stack_set, shape=[-1, self.filter_size**2])

    #Kind of reformats (except with some duplication) the (quantum?) tensor into a usable form
    circuit_inputs = tf.tile([self.circuit_tensor], [tf.shape(inputs)[0], 1])
    circuit_inputs = tf.reshape(circuit_inputs, shape=[-1])
    tf.fill([tf.shape(inputs)[0]*self.num_x*self.num_y, 1], 1)
    
    #Gonna take our inputs (now in the form of the stack) and pass them through our kernals
    outputs = []
    for i in range(self.depth):
      #Now we call the kernals we defined in build
      controller = tf.tile(self.kernel[i], [tf.shape(inputs)[0]*self.num_x*self.num_y, 1])
      #Actually passing into the QCNN layer
      outputs.append(self.single_depth_QCNN(stack_set, controller, circuit_inputs))
    #reformating
    output_tensor = tf.stack(outputs, axis=3)
    output_tensor = tf.math.acos(tf.clip_by_value(output_tensor, -1+1e-5, 1-1e-5)) / np.pi
    
    #Take our output, shove it through the activation, and then return it
    return self.activation(output_tensor)
  
  def single_depth_QCNN(self, input_data, controller, circuit_inputs):
    #Reformat the input data
    input_data = tf.concat([input_data, controller],1)
    #Then taking our input and shoving it through the QCNN (along with it's paramters)
    
    import tensorflow_quantum as tfq
    QCNN_output = tfq.layers.Expectation()(circuit_inputs,
                                            symbol_names = self.params,
                                            symbol_values = input_data,
                                            operators = self.op)
    #Reformat x 2
    QCNN_output = tf.reshape(QCNN_output, shape=[-1, self.num_x, self.num_y, self.channel])
    return tf.math.reduce_sum(QCNN_output, 3)

def DCM(cm,index,columns):
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm_perc, index=classes, columns=classes)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'

    return cm, annot


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred) #### sugestão do Bernardo Fraga


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.) ### sugestão do Bernardo Fraga
    #return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)

def dice_coef_plot(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.) ### sugestão do Bernardo Fraga
    #return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)

#def dice_coef_loss(y_true, y_pred):
#    return -dice_coef(y_true, y_pred)


def MinMaxScaler(array, scale=[0, 1]):
    array_max = array.max()
    array_min = array.min()
    return scale[0] + (((array - array_min) * (scale[1] - scale[0])) / (array_max - array_min))


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    if stepSize == 1:
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
    else:
        Y = np.arange(0, image.shape[0], stepSize)
        X = np.arange(0, image.shape[1], stepSize)
        for n,y in enumerate(Y):
            for m,x in enumerate(X):
                if n != len(Y)-1 or m != len(X)-1: 
                    yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
                if n == len(Y)-1 and m != len(X)-1 and y+windowSize[1] < image.shape[0]:
                    yield (x, image.shape[0]-windowSize[1], image[image.shape[0]-windowSize[1]:image.shape[0], x:x + windowSize[0]])
                if n != len(Y)-1 and m == len(X)-1 and x+windowSize[0] < image.shape[1]:
                    yield (image.shape[1]-windowSize[0], y, image[y:y + windowSize[1], image.shape[1]-windowSize[0]:image.shape[1]])
                if n == len(Y)-1 and m == len(X)-1: 
                    if x+windowSize[0] < image.shape[1] and y+windowSize[1] < image.shape[0]:
                        yield (image.shape[1]-windowSize[0],image.shape[0]-windowSize[1],image[image.shape[0]-windowSize[1]:image.shape[0],image.shape[1]-windowSize[0]:image.shape[1]])

def sliding_windowStand(image, stepSize, windowSize):
    # slide a window across the image
    if stepSize == 1:
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                img = image[y:y + windowSize[1], x:x + windowSize[0]]
                yield (x, y, img)
    else:
        Y = np.arange(0, image.shape[0], stepSize)
        X = np.arange(0, image.shape[1], stepSize)
        for n,y in enumerate(Y):
            for m,x in enumerate(X):
                if n != len(Y)-1 or m != len(X)-1:
                    img = image[y:y + windowSize[1], x:x + windowSize[0]]
                    yield (x, y, img)
                if n == len(Y)-1 and m != len(X)-1 and y+windowSize[1] < image.shape[0]:
                    img = image[image.shape[0]-windowSize[1]:image.shape[0], x:x + windowSize[0]]
                    yield (x, image.shape[0]-windowSize[1], img)
                if n != len(Y)-1 and m == len(X)-1 and x+windowSize[0] < image.shape[1]:
                    img = image[y:y + windowSize[1], image.shape[1]-windowSize[0]:image.shape[1]]
                    yield (image.shape[1]-windowSize[0], y, img)
                if n == len(Y)-1 and m == len(X)-1:
                    img = image[image.shape[0]-windowSize[1]:image.shape[0],image.shape[1]-windowSize[0]:image.shape[1]]
                    if x+windowSize[0] < image.shape[1] and y+windowSize[1] < image.shape[0]:
                        yield (image.shape[1]-windowSize[0],image.shape[0]-windowSize[1],img)

def plotMosaic(img_arrs, path=os.getcwd()):
    """ img_arrs = array list append 
    https://www.kaggle.com/code/jiny333/tutorial-on-using-subplots-in-matplotlib"""
    columns = 3
    rows = int(len(img_arrs)/columns)

    fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(columns*columns, rows*columns))

    for num in np.arange(0, rows*columns,1):
    
        fig.add_subplot(rows, columns, num+1)
    
        plt.imshow(img_arrs[num], cmap='gray') #, vmin=0.0, vmax=1.0)
    
        fig.tight_layout() # used to adjust padding between subplots

        #cols = ['netINPUT', 'netOUTPUT','ColorPlotOUTPUT']

        #for ax, col in zip(axes[0], cols):
        #    ax.set_title(col)

        for idx, ax in enumerate(axes.flat):
            ax.set_xticks([])
            ax.set_yticks([])

    fig.savefig(path)
    plt.pause(1)
    plt.close()


def report_model(model, pathsave):
    from contextlib import redirect_stdout
    print(model.summary())
    with open(pathsave, 'w') as f:
        with redirect_stdout(f):
            model.summary()
