#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from matplotlib import pyplot as plt

import tensorflow as tf
import keras
from keras import backend as K
from keras import initializers
from keras.layers import Dense, Input, Activation, multiply
from keras.models import Sequential, Model, load_model
from keras.layers.merge import add, concatenate

from deap import base, creator, tools, algorithms
from multiprocessing import Pool
from scoop import futures


# In[2]:


act_dict = {0: 'linear', 1: 'multiply', 2: 'squared', 3: 'sqrt', 4: 'cubed'}
#np.random.seed(np.random.randint(1000, size = 1)
np.random.seed(1000)
weight_dict = {0: 0, 1: 1, 2: np.random.uniform(0,0.001,1)[0]}
bias_dict = {0: 0, 1: np.random.uniform(0,0.001,1)[0]}
#bias_dict = {0: 0, 1: 1.5}
print(weight_dict)
print(bias_dict)
nact_terms = 4
nweight_terms = 12
nbias_terms = 12


# In[3]:


df = pd.read_csv("PropellantData.csv")
print (df.shape)


# In[4]:


df


# In[ ]:


inputs = np.array(df[['HeatofFormation', 'MolesPerGram', 'Q']])
inputs = inputs.astype(np.float)
outputs = np.array(df['Isp']).reshape(-1, 1)
inputs = np.around(inputs, decimals=6)
outputs = np.around(outputs, decimals=2)
print (inputs.shape, outputs.shape)

#train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, outputs, test_size=0.2, random_state=0)
#print (train_inputs.shape, train_outputs.shape)
#print (test_inputs.shape, test_outputs.shape)


# In[ ]:


def squared_act(x):
    return x*x


# In[ ]:


def cubed_act(x):
    return x**3


# In[ ]:


def sqrt_act(x):
    return x**(1/2)


# In[ ]:


class CustomDense(keras.layers.Layer):
    def __init__(self, num_units, input_num, activation, name, trainable_weight, trainable_bias):
        super(CustomDense, self).__init__()
        self.num_units = num_units
        self.activation = Activation(activation)
        self.trainable_weight = trainable_weight
        self.trainable_bias = trainable_bias
        self.name = name
        name_w = 'w'+self.name[1:]
        name_b = 'b'+self.name[1:]
        self.weight = self.add_weight(shape=(input_num, self.num_units), name=name_w, trainable=self.trainable_weight, initializer="zeros")
        self.bias = self.add_weight(shape=(self.num_units,), name=name_b, trainable=self.trainable_bias, initializer="zeros")
        
    def call(self, input):
        y = tf.matmul(input, self.weight) + self.bias
        y = self.activation(y)
        return y


# In[ ]:


def create_node(input1, input2, input3, name, trainable1, trainable2, trainable3, act, bias1, bias2, bias3):
    base = name
    n1 = base + "1"
    n2 = base + "2"
    n3 = base + "3"
    
    an1 = CustomDense(1, 1, activation = 'linear', name=n1, trainable_weight=trainable1, trainable_bias=bias1) (input1)
    an2 = CustomDense(1, 1, activation = 'linear', name=n2, trainable_weight=trainable2, trainable_bias=bias2) (input2)
    an3 = CustomDense(1, 1, activation = 'linear', name=n3, trainable_weight=trainable3, trainable_bias=bias3) (input3)  
    if (act == "multiply"):
        an = multiply([an1, an2, an3])
    else:
        an = add([an1, an2, an3])
        if (act == "squared"):
            an = Activation(squared_act) (an)
        elif (act == "cubed"):
            an = Activation(cubed_act) (an)
        elif (act == "sqrt"):
            an = Activation(sqrt_act) (an)
        else:
            an = Activation(act) (an)
    return an


# In[ ]:


def create_model(x):
    #print("Individual: ", x)
    #initializer = keras.initializers.RandomUniform(minval=-0.001, maxval=0.001, seed=0)
    bias_initial = keras.initializers.Zeros()

    trainable_list = []
    for i in range(nweight_terms):
        if (x[i+nact_terms] == 2):
            trainable_list.append(True)
        else:
            trainable_list.append(False)

    for i in range(nbias_terms):
        if (x[i+nact_terms+nweight_terms] == 1):
            trainable_list.append(True)
        else:
            trainable_list.append(False)
    
    input1 = Input(shape=(1,))
    input2 = Input(shape=(1,))
    input3 = Input(shape=(1,))
    #print(trainable_list)
    a1 = create_node(input1, input2, input3, "a1", trainable_list[0], trainable_list[1], trainable_list[2], act_dict[x[0]], trainable_list[12], trainable_list[13], trainable_list[14])
    a2 = create_node(input1, input2, input3, "a2", trainable_list[3], trainable_list[4], trainable_list[5], act_dict[x[1]], trainable_list[15], trainable_list[16], trainable_list[17])
    a3 = create_node(input1, input2, input3, "a3", trainable_list[6], trainable_list[7], trainable_list[8], act_dict[x[2]], trainable_list[18], trainable_list[19], trainable_list[20])

    output = create_node(a1, a2, a3, "output", trainable_list[9], trainable_list[10], trainable_list[11], act_dict[x[3]], trainable_list[21], trainable_list[22], trainable_list[23])

    model = Model(inputs=[input1, input2, input3], outputs=[output])
   # Learning rate decay/schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-1,
                                                                decay_steps=10000,
                                                                decay_rate=0.1,
                                                                staircase=False)
    
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(loss='mse', optimizer=optimizer)
    model.layers.sort(key = lambda x: x.name, reverse=False)
    
    layer_list = []
    for i in range(len(model.layers)):
        name = model.layers[i].name
        if ( ("activation" in name) or ("input" in name) or ("add" in name) or ("multiply" in name) ):
            continue
        else:
            layer_list.append(i)
    
    for i in range(len(layer_list)):
        if (model.layers[layer_list[i]].get_weights()[0].shape==(1,1)):
            model.layers[layer_list[i]].set_weights( [ np.array( [[ weight_dict[x[nact_terms+i]] ]] ),  np.array( [ bias_dict[x[nact_terms+nweight_terms+i]] ] ) ] )
        else:
            model.layers[layer_list[i]].set_weights( [ np.array( [ bias_dict[x[nact_terms+nweight_terms+i]] ] ),  np.array( [[ weight_dict[x[nact_terms+i]] ]] ) ] )
        
    #model.summary()

    return model, trainable_list


# In[ ]:


losses = []
class PrintEpNum(keras.callbacks.Callback): # This is a function for the Epoch Counter
    def on_epoch_end(self, epoch, logs):
        sys.stdout.flush()
        #sys.stdout.write("Current Epoch: " + str(epoch+1) + ' Loss: ' + str(logs.get('loss')) + '                     \n')
        losses.append(logs.get('loss'))

def train(model, train_inputs, train_outputs, verbose=False):
    mae_es= keras.callbacks.EarlyStopping(monitor='val_loss', patience=1000,
                                          min_delta=1e-8, verbose=1, mode='auto', restore_best_weights=True)
    
    terminate = keras.callbacks.TerminateOnNaN()

    EPOCHS = 50000 # Number of EPOCHS
    history = model.fit([train_inputs[:,0], train_inputs[:,1], train_inputs[:,2]], train_outputs[:,0], epochs=EPOCHS,
                        shuffle=False, batch_size=32, verbose = False, callbacks=[terminate, mae_es],
                        validation_split=0.2)

    if verbose:
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Sq Error')
        plt.plot(history.epoch, np.array(history.history['loss']),label='Training loss')
        plt.legend()
        plt.show()
    return history


# In[ ]:


def cv_error(individual, inputs, outputs):
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    kf.get_n_splits(inputs)
    
    cv_mse_list = []
    

    for train_index, test_index in kf.split(inputs):
        new_model, trainable = create_model(individual)
        #print ("Trainable: ", trainable)
        if (any(trainable[:nweight_terms+nbias_terms]) == True):
            train(new_model, inputs[train_index, :], outputs[train_index], verbose=False)
    
        weights = new_model.get_weights()
        weight_list = []
        bias_list = []
        for weight in weights:
            if (weight.shape == (1,1)):
                weight_list.append(weight[0])
            else:
                bias_list.append(weight[0])
        weight_list = np.array(weight_list)
        bias_list = np.array(bias_list)
        #print(weight_list)
        #print(bias_list)
    
        #handle nan weights and biases
        if (np.isnan(weight_list).any()):
            cv_mse = 1e50
        elif (np.isnan(bias_list).any()):
            cv_mse = 1e50
        else:
            cv_mse = new_model.evaluate([inputs[test_index, 0], inputs[test_index, 1], inputs[test_index, 2]], outputs[test_index])
            if (np.isnan(cv_mse)):
                cv_mse = 1e50
            else:
                cv_mse = np.around(cv_mse, decimals=6)
        cv_mse_list.append(cv_mse)
    #print(cv_mse_list)
    return np.mean(cv_mse_list)


# In[ ]:


def f3(w):
    return w


# In[ ]:


def objective_function(individual):
    mse_test = cv_error(individual, inputs, outputs)

    acts = individual[:nact_terms]
    actfunc_term = 0
    for i in range(nact_terms):
        actfunc_term += f3(acts[i])

    wtbs = individual[nact_terms:]
    wtbs_term = 0
    for j in range(nweight_terms):
        if (f3(wtbs[j]) == 0):
                wtbs_term += 0
        elif (f3(wtbs[j]) == 2):
            wtbs_term += 2
        else:
            wtbs_term += 1
            
    for i in range(nbias_terms):
        j = i + nweight_terms
        if (f3(wtbs[j]) == 0):
            wtbs_term += 0
        elif (f3(wtbs[j]) == 1):
            wtbs_term += 2
        else:
            wtbs_term += 1
    
    mse_test_term = mse_test 

    obj = mse_test_term + 0.001*(np.sum(actfunc_term) + wtbs_term)
    #obj = mse_test_term
    print ("Individual: ", individual, flush=True)
    print ("Objective function: ", mse_test, np.sum(actfunc_term), wtbs_term, obj, flush=True)
    
    K.clear_session()
    #tf.reset_default_graph()
    return (obj,) 


# In[ ]:


################### DEAP #####################
#create fitness class and individual class
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
#pool = Pool(1)
#toolbox.register("map", futures.map)
#toolbox.register("attr_int", random.randint, 0, 3)

def custom_initRepeat(container, func, max1, max2, max3, n):
    func_list = []
    for i in range(n):
        if (i < nact_terms):
            func_list.append(func(0, max1))
        elif (i >= nact_terms+nweight_terms):
            func_list.append(func(0, max3))
        else:
            func_list.append(func(0, max2))
    return container(func_list[i] for i in range(n))

#gen = initRepeat(list, random.randint, 3, 7, 4)
toolbox.register("create_individual", custom_initRepeat, creator.Individual, random.randint, 
                 max1=4, max2=2, max3=1, n=nact_terms+nweight_terms+nbias_terms)
toolbox.register("population", tools.initRepeat, list, toolbox.create_individual)

cxpb = 0.5
mutpb = 0.3
ngens = 50

def custom_mutation(individual, max1, max2, max3, indpb):
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            if (i < nact_terms):
                individual[i] = random.randint(0, max1)
            elif (i >= nact_terms+nweight_terms):
                individual[i] = random.randint(0, max3)
            else: 
                individual[i] = random.randint(0, max2)
    return individual,

toolbox.register("mate", tools.cxTwoPoint)
#toolbox.register("mutate", tools.mutUniformInt, low=0, up=3, indpb=mutpb)
toolbox.register("mutate", custom_mutation, max1=3, max2=2, max3=1, indpb=mutpb)
toolbox.register("select", tools.selTournament, tournsize=10)
toolbox.register("evaluate", objective_function)

def main():
    random.seed(10000)
    
    #correct_individual = [[0, 0, 0, 3, 0, 0, 2, 0, 2, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #            [0, 0, 0, 3, 0, 0, 2, 0, 2, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #            [0, 0, 0, 3, 0, 0, 2, 0, 2, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #            [0, 0, 0, 3, 0, 0, 2, 0, 0, 0, 0, 2, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    #            [0, 0, 0, 3, 0, 0, 2, 0, 0, 0, 0, 2, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]
    
    population = toolbox.population(n=200)
    #for i in range(len(correct_individual)):
    #    population[0][i] = correct_individual[i]
    #    ilist.append(population[0][i]())
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    pop, logbook = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngens, stats=stats, halloffame=hof, verbose=True)

    return pop, logbook, hof

if __name__ == "__main__":
    pop, logbook, hof = main()
    print (logbook, flush=True)
    print (hof, flush=True)


# In[ ]:




