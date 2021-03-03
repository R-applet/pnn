import sys
import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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


act_dict = {0: 'linear', 1: 'multiply', 2: 'sqrt', 3: '4rt'}
np.random.seed(100000)
weight_dict = {0: 0, 1: 1, 2: np.random.uniform(1.1,2.1,1)[0]}
bias_dict = {0:0,1:1} #,2: np.random.uniform(-0.001,0.001,1)[0]}
nact_terms = 5
nweight_terms = 20
nbias_terms = 5


data = pd.read_csv("data.csv")
print (data.shape)


inputs = np.array(data[['N','M','Q','r0']])
#outputs = np.array(data[['D','P']])
outputs = np.array(data['D_KJ'])
#inputs = np.around(inputs, decimals=6)
#outputs = np.around(outputs, decimals=6)

print (inputs.shape, outputs.shape)

train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, outputs, test_size=0.2, random_state=0)
print (train_inputs.shape, train_outputs.shape)
print (test_inputs.shape, test_outputs.shape)


def sqrt_act(x):
    return x**(1/2)

def frt_act(x):
    return x**(1/4)



def create_node(input1, input2, input3, input4, name, trainable1, trainable2, trainable3, trainable4, act):
    base = name
    n1 = base + "1"
    n2 = base + "2"
    n3 = base + "3"
    n4 = base + "4"
    an1 = Dense(1, activation = 'linear', use_bias = False, name=n1, trainable=trainable1) (input1)
    an2 = Dense(1, activation = 'linear', use_bias = False, name=n2, trainable=trainable2) (input2)
    an3 = Dense(1, activation = 'linear', use_bias = False, name=n3, trainable=trainable3) (input3)
    an4 = Dense(1, activation = 'linear', use_bias = True, name=n4, trainable=trainable4) (input4)
    if (act == "multiply"):
        an = multiply([an1, an2, an3, an4])
    
    else:
        an = add([an1, an2, an3, an4])
        if (act == "4rt"):
            an = Activation(frt_act) (an)
        elif (act == "sqrt"):
            an = Activation(sqrt_act) (an)
        else:
            an = Activation(act) (an)
    return an

def create_model(x):
    #initializer = keras.initializers.RandomUniform(minval=-0.001, maxval=0.001, seed=0)
    bias_initial = keras.initializers.Zeros()
    print(x)
    trainable_list = []
    for i in range(nweight_terms):
        if (x[i+nact_terms] == 2):
            trainable_list.append(True)
        else:
            trainable_list.append(False)

    input1 = Input(shape=(1,))
    input2 = Input(shape=(1,))
    input3 = Input(shape=(1,))
    input4 = Input(shape=(1,))

    a1 = create_node(input1, input2, input3, input4, "a1", trainable_list[0], trainable_list[1], trainable_list[2], trainable_list[3], act_dict[x[0]])
    a2 = create_node(input1, input2, input3, input4, "a2", trainable_list[4], trainable_list[5], trainable_list[6], trainable_list[7], act_dict[x[1]])
    a3 = create_node(input1, input2, input3, input4, "a3", trainable_list[8], trainable_list[9], trainable_list[10], trainable_list[11], act_dict[x[2]])
    a4 = create_node(input1, input2, input3, input4, "a4", trainable_list[12], trainable_list[13], trainable_list[14], trainable_list[15], act_dict[x[3]])

    output = create_node(a1, a2, a3, a4, "output", trainable_list[16], trainable_list[17], trainable_list[18], trainable_list[19], act_dict[x[4]])
   # output2 = create_node(a1, a2, a3, a4, "output2", trainable_list[20], trainable_list[21], trainable_list[22], trainable_list[23], act_dict[x[5]])

    model = Model(inputs=[input1, input2, input3, input4], outputs=output)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    model.compile(loss='mse', optimizer=optimizer)
    
    layer_list = []
    for i in range(len(model.layers)):
        name = model.layers[i].name
        if ( ("activation" in name) or ("input" in name) or ("add" in name) or ("multiply" in name) ):
            continue
        else:
            layer_list.append(i)
    
    for i in range(len(layer_list)):
        name = model.layers[layer_list[i]].name
        if (("a14" in name) or ("a24" in name) or ("a34" in name) or ("a44" in name) or ("output4" in name)):
            model.layers[layer_list[i]].set_weights( [ np.array( [[ weight_dict[x[nact_terms+i]] ]] ) , np.array( [ bias_dict[x[nact_terms+nweight_terms+int((i+1)/5)]] ] ) ] )
        else:
            model.layers[layer_list[i]].set_weights( [ np.array( [[ weight_dict[x[nact_terms+i]] ]] ) ] )
        

    #model.summary()

    return model, trainable_list


losses = []
class PrintEpNum(keras.callbacks.Callback): # This is a function for the Epoch Counter
    def on_epoch_end(self, epoch, logs):
        sys.stdout.flush()
        sys.stdout.write("Current Epoch: " + str(epoch+1) + ' Loss: ' + str(logs.get('loss')) + '                     \n')
        losses.append(logs.get('loss'))
        
def train(model, train_inputs, train_outputs, verbose=False):
    mae_es= keras.callbacks.EarlyStopping(monitor='val_loss', patience=1000,
                                          min_delta=1e-5, verbose=1, mode='auto', restore_best_weights=True)

    terminate = keras.callbacks.TerminateOnNaN()

    EPOCHS = 10000 # Number of EPOCHS
    history = model.fit([train_inputs[:,0], train_inputs[:,1], train_inputs[:,2], train_inputs[:,3]], train_outputs, epochs=EPOCHS,
                        shuffle=False, batch_size=len(train_inputs), verbose = False, callbacks=[terminate, mae_es],
                        validation_split=0.2)

    if verbose:
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Sq Error')
        plt.plot(history.epoch, np.array(history.history['loss']),label='Training loss')
        plt.legend()
        plt.show()
    return history


def f3(w):
    return w

def objective_function(individual):
    new_model, trainable = create_model(individual)
    print ("Trainable: ", trainable)
    if (any(trainable) == True):
        train(new_model, train_inputs, train_outputs, verbose=False)
    
    wt_bs = new_model.get_weights()
    weight_list = []
    bias_list = []
    for i,value in enumerate(wt_bs):
        if ((i == 4) or (i == 9) or (i == 14) or (i == 19) or (i == 24)):
            bias_list.append(value[0])
        else:
            weight_list.append(value[0])

    bias_list = np.array(bias_list)        
    bias_list = np.transpose(bias_list)
    weight_list = np.array(weight_list)
    weight_list = np.transpose(weight_list)
    print (weight_list)
    print (bias_list)
    
    #handle nan weights
    if (np.isnan(weight_list).any()):
        mse_test = 1e50
    else:
        mse_test = new_model.evaluate([test_inputs[:, 0], test_inputs[:, 1], test_inputs[:, 2], test_inputs[:,3]], test_outputs)
        
    acts = individual[:nact_terms]
    actfunc_term = 0
    for i in range(nact_terms):
        actfunc_term += f3(acts[i])

    wtbs = individual[nact_terms:]
    wtbs_term = 0
    for j in range(nweight_terms+nbias_terms):
        wtbs_term += f3(wtbs[j])

    mse_test_term = mse_test

    obj = mse_test_term + np.sum(actfunc_term) + wtbs_term
    print ("Individual: ", individual, flush=True)
    print ("Objective function: ", mse_test, np.sum(actfunc_term), wtbs_term, obj, flush=True)
    
    K.clear_session()
    tf.reset_default_graph()
    return (obj,) 


################### DEAP #####################
#create fitness class and individual class
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
#pool = Pool(1)
toolbox.register("map", futures.map)
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
                 max1=3, max2=2, max3=1, n=nact_terms+nweight_terms+nbias_terms)
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

correct_individual = [2, 3, 3, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 2, 1, 1, 1, 0, 0, 0, 1, 0]

def main():
    random.seed(100000)
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
