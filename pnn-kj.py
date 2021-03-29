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


act_dict = {0: 'linear', 1: 'multiply', 2: 'sqrt', 3: '4rt'}
np.random.seed(100000)
weight_dict = {0: 0, 1: 1, 2: np.random.uniform(0.0,1.0,1)[0]}
bias_dict = {0: 0, 1: 1, 2: np.random.uniform(0.0,0.001,1)[0]}
nact_terms = 5
nweight_terms = 20
nbias_terms = 5


df = pd.read_csv('CHNO_data.csv')
#print (df.shape)

df = df.round({'N_arb': 6, 'M_arb': 3, 'Q_arb': 3, 'Density [g/cc]': 3, 'D (exp) [km/s]': 3})

inputs = np.array(df[['N_arb','M_arb','Q_arb','Density [g/cc]']])

outputs = np.array(df['D (exp) [km/s]'])


#Train_inputs, Test_inputs, Train_outputs, Test_outputs = train_test_split(inputs, outputs, test_size=0.2, random_state=42)


def sqrt_act(x):
    return x**(1/2)

def frt_act(x):
    return x**(1/4)

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

def create_node(input1, input2, input3, input4, name, trainable1, trainable2, trainable3, trainable4, act, bias):
    base = name
    n1 = base + "1"
    n2 = base + "2"
    n3 = base + "3"
    n4 = base + "4"
    an1 = CustomDense(1, 1, activation = 'linear', name=n1, trainable_weight=trainable1, trainable_bias = 0) (input1)
    an2 = CustomDense(1, 1, activation = 'linear', name=n2, trainable_weight=trainable2, trainable_bias = 0) (input2)
    an3 = CustomDense(1, 1, activation = 'linear', name=n3, trainable_weight=trainable3, trainable_bias = 0) (input3)
    an4 = CustomDense(1, 1, activation = 'linear', name=n4, trainable_weight=trainable4, trainable_bias=bias) (input4)
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
    
    bias_initial = keras.initializers.Zeros()

    trainable_list = []
    for i in range(nweight_terms):
        if (x[i+nact_terms] == 2):
            trainable_list.append(True)
        else:
            trainable_list.append(False)
    
    for i in range(nbias_terms):
        if (x[i+nact_terms+nweight_terms] == 2):
            trainable_list.append(True)
        else:
            trainable_list.append(False)

    input1 = Input(shape=(1,))
    input2 = Input(shape=(1,))
    input3 = Input(shape=(1,))
    input4 = Input(shape=(1,))

    a1 = create_node(input1, input2, input3, input4, "a1", trainable_list[0], trainable_list[1], trainable_list[2], trainable_list[3], act_dict[x[0]], trainable_list[20])
    a2 = create_node(input1, input2, input3, input4, "a2", trainable_list[4], trainable_list[5], trainable_list[6], trainable_list[7], act_dict[x[1]], trainable_list[21])
    a3 = create_node(input1, input2, input3, input4, "a3", trainable_list[8], trainable_list[9], trainable_list[10], trainable_list[11], act_dict[x[2]], trainable_list[22])
    a4 = create_node(input1, input2, input3, input4, "a4", trainable_list[12], trainable_list[13], trainable_list[14], trainable_list[15], act_dict[x[3]], trainable_list[23])

    output = create_node(a1, a2, a3, a4, "output", trainable_list[16], trainable_list[17], trainable_list[18], trainable_list[19], act_dict[x[4]], trainable_list[24])
   

    model = Model(inputs=[input1, input2, input3, input4], outputs=output)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
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
        
            if (model.layers[layer_list[i]].get_weights()[0].shape==(1,1)):
                model.layers[layer_list[i]].set_weights( [ np.array( [[ weight_dict[x[nact_terms+i]] ]] ),  np.array( [ bias_dict[x[nact_terms+nweight_terms+int((i+1)/5)]] ] ) ] )
            else:
                model.layers[layer_list[i]].set_weights( [ np.array( [ bias_dict[x[nact_terms+nweight_terms+int((i+1)/5)]] ] ),  np.array( [[ weight_dict[x[nact_terms+i]] ]] ) ] )
            
        else:
            model.layers[layer_list[i]].set_weights( [ np.array( [[ weight_dict[x[nact_terms+i]] ]] ), np.array( [0.] ) ] )

    

    return model, trainable_list



class ValidLossNaN(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if np.isnan(logs.get('loss')):
            self.model.stop_training=True
            

def train(model, train_inputs, train_outputs, verbose=False):
    mae_es= keras.callbacks.EarlyStopping(monitor='val_loss', patience=500,
                                          min_delta=1e-4, verbose=1, mode='auto', restore_best_weights=True)

    terminate = keras.callbacks.TerminateOnNaN()

    EPOCHS = 10000 # Number of EPOCHS
    history = model.fit([train_inputs[:,0], train_inputs[:,1], train_inputs[:,2], train_inputs[:,3]], train_outputs, epochs=EPOCHS,
                        shuffle=False, batch_size=len(train_inputs), verbose = False, callbacks=[terminate, mae_es, ValidLossNaN()],
                        validation_split=0.2)

def cv_error(individual, inputs, outputs):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    kf.get_n_splits(inputs)
    
    cv_mse_list = []
    

    for train_index, test_index in kf.split(inputs):
        new_model, trainable = create_model(individual)
        train_inputs, test_inputs = inputs[train_index], inputs[test_index]
        train_outputs, test_outputs = outputs[train_index], outputs[test_index]
        
        if (any(trainable) == True):
            try:
                train(new_model, train_inputs, train_outputs, verbose=False)
            except TypeError:
                print('Failed to train!')
                print('Bad Network:',individual)
                new_model, trainable = create_model(individual)

            wt_bs = new_model.get_weights()
            weight_list = []
            bias_list = []

            for weight in wt_bs:
                if (weight.shape == (1,1)):
                    weight_list.append(weight[0])
                else:
                    bias_list.append(weight[0])
                

                #handle nan weights
            if (np.isnan(weight_list).any()):
                cv_mse = 1e50
            elif (np.isnan(np.array(bias_list)).any()):
                cv_mse = 1e50
            else:
                cv_mse = new_model.evaluate([test_inputs[:,0],test_inputs[:,1],test_inputs[:,2],test_inputs[:,3]], test_outputs)

                if (np.isnan(cv_mse)):
                    cv_mse = 1e50
                else:
                    cv_mse = np.around(cv_mse,decimals=6)

            cv_mse_list.append(cv_mse)
        
        else:
            wt_bs = new_model.get_weights()
            weight_list = []
            bias_list = []

            for weight in wt_bs:
                if (weight.shape == (1,1)):
                    weight_list.append(weight[0])
                else:
                    bias_list.append(weight[0])

            if (np.isnan(np.array(weight_list)).any()):
                cv_mse=1e50
            elif (np.isnan(np.array(bias_list)).any()):
                cv_mse = 1e50

            else:
                cv_mse = new_model.evaluate([test_inputs[:,0],test_inputs[:,1],test_inputs[:,2],test_inputs[:,3]], test_outputs)

                if (np.isnan(cv_mse)):
                    cv_mse = 1e50
                else:
                    cv_mse = np.around(cv_mse,decimals=6)

            cv_mse_list.append(cv_mse)
        
        
        
    print(cv_mse_list)
    return np.mean(cv_mse_list)

def f3(w):
    return w

def objective_function(individual):
    mse_term = cv_error(individual, inputs, outputs)

    acts = individual[:nact_terms]
    actfunc_term = 0
    for i in range(nact_terms):
        actfunc_term += f3(acts[i])

    wtbs = individual[nact_terms:]
    wtbs_term = 0
    for j in range(nweight_terms+nbias_terms):
        wtbs_term += f3(wtbs[j])**2

    
    cplx_term = actfunc_term+wtbs_term

    obj = mse_term + 0.1*cplx_term
    
    print ("=============================================================================================================================")
    #print ("Trainable: ", trainable)
    print ("Individual: ", individual)
    print ("Objective function: ", mse_term, actfunc_term, wtbs_term, obj, flush=True)
    print ("=============================================================================================================================")    

    K.clear_session()
    tf.reset_default_graph()
    return (obj,) 


################### DEAP #####################
#create fitness class and individual class
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("map", futures.map)


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


toolbox.register("create_individual", custom_initRepeat, creator.Individual, random.randint, 
                 max1=3, max2=2, max3=2, n=nact_terms+nweight_terms+nbias_terms)
toolbox.register("population", tools.initRepeat, list, toolbox.create_individual)

cxpb = 0.5
mutpb = 0.3
ngens = 40

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
toolbox.register("mutate", custom_mutation, max1=3, max2=2, max3=2, indpb=mutpb)
toolbox.register("select", tools.selTournament, tournsize=15)
toolbox.register("evaluate", objective_function)

correct_individual = [2, 3, 3, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 2, 1, 1, 1, 0, 0, 0, 1, 0]

def main():
    random.seed(100000)
    population = toolbox.population(n=400)
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

