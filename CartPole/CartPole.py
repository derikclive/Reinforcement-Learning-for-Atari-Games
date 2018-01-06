
# coding: utf-8

# In[1]:


import gym
import random
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.models import Model
from keras.layers import BatchNormalization,Convolution2D,Conv2D, LeakyReLU,ELU,Input, UpSampling2D, Activation, merge, MaxPooling2D, Deconvolution2D, Reshape, Permute
from keras.optimizers import SGD, Adam, Nadam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
import tensorflow as tf

from statistics import median, mean
from collections import Counter

LR = 1e-3
env = gym.make("CartPole-v0")
env.reset()
goal_steps = 1000


# ## Objective
# 
# The idea of CartPole is that there is a pole standing up on top of a cart. The goal is to balance this pole by wiggling/moving the cart from side to side to keep the pole balanced upright.
# 
# The environment is deemed successful if we can balance for 200 frames, and failure is deemed when the pole is more than 15 degrees from fully vertical.
# 
# Every frame that we go with the pole "balanced" (less than 15 degrees from vertical), our "score" gets +1, and our target is a score of 200.

# In[2]:


def some_random_games_first():
    # Each of these is its own game.
    for episode in range(5):
        env.reset()
        # this is each frame, up to 200...but we wont make it that far.
        for t in range(200):
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            env.render()
            
            # This will just create a sample action in any environment.
            # In this environment, the action can be 0 or 1, which is left or right
            action = env.action_space.sample()
            
            # this executes the environment with an action, 
            # and returns the observation of the environment, 
            # the reward, if the env is over, and other info.
            observation, reward, done, info = env.step(action)
            if done:
                break
                
some_random_games_first()


# ## Creating the training Data
# Here we create the training data by playing the games using random actions(left/right) and storing the observations and the actions associated with each oberservation.

# In[9]:


def initial_population(model=False, num_games= 5000, score_requirement = 50):
    # [OBS, MOVES]
    training_data = []
    # all scores:
    scores = []
    # just the scores that met our threshold:
    accepted_scores = []
    # iterate through however many games we want:
    for _ in range(num_games):
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        # for each frame in 200
        for _ in range(goal_steps):
            # choose random action (0 or 1)
            if not model or len(prev_observation)==0:
                action = random.randrange(0,2)
            else:
                action = np.argmax(model.predict(prev_observation.reshape(-1,len(prev_observation)))[0])
            # do it!
            observation, reward, done, info = env.step(action)
            
            
            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score+=reward
            if done: break

        # IF our score is higher than our threshold, we'd like to save
        # every move we made
        # NOTE the reinforcement methodology here. 
        # all we're doing is reinforcing the score, we're not trying 
        # to influence the machine in any way as to HOW that score is 
        # reached.
        
        if score >= score_requirement:
            #print(score)
            accepted_scores.append(score)
            for data in game_memory:
                # convert to one-hot (this is the output layer for our neural network)
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                    
                # saving our training data
                training_data.append([data[0], output])

        # reset env to play again
        env.reset()
        # save overall scores
        scores.append(score)
    
    # just in case you wanted to reference later
    training_data_save = np.array(training_data)
    np.save('saved.npy',training_data_save)
    
    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))
    
    return training_data


# ## The neural network
# 
# Now we will make our neural network. We're just going to use a simple multilayer perceptron model. We use the keras API to build our neural network model.

# In[10]:


def network(input_size):
    model = Sequential()
    
    model.add(Dense(128, activation='relu', input_dim=input_size))
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(2, activation='softmax'))
    
    model.compile(loss='binary_crossentropy',optimizer=Adam(lr=LR),metrics=['accuracy'])
    return model

    


# In[11]:


def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]))
    y = [i[1] for i in training_data]

    print(X.shape)
    if not model:
        model = network(input_size = len(X[0]))
    
    model.fit(X, y, epochs=3, verbose=True)
    return model


# In[12]:


training_data = initial_population(False)
model = train_model(training_data)


# Here, we use the aldready trained model to predict what action to take next given an observation, instead of random actions. We then use these new obervations to further train our model to improve the policy.

# In[13]:


for i in range(3):
    training_data = initial_population(model, 100, 185)
    model = train_model(training_data, model)


# ## Testing our model

# In[14]:


scores = []
choices = []
for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()

        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs)))[0])

        choices.append(action)
                
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score+=reward
        if done: break

    scores.append(score)

print('Average Score:',sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))


# In[ ]:




