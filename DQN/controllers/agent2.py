# -*- coding: utf-8 -*-
import random

import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

from keras.models import model_from_json

from .database import Database
from .env import Env

import tensorflow as tf
from matplotlib import pyplot

EPISODES = 5000

class DQNAgent:
    def __init__(self):

        self.rewards = [120,10,1]
        self.env = Env()
        self.db = Database()
        self.state_size = self.env.ACTION_SPACE
        self.action_size = self.env.ACTION_SPACE
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.loadModel()

    """Huber loss for Q Learning
    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(50, input_dim=self.state_size, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
       # if np.random.rand() <= self.epsilon:
        #    return random.randrange(self.action_size)

        state = np.reshape(state, [1, self.action_size])
        act_values = self.model.predict(state)

        ##print("act_values :: ")
        ##print(act_values)
        actual_act_values=[]

        for i in range(len(state[0])):
            if state[0][i] == 1:
                actual_act_values.append(act_values[0][i])
            else:
                actual_act_values.append(-100000000)
                act_values[0][i]=0

        ##print("actual act_values after removing zeros:: ")
        ##print(act_values)
        print(actual_act_values)

        #f = open("Q_history", "a")
        #f.write(str(actual_act_values))
        #f.close()
        return np.argmax(actual_act_values)  # returns action

    def replay(self, batch_size):
        minibatch = []
        loss=[]
        for i in range(0, 80):
           minibatch.append(self.memory[i])

        #minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, [1, self.action_size])
            next_state = np.reshape(next_state, [1, self.action_size])

            #print("### ramadan ###")
            #print(state)
            #print(reward)
            #print(action)
            target = self.model.predict(state)
            #print(target)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            history=self.model.fit(state, target, epochs=1, verbose=0)
            loss.append(history.history["loss"])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        #print("Loss is ")
        #print(loss)
        return loss
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def get_q_table(self):
        for i in range(len(self.env.R)):
            self.act(self.env.R[i,].tolist()[0])

    def learning(self):

        done = False

        batch_size = 80

        #self.env.reset()
        #self.env.state = [1,1,1]

        state = self.env.state[:]


        actions = self.env.available_actions()

        #print("state is ")
        #print(state)

        #print("available actions :: ")

        #print(actions)

        for i in range(len(actions)):
            # env.render()
            #action = self.act(state)

            next_action = actions[i]

            orders = self.get_query_conditions_order(np.asarray(self.env.state, dtype=np.float32), next_action)

            #print("orders")
            #print(orders)
            next_state, reward,done =self.env.step(next_action, orders)

            #print("reward")
            #print(reward)

            state_reshaped = np.reshape(state, [1, self.action_size])
            #print(state_reshaped)

            target = self.model.predict(state_reshaped)

            target[0][next_action] = reward

            #print(target)
            self.model.fit(state_reshaped, target, epochs=1, verbose=0)
            #next_state = np.reshape(next_state, [1, state_size])
            self.remember(state, next_action, reward, next_state, done)
            #state = next_state
            #if done:
            self.update_target_model()
              #  break
            #if len(self.memory) > batch_size:
             #  self.replay(batch_size)

            #self.update_target_model()
    def test_learning(self):

        state = [1,1,1]

        next_state, reward = [0,1,1],self.rewards[0]

        # next_state = np.reshape(next_state, [1, state_size])
        self.remember(state, 0, reward, next_state,1)

        self.update_target_model()



        state = [1, 1, 1]

        next_state, reward = [1, 0, 1], self.rewards[1]

        # next_state = np.reshape(next_state, [1, state_size])
        self.remember(state, 1, reward, next_state, 1)

        self.update_target_model()

        state = [1, 1, 1]

        next_state, reward = [1, 1, 0], self.rewards[2]

        # next_state = np.reshape(next_state, [1, state_size])
        self.remember(state, 2, reward, next_state, 1)

        self.update_target_model()

        self.replay(7)

    def start_learning(self):
            for i in range(len(self.env.R)):
                self.env.reset(i)
                self.learning()
                #self.env.reset(i)

    def generate_greedy_query_conditions_order(self):
        max_Q_values_vs_optimizer = np.matrix(np.zeros([self.env.OBSERVATION_SPACE, 2]))
        for i in range(1,len(self.env.R)):
           #optimzer_order = self.get_optimzer_order(self.env.R[i,].tolist()[0])

            order = self.get_state_query_conditions_order(self.env.R[i,].tolist()[0])

           # print("order for state")
            #print(self.env.R[i,].tolist()[0])
            #print(optimzer_order)
            #print(order)


            optimzer_action = self.env.optimzer_take_action(order)

            greedy_action = self.env.take_action(order)
            print("greedy")
            print(greedy_action)
            print("optimizer")
            print(optimzer_action)
            max_Q_values_vs_optimizer[i, 0] = float("{0:.2f}".format(greedy_action))
            max_Q_values_vs_optimizer[i, 1] = float("{0:.2f}".format(optimzer_action))


           # self.max_Q_values_vs_optimizer[i, 0] = greedy_action
            #self.max_Q_values_vs_optimizer[i, 1] = optimizer_time

        return max_Q_values_vs_optimizer

    def get_state_query_conditions_order(self,state):

        action = self.act(state)

        return self.get_query_conditions_order(state,action)

    def get_query_conditions_order(self, state, action):

        current_conditions_order = [action]

        # self.env.state = self.env.get_current_state()

        state[action] = 0
        #print("get_query_conditions_order222")

        actions = self.env.available_actions_state(state)
        #print(actions)
        #print(state)
        #print(self.env.state)

        for i in range(len(actions)):
            action = self.act(state)
            state[action] = 0
            current_conditions_order.append(action)
        #print("get_query_conditions_order")
        #print(current_conditions_order)
        return current_conditions_order

    def get_optimzer_order(self,state):
        actions = self.env.available_actions_state(state)
        return actions


    def saveModel(self):
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        #print("Saved model to disk")

    def loadModel(self):
        #json_file = open('model.json', 'r')
        #loaded_model_json = json_file.read()
        #json_file.close()
        #self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("model.h5")
        #print("Loaded model from disk")
        self.update_target_model()

    def plotMetrics(self,data):

        pyplot.plot(data)
        pyplot.show()