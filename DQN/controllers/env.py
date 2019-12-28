import numpy as np
import itertools

from .database import Database



class Env:

    def __init__(self):

        self.db = Database()


        self.ACTION_SPACE =  len(self.db.conditions)

        self.OBSERVATION_SPACE = 2 ** self.ACTION_SPACE

        self.R = np.matrix(list(sorted(itertools.product([0, 1], repeat=self.ACTION_SPACE),
                                       key=lambda x: (sum(x), x))))

        self.state = self.R[0,].tolist()[0]

        #self.state=[1,1,1]


    def step(self, action,orders):

        reward = self.take_action(orders)

        state = self.state[:]
        state[action]=0

        #state_number = self.get_state(self.state)

        return state,reward,1 #,done ,{}

    def reset(self,state_number):
        self.state = self.R[state_number,].tolist()[0]
        #print("reset")
        #print(self.state)

    def render(self, mode='human', close=False):
        pass

    def available_actions(self):
        current_state_row = self.R[self.get_state(self.state),]
        #print(current_state_row)
        av_act = np.where(current_state_row > 0)[1]
        return av_act

    def available_actions_state(self,state):
        current_state_row = self.R[self.get_state(state),]
        #print(current_state_row)
        av_act = np.where(current_state_row > 0)[1]
        return av_act

    def take_action(self, orders ):

        #conditions_order = self.get_query_conditions_order(action)

        conditions_order = orders

        print("hhh")
        print(orders)
        query = self.db.make_query(conditions_order)
        print("greedy query")
        print(query)

        new_value = -1 * self.db.get_query_response_time(query)

        return new_value

    def optimzer_take_action(self, conditions_order ):

        query = self.db.make_query(conditions_order,0)
        print("optimzer")
        print(query)
        new_value = -1 * self.db.get_query_response_time(query)

        return new_value

    def get_state(self, row):
        return np.where((self.R == row).all(axis=1))[0][0]

    def get_current_state(self):
        return self.state[:]



