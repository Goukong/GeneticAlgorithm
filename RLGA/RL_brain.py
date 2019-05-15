import numpy as np 
import pandas as pd 

class QLearningTable:
	def __init__(self,POP_SIZE,learning_rate=0.02,reward_decay=0.9,e_greedy=0.6):
		
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy

		#actions = [0,1,2,...,99]
		actions = np.array(range(POP_SIZE))
		self.actions = actions

		#create q_table
		self.q_table = pd.DataFrame(columns=actions,dtype=np.float64)
		self.q_table = self.q_table.append(
				pd.Series(
						[0]*len(self.actions),
						index = self.q_table.columns,
						name = 'start',
					)
			)
		'''
			q_table
			  action 0 1 2 3 ... 99
			state
		 	   start x x 0 0 ... x 
		'''

	#选择下一步的action
	def choose_action(self):
		#“贪婪”选择方法
		if np.random.uniform() < self.epsilon:
			#选择第一行的所有数据
			state_action = self.q_table.loc['start',:]
			#随机的选择，是为了有多个最大值的情况下也要保持随机性
			action = np.random.choice(state_action[state_action == np.max(state_action)].index)
		else:
			action = np.random.choice(self.actions)
		return action

	#不用判断了，因为都是一步到位
	def learn(self,a,r):
		q_predict = self.q_table.loc['start',a]
		q_target = r
		self.q_table.loc['start',a] += self.lr * (q_target - q_predict)




