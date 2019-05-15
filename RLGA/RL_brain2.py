import numpy as np 
import pandas as pd 

class QLearningTable:
	def __init__(self,POP_SIZE,learning_rate=0.02,reward_decay=0.9,e_greedy=0.6):
		#actions = [up,down,left,right]
		self.actions = np.array(range(POP_SIZE))
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy
		self.q_table = pd.DataFrame(columns=self.actions,dtype=np.float64)

	#根据现在的位置选择下一步的action
	def choose_action(self,observation):
		#检查当前状态是否存在，不存在的话创建
		self.check_state_exist(observation)
		#“贪婪”选择方法
		if np.random.uniform() < self.epsilon:
			state_action = self.q_table.loc[observation,:]
			action = np.random.choice(state_action[state_action == np.max(state_action)].index)
		else:
			action = np.random.choice(self.actions)
		return action

	#params means state,action,reward,new_state
	def learn(self,s,a,r,s_):
		self.check_state_exist(s_)
		q_predict = self.q_table.loc[s,a]
		if s_ != 'terminal':
			q_target = r + self.gamma * self.q_table.loc[s_,:].max()
		else:
			q_target = r 
		self.q_table.loc[s,a] += self.lr * (q_target - q_predict)

	#如果能在行索引中找到state的名称，那么将会被找到，但是如果找不到就添加该状态进入q表
	def check_state_exist(self,state):
		if state not in self.q_table.index:
			self.q_table = self.q_table.append(
				pd.Series(
					[0]*len(self.actions),
					index = self.q_table.columns,
					name = state,
					)
				)



