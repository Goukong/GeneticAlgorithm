#5月9日 实现了RLGA
#但是效果并不太好
#如果将分割变得少一些，会不会效果好一些
#还有可能是不是压根就效果不好
import numpy as np
import cv2
import TargetTracing as TT 
import RL_brain
import matplotlib.pyplot as plt 

DNA_SIZE = 6
POP_SIZE = 100
N_GENERATION = 200

DNA_BOUND1 = [0.01,0.1] # for qualityLevel
DNA_BOUND2 = [0,100] # for minDistance,winSize,maxlevel,COUNT
DNA_BOUND3 = [0,1] #for EPS


max_found = 0
max_param = []
last_reward = 0

def get_fitness(child):
	#translate the DNA into params
	feature_params = dict(
		maxCorners = 1000,
		qualityLevel = child[0],
		minDistance = child[1],
		blockSize = 7
		)
	lk_params = dict(
		winSize = (int(child[2]),int(child[2])),
		maxLevel = int(child[3]),
		criteria = (
			cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
			int(child[4]),child[5])
			)
	#get the tracing result from tragetTrace func as fitness
	result = TT.targetTrace(feature_params,lk_params) 
	return result
def getAllFitness(pop):
	#store results from targetTracing
	fitness = np.empty(POP_SIZE)
	#get the fitness
	for i in range(POP_SIZE):
		#translate the DNA into params
		feature_params = dict(maxCorners = 1000,
			qualityLevel = pop[i][0],
			minDistance = pop[i][1],
			blockSize = 7
			)
		lk_params = dict(winSize = (int(pop[i][2]),int(pop[i][2])),
			maxLevel = int(pop[i][3]),
			criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,int(pop[i][4]),pop[i][5])
			)
		#get the tracing result from tragetTrace func as fitness
		fitness[i] = TT.targetTrace(feature_params,lk_params) 
	return fitness

def createQ_table():
	#First we devide the dna into 6 parts
	#and create 6 q-table with 100 action choice 
	t1 = RL_brain.QLearningTable(POP_SIZE)
	t2 = RL_brain.QLearningTable(POP_SIZE)
	t3 = RL_brain.QLearningTable(POP_SIZE)
	t4 = RL_brain.QLearningTable(POP_SIZE)
	t5 = RL_brain.QLearningTable(POP_SIZE)
	t6 = RL_brain.QLearningTable(POP_SIZE)
	agents =[t1,t2,t3,t4,t5,t6]

	return agents

def RL(pop,agents):

	global max_found,max_param,last_reward
	
	#then choose each action and make a new child
	child = []
	#store the idx each table choose
	state = []
	for i,agent in enumerate(agents):
		idx = agent.choose_action()
		state.append(idx)
		current = pop[idx]
		child.append(current[i])

	#get reward 
	fitness = get_fitness(child)
	if fitness > max_found:
		max_found = fitness
		max_param = child
	reward = fitness - last_reward
	last_reward = fitness

	#caculate Q and update q_table 
	for i,agent in enumerate(agents):
		agent.learn(state[i],reward)
	
	return agents

def create():
	#create a module for DNA
	dna_mod = np.empty((POP_SIZE,DNA_SIZE))
	#each DNA has its own bound
	for i in range(POP_SIZE):
		#qualityLevel [0.01,0.1]
		dna_mod[i][0] = np.random.rand()/10
		
		#mindistance  [0,100] 
		dna_mod[i][1] = np.random.randint(0,100,1)
		
		#winSize  (2,100]
		dna_mod[i][2] = np.random.randint(3,100,1)
		
		#maxlevel,COUNT [0,100]
		dna_mod[i][3] = np.random.randint(0,100,1)
		dna_mod[i][4] = np.random.randint(0,100,1)
		
		#EPS [0,1]
		dna_mod[i][5] = np.random.rand()
	
	return dna_mod

if __name__ == '__main__':
	#global max_found,max_param
	pop = create()
	y = []
	x = list(range(200))
	agents = createQ_table()
	for _ in range(N_GENERATION):
		agents = RL(pop,agents)
		y.append(max_found)
		print("max match:",max_found)
		print("params:",max_param)
	plt.plot(x,y)
	lb = -1 
	for a,b in zip(x,y):
		if lb != b:
			plt.text(a,b+0.1,'%.0f'%b,ha = 'center',va = 'bottom',fontsize=7)
			lb = b
	plt.text(-1,max(y)+4,'max_param:%s'%str(max_param))
	plt.show()
	
	TT.setParamAndGetResult(max_param)