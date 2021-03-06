import numpy as np 
import matplotlib.pyplot as plt 

def F(x): return np.sin(10*x)*x + np.cos(2*x)*x 

DNA_SIZE = 1
DNA_BOUND = [0,5]
N_GENERATIONS = 100
MUT_STRENGTH = 10

def get_fitness(preb):return preb.flatten()

def make_kid(parent):
	#要以均值为中心变化
	kid = parent + MUT_STRENGTH * np.random.randn(DNA_SIZE)
	kid = np.clip(kid,*DNA_BOUND) 
	return kid

def kill_bad(parent,kid):
	global MUT_STRENGTH
	p_target = 1/5
	fp = get_fitness(F(parent))
	fc = get_fitness(F(kid))
	if fc > fp:
		ps = 1
		parent = kid
	else:
		ps = 0
	MUT_STRENGTH = MUT_STRENGTH * np.exp(1/3*(ps-p_target)/1-p_target)

	return parent

parent = np.random.rand(DNA_SIZE)*DNA_BOUND[1]
plt.ion()       # something about plotting
x = np.linspace(*DNA_BOUND, 200)

for _ in range(N_GENERATIONS):

	kid = make_kid(parent)
	py,ky = F(parent),F(kid)
	parent = kill_bad(parent,kid)
	plt.cla()
	plt.scatter(parent, py, s=200, lw=0, c='red', alpha=0.5,)
	plt.scatter(kid, ky, s=200, lw=0, c='blue', alpha=0.5)
	plt.text(0, -7, 'Mutation strength=%.2f' % MUT_STRENGTH)
	plt.plot(x, F(x)); plt.pause(0.05)
plt.ioff(); plt.show()