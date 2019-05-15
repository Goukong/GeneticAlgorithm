import numpy as np 
import matplotlib.pyplot as plt

DNA_SIZE = 10    #DNA length
POP_SIZE = 100   #the whole population size
CROSS_RATE = 0.8 #mating probability
MUTATION_RATE = 0.003
N_GENERATIONS = 200
X_BOUND = [0,5] #x axis upper and lower bounds

#the original func
def F(x):return np.sin(10*x)*x + np.cos(2*x)*x

#get fitness func
def get_fitness(pred):
	return pred+1e-3-np.min(pred)

def translateDNA(pop):
	#first convert binary to decimal
	pop = pop.dot(2**np.arange(DNA_SIZE)[::-1])
	#then normalize it to a range(0,5)
	pop = pop/float(2**DNA_SIZE-1) * X_BOUND[1]
	#finally it becomes the type [1,100]
	return pop

def select(pop,fitness):
	idx = np.random.choice(POP_SIZE,POP_SIZE,True,fitness/fitness.sum())
	return pop[idx]

def crossover(parent,pop): #mating process(genes crossover)
	if np.random.rand() < CROSS_RATE:
		i_ = np.random.randint(0,POP_SIZE,size = 1)
		choose = np.random.randint(0,2,size = DNA_SIZE).astype(np.bool);
		parent[choose] = pop[i_,choose]
	return parent

def mutation(child):
	for e in child:
		if np.random.rand() < MUTATION_RATE:
			e = 1 if e == 0 else 0
	return child


pop = np.random.randint(0,2,size=(POP_SIZE,DNA_SIZE))

plt.ion()
x = np.linspace(X_BOUND[0],X_BOUND[1],200)
plt.plot(x,F(x))

for _ in range(N_GENERATIONS):
	f_value = F(translateDNA(pop))
	
	if 'sca' in globals():sca.remove()
	sca = plt.scatter(translateDNA(pop),f_value,s=200,lw=0,c='pink',alpha=0.5);plt.pause(0.05)

	fitness = get_fitness(f_value)

	pop = select(pop,fitness)

	pop_copy = pop.copy()
	for parent in pop:
		child = crossover(parent,pop)
		child = mutation(child) #???????? what if when crossover fails but mutate success
		parent = child


plt.ioff();plt.show()
