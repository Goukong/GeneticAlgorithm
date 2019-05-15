import numpy as np
import matplotlib.pyplot as plt

POP_SIZE = 20
DNA_SIZE = 10
DNA_BOUND = [0,2]
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.01
N_GENERATIONS = 200
X_BOUND = [0,5]

def F(x): return np.sin(10*x)*x + np.cos(2*x)*x

class MGA(object):
	"""docstring for MGA"""
	def __init__(self, DNA_size,DNA_bound,crossover_rate,mutation_rate,pop_size):
		self.DNA_size = DNA_size
		self.DNA_bound = DNA_bound
		self.crossover_rate = crossover_rate
		self.mutation_rate = mutation_rate
		self.pop_size = pop_size

		self.pop = np.random.randint(0,2,size=(pop_size,DNA_size))

	def translateDNA(self,pop):
		pop = pop.dot(2**np.arange(self.DNA_size)[::-1])
		pop = pop/float(2**self.DNA_size-1)*X_BOUND[1]
		return pop

	def get_fitness(self,value):
		return value

	def crossover(self,loser_winner):
		#比以往的crossover增大了交叉的概率，我觉得这种才是正确的
		cross_idx = np.empty((self.DNA_size,)).astype(np.bool)
		for i in range(self.DNA_size):
			cross_idx[i] = True if np.random.rand() < self.crossover_rate else False
		loser_winner[0,cross_idx] = loser_winner[1,cross_idx]
		return loser_winner

	def mutate(self,loser_winner):

		for point in loser_winner[0]:
			if np.random.rand() < self.mutation_rate:
				point = 1 if point == 0 else 0
		return loser_winner

	def evolve(self,n):
		for _ in range(n):
			pop = self.pop
			idx = np.random.choice(np.arange(0,self.pop_size),size = 2,replace=False)
			sub_pop = pop[idx]
			fitness = self.get_fitness(F(self.translateDNA(sub_pop)))
			loser_winner_idx = np.argsort(fitness)
			loser_winner = sub_pop[loser_winner_idx]
			loser_winner = self.crossover(loser_winner)
			loser_winner = self.mutate(loser_winner)
			self.pop[idx] = loser_winner 
		DNA_prod = self.translateDNA(self.pop)
		pred = F(DNA_prod)
		return DNA_prod,pred

if __name__ == '__main__':
	plt.ion()       # something about plotting
	x = np.linspace(*X_BOUND, 200)
	plt.plot(x, F(x))
	ga = MGA(DNA_size=DNA_SIZE, DNA_bound=DNA_BOUND, crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE, pop_size=POP_SIZE)
	for _ in range(N_GENERATIONS):
		DNA_prod,pred = ga.evolve(5)
		if 'sca' in globals(): sca.remove()
		sca = plt.scatter(DNA_prod, pred, s=200, lw=0, c='red', alpha=0.5); plt.pause(0.05)
	plt.ioff();plt.show()