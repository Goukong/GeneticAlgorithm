#学习到了对于点集DNA形式的处理方式，以及计算点与点之间距离等numpy方式
#体会到了crossover和mutate针对特殊问题的特殊处理方式
#进一步熟悉了GA框架
import numpy as np 
import matplotlib.pyplot as plt 

N_CITIES = 20
POP_SIZE = 500
CROSS_RATE = 0.1
MUTATION_RATE = 0.02
N_GENERATIONS = 500

class GA(object):
	def __init__(self,DNA_size,cross_rate,mutation_rate,pop_size):
		self.DNA_size = DNA_size
		self.cross_rate = cross_rate
		self.mutation_rate = mutation_rate
		self.pop_size = pop_size

		self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(pop_size)])

	def translateDNA(self,DNA,city_position):
		line_x = np.empty_like(DNA,dtype = np.float64)
		line_y = np.empty_like(DNA,dtype = np.float64)
		for i,d in enumerate(DNA):
			city_coord = city_position[d]
			line_x[i,:] = city_coord[:,0]
			line_y[i,:] = city_coord[:,1]
		#line_x,line_y stores all points
		return line_x,line_y

	def get_fitness(self,line_x,line_y):
		#we just need to caculate the distance from the nearest points
		total_distance = np.empty((line_x.shape[0],),dtype=np.float64)
		for i,(xs,ys) in enumerate(zip(line_x,line_y)):
			total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs))+np.square(np.diff(ys))))
		fitness = np.exp(self.DNA_size*2/total_distance)
		return fitness,total_distance

	def select(self,fitness):
		idx = np.random.choice(np.arange(self.pop_size),size=self.pop_size,replace=True,p=fitness/fitness.sum())
		return self.pop[idx]

	def crossover(self,parent,pop):
		if np.random.rand() < self.mutation_rate:
			i_ = np.random.randint(0,self.pop_size,size=1)
			cross_points = np.random.randint(0,2,size=self.DNA_size).astype(np.bool)
			keepcity = parent[~cross_points]
			swapcity = pop[i_,np.isin(pop[i_].ravel(),keepcity,invert=True)]
			parent = np.concatenate((keepcity,swapcity))
		return parent

	def mutate(self,child):
		for point in range(self.DNA_size):
			if np.random.rand()<self.mutation_rate:
				swap_point = np.random.randint(0,self.DNA_size)
				swapA,swapB = child[point],child[swap_point]
				child[point],child[swap_point] = swapB,swapA
		return child

	def evolve(self,fitness):
		pop = self.select(fitness)
		pop_copy = pop.copy()
		for parent in pop:
			child = self.crossover(parent,pop_copy)
			child = self.mutate(child)
			parent = child
		self.pop = pop

class TravelSalesPerson(object):
    def __init__(self, n_cities):
        self.city_position = np.random.rand(n_cities, 2)
        plt.ion()

    def plotting(self, lx, ly, total_d):
        plt.cla()
        plt.scatter(self.city_position[:, 0].T, self.city_position[:, 1].T, s=100, c='k')
        plt.plot(lx.T, ly.T, 'r-')
        plt.text(-0.05, -0.05, "Total distance=%.2f" % total_d, fontdict={'size': 20, 'color': 'red'})
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.pause(0.01)


if __name__ == '__main__':
	ga = GA(DNA_size=N_CITIES,cross_rate = CROSS_RATE,mutation_rate=MUTATION_RATE,pop_size=POP_SIZE)
	env = TravelSalesPerson(N_CITIES)
	for generation in range(N_GENERATIONS):
		lx,ly = ga.translateDNA(ga.pop,env.city_position)
		fitness,total_distance = ga.get_fitness(lx,ly)
		ga.evolve(fitness)
		best_idx = np.argmax(fitness)
		print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx],)
		env.plotting(lx[best_idx], ly[best_idx], total_distance[best_idx])
	plt.ioff()
	plt.show()