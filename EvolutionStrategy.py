import numpy as np 
import matplotlib.pyplot as plt 

DNA_SIZE = 1 # 原来的二进制其实只对应一个实数而已
DNA_BOUND = [0,5]
N_GENERATIONS = 200
POP_SIZE = 100
N_KID = 50   # ???


def F(x): return np.sin(10*x)*x + np.cos(2*x)*x 

def get_fitness(pred):
	#将100个不同的个体压缩成一维
	return pred.flatten()

def make_kid(pop,n_kid):
	kids = {'DNA':np.empty((n_kid,DNA_SIZE))}
	kids['mut_strength'] = np.empty_like(kids['DNA'])
	for kv,ks in zip(kids['DNA'],kids['mut_strength']):
		p1,p2 = np.random.choice(np.arange(POP_SIZE),size=2,replace=False)
		cp = np.random.randint(0,2,size=DNA_SIZE,dtype=np.bool)
		kv[cp] = pop['DNA'][p1,cp]
		kv[~cp] = pop['DNA'][p2,~cp]
		ks[cp] = pop['mut_strength'][p1,cp]
		ks[~cp] = pop['mut_strength'][p2,~cp]

		#mutate
		#变异强度本身的变异，令其大于0可以更快，但是不懂原理、、
		ks[:] = np.maximum(ks+(np.random.randn(*ks.shape)-0.5),0.0) #0.5????难道是让收敛更快？是的
		kv += ks * np.random.randn(*kv.shape)
		kv = np.clip(kv,*DNA_BOUND)
		
	return kids 

def kill_bad(pop,kids):
	#把原一代和新一代连接起来
	for key in ['DNA','mut_strength']:
		pop[key] = np.vstack((pop[key],kids[key]))
	fitness = get_fitness(F(pop['DNA']))
	idx = np.arange(pop['DNA'].shape[0])#POP_SIZE + n_kids
	good_idx = idx[fitness.argsort()][-POP_SIZE:]#要最大的POP_SIZE个,多余的为空，在赋值的时候直接舍去
	
	for key in ['DNA','mut_strength']:
		pop[key] = pop[key][good_idx]#已经kill掉n_kids的数量
	pop['DNA'] = np.clip(pop['DNA'],*DNA_BOUND)
	return pop

#rand() 返回【0，1）所有值都是随机时，收敛会快一些，本例快十代左右
pop = dict(DNA=5*np.random.rand(POP_SIZE,DNA_SIZE),
	mut_strength=np.random.rand(POP_SIZE,DNA_SIZE))
 

plt.ion()
x = np.linspace(*DNA_BOUND,200)
plt.plot(x,F(x))

for i in range(N_GENERATIONS):
	if 'sca' in globals():sca.remove()
	sca = plt.scatter(pop['DNA'],F(pop['DNA']),s=200,lw=0,c='red',alpha=0.5);plt.pause(0.05)
	if 'text' in globals():text.remove()
	text = plt.text(0,5,'times:%s'%i,fontsize=40)
	kids = make_kid(pop,N_KID)
	pop = kill_bad(pop,kids)

plt.ioff();plt.show()