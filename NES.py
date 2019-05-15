#Natural Evolution Strategy
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
#???
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance

DNA_SIZE = 1
N_POP = 100
N_GENERATION = 500
LR = 0.01
DNA_BOUND = [0,5]

def F(x): return np.sin(x) #np.sin(10*x)*x + np.cos(2*x)*x 

def get_fitness(pred):
	return pred.flatten()

mean = tf.Variable(tf.random_normal([1,],2.5,1.),dtype=tf.float32)
cov = tf.Variable(tf.eye(DNA_SIZE),dtype = tf.float32)
mvn = MultivariateNormalFullCovariance(loc = mean,covariance_matrix = cov)
make_kid = tf.clip_by_value(mvn.sample(N_POP),0,5)
tfkids_fit = tf.placeholder(tf.float32, [N_POP, ])
tfkids = tf.placeholder(tf.float32, [N_POP, DNA_SIZE])
loss = -tf.reduce_mean(mvn.log_prob(tfkids)*tfkids_fit)         # log prob * fitness
train_op = tf.train.GradientDescentOptimizer(LR).minimize(loss) # compute and apply gradients for mean and cov

sess = tf.Session()
sess.run(tf.global_variables_initializer()) # initialize tf variables

plt.ion()
x = np.linspace(*DNA_BOUND,200)
plt.plot(x,F(x))

for g in range(N_GENERATION):
	kids = sess.run(make_kid)
	kids_fit = get_fitness(F(kids))
	sess.run(train_op,{tfkids_fit:kids_fit,tfkids:kids})

	if 'sca' in globals(): sca.remove()
	sca = plt.scatter(kids[:], F(kids[:]), s=30, c='k');plt.pause(0.01)

