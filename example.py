import pypooling
import theano
import theano.tensor as T
import numpy as np


arr = np.matrix([[1.74,1.88,-2.5, 3.4, 1.22], [1.55,1.88,-1.3,-1.2,0.74], [7.44, 2.5,2.33, 1.77,-1.11], [2.4,2.3,1.1,1.2,1.2], [2.5,2.6,2.7,2.3,-1.1]])
stochastic = pypooling.StochasticPooling()
value = T.matrix('X')
result = stochastic.fit(value, 3)
func = theano.function([value], result)
print(func(arr))
