import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class PyPooling:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, *args, **kwargs):
        raise NotImplemented


class MaxPooling2D:
    def fit(X):
        return T.signal.downsample.max_pool_2d(X)


class StochasticPooling:
    def __init__(self):
        pass

    def fit(self, X, num_regions, rng=None, *args, **kwargs):
        '''
        num_regions - sample this number from distriution
        Optional params
        regionX - size of region of axisX
        regionY - size of region of axisY
        '''
        if rng == None:
            rng = RandomStreams()
        shape = X.shape
        total_width = shape[0]
        total_height = shape[1]
        afterReLU = T.switch(X<0, 0, X)
        max_region = T.max(afterReLU)
        newX = afterReLU
        newX /= max_region
        #result = rng.multinomial(pvals=newX.reshape((total_width, total_height)), dtype='float32')
        return rng.multinomial(pvals=[10], size=(10,10), n=5)
