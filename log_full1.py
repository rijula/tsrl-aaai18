import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
#from load import mnist
from theano import sparse 

srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def RMSprop(cost, params, lr=0.01, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates
def sgd(cost, params, lr=0.1):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates
def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def model(X, w_o, p_drop_input, p_drop_hidden):
    #X = dropout(X, p_drop_input)
    #h = rectify(sparse.basic.dot(X, w_h))
    #h = T.nnet.sigmoid(sparse.basic.dot(X, w_h))

    #h = dropout(h, p_drop_hidden)
    #h2 = T.dot(h, w_h2)

    #h2 = dropout(h2, p_drop_hidden)
    py_x = softmax(sparse.basic.dot(X, w_o))
    return  py_x

#trX, teX, trY, teY = mnist(onehot=True)

X = sparse.csr_matrix(name='X', dtype='float64')
Y = T.fmatrix()
mask=T.fmatrix()
ind=T.ivector()
col=T.ivector()
#w_h = init_weights((16074140, 100))
#w_h2 = init_weights((625, 625))
w_o = init_weights((1000000, 159))
#w_o = init_weights((16074140, 107))
print "Weight initialization done..."
#w_o2=w_o[col]
#print w_o.get_value().shape
#print w_o2.get_value().shape
w_o1=w_o[col[:,None],ind]
#noise_h, noise_py_x = model(X, w_h, w_h2, w_o, 0.2, 0.5)

py_x = model(X,w_o1, 0., 0.)
new_py_x=T.mul(py_x,mask)
y_x = T.argmax(py_x, axis=1)
cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y)) + 0.1 * (w_o1 ** 2).sum()
params = [w_o1]
g = T.grad(cost, wrt=w_o1)

lr=0.1
#updates = T.inc_subtensor(w_o1, g*lr)
#updates = T.set_subtensor(w_o1, w_o1 + g*lr)

updates = [(w_o,T. set_subtensor(w_o1, w_o1-g*lr))]
train = theano.function(inputs=[X, Y,col,ind], outputs=cost, updates=updates, allow_input_downcast=True)

predict = theano.function(inputs=[X,col,ind], outputs=y_x, allow_input_downcast=True)

'''for i in range(100):
    print i
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    print np.mean(np.argmax(teY, axis=1) == predict(teX))'''
