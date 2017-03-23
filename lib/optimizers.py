import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv
from theano.tensor.nnet import conv2d, relu

def Adam(cost, params, lr=0.0002, b1=0.9, b2=0.999, e=1e-8, c=None):
    #https://gist.github.com/Newmu/acb738767acb4788bac3
    #Standard literature says b1=.9
    #DCGAN paper says b1 = .5
    b1 = 1-b1
    b2 = 1-b2
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(np.float32(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t) if c is None else T.clip(p - (lr_t * g_t), -c, c)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates
	
def RMSprop(cost, params, lr=0.00005, rho=0.99, epsilon=1e-8, c=None):
    #https://github.com/Newmu/Theano-Tutorials
    #rho = .99 is torch default, used in paper
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        update = p - lr * g if c is None else T.clip(p - lr * g, -c, c)
        updates.append((p, update))
    return updates
	
def gradDesc(cost, params, lr=.00005, c = None):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        update = p - lr * g if c is None else T.clip(p - lr * g, -c, c)
        updates.append((p, update))
    return updates