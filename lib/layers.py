import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv
from theano.tensor.nnet import conv2d, relu

def deconv(X, w, subsample=(1, 1), border_mode=(0, 0), conv_mode='conv'):
    #https://github.com/Newmu/dcgan_code/lib/ops.py
    from theano.sandbox.cuda.basic_ops import (gpu_contiguous, gpu_alloc_empty)
    from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConvGradI
    """ 
    sets up dummy convolutional forward pass and uses its grad as deconv
    currently only tested/working with same padding
    """
    img = gpu_contiguous(X)
    kerns = gpu_contiguous(w)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample,
                          conv_mode=conv_mode)(gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1]).shape, kerns.shape)
    out = gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1])
    d_img = GpuDnnConvGradI()(kerns, img, out, desc)
    return d_img
	
class ConvLayer(object):
    #https://github.com/mikesj-public/dcgan-autoencoder
    #Output must be an integer multiple of input, please use powers of 2
    def __init__(self, input,
                 input_size, output_size, 
                 num_input_filters, num_output_filters, 
                 W = None, filter_size = 5, 
                 activation = None,
                 rng = np.random.RandomState()):
        self.input = input
        
        is_deconv = output_size >= input_size

        #Size of 4d convolution tensor
        w_size = np.array([num_input_filters, num_output_filters, filter_size, filter_size]) \
                if is_deconv \
                else np.array([num_output_filters, num_input_filters, filter_size, filter_size])
        #Initialize weights
        if W == None:
            W_values = np.asarray(
                rng.normal(
                    scale = .02,
                    size = (w_size)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=W_values, name='W', borrow=True)
        self.W = W
        
        conv_method = deconv if is_deconv else conv2d

        #Size of subsampling
        sub = output_size / input_size \
            if is_deconv else input_size / output_size
        sub = int(sub)
        #Border size
        if filter_size == 3:
            bm = 1
        else:
            bm = 2
            
        #Output
        lin_output = conv_method(input, W, subsample=(sub, sub), border_mode=(bm, bm))
        if activation is not None:
            output = activation(lin_output)
        else: output = lin_output
        self.output = output
        
        #Model parameters
        self.params = [self.W] 
		
class FullyConnected(object):
    #http://deeplearning.net/tutorial/mlp.html
    def __init__(self, input, n_in, n_out, W = None, b = None,
                 activation = None, rng = np.random.RandomState()):
        
        self.input = input
        
        if W is None:
            W_values = np.asarray(
                rng.normal(
                    scale = .02,
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=W_values, name='W', borrow=True)
        else: self.W = W

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        else: self.b = b

        self.W = W
        self.b = b
        
        # parameters of the model
        self.params = [self.W, self.b]
        
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
		
