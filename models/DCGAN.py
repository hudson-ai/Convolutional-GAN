import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv
from theano.tensor.nnet import conv2d, relu
from six.moves import cPickle
import scipy.signal

from lib.utils import log_progress
from lib.layers import FullyConnected, ConvLayer
from lib.activations import batchnorm, lrelu
from lib.optimizers import RMSprop, Adam, gradDesc

class Generator(object):
    #https://arxiv.org/pdf/1511.06434.pdf
    def __init__(self, input, params = None, 
                 rng = np.random.RandomState(),
                 zsize = 100):
        self.input = input
        
        h_input = input
        
        h = FullyConnected(
            input=h_input,
            n_in=zsize,
            n_out=4*4*1024,
            W = params[0] if params is not None else None,
            b = params[1] if params is not None else None,
            rng=rng
        )
        h_out = relu(batchnorm(h.output.reshape((input.shape[0],1024,4,4))))
        
        conv1 = ConvLayer(h_out, 4, 8, 1024, 512,
                          rng = rng,
                          W = params[2] if params is not None else None
                         ) 
        conv1_out = relu(batchnorm(conv1.output))
        
        conv2 = ConvLayer(conv1_out, 8, 16, 512, 256,
                          rng = rng,
                          W = params[3] if params is not None else None
                         ) 
        conv2_out = relu(batchnorm(conv2.output))
        
        conv3 = ConvLayer(conv2_out, 16, 32, 256, 128,
                          rng = rng,
                          W = params[4] if params is not None else None
                         ) 
        conv3_out = relu(batchnorm(conv3.output))
        
        conv4 = ConvLayer(conv3_out, 32, 64, 128, 3,
                          rng = rng,
                          W = params[5] if params is not None else None
                         ) 
        conv4_out = T.tanh(conv4.output)
        
        self.output = conv4_out
        self.params = h.params + conv1.params + conv2.params + \
                conv3.params + conv4.params 
				
				
class Critic(object):
    #https://arxiv.org/pdf/1511.06434.pdf
    def __init__(self, input, params = None, 
                 rng = np.random.RandomState()):
        self.input = input
        
        conv1 = ConvLayer(input,64,32,3,128,
                          rng = rng,
                          W = params[0] if params is not None else None
                         ) 
        conv1_out = lrelu(conv1.output)
        
        conv2 = ConvLayer(conv1_out, 32, 16, 128, 256,
                          rng = rng,
                          W = params[1] if params is not None else None
                         ) 
        conv2_out = lrelu(batchnorm(conv2.output))
        
        conv3 = ConvLayer(conv2_out, 16, 8, 256, 512,
                          rng = rng,
                          W = params[2] if params is not None else None
                         ) 
        conv3_out = lrelu(batchnorm(conv3.output))
        
        conv4 = ConvLayer(conv3_out, 8, 4, 512, 1024,
                          rng = rng,
                          W = params[3] if params is not None else None
                         ) 
        conv4_out = lrelu(batchnorm(conv4.output))
        
        h_input = conv4_out.flatten(2)
        h = FullyConnected(
            input=h_input,
            n_in=1024*4*4,
            n_out=1,
            W = params[4] if params is not None else None,
            b = params[5] if params is not None else None,
            rng=rng
        )
        h_out = T.nnet.sigmoid(h.output)
        
        self.output = h.output
        self.params = conv1.params + conv2.params + \
                conv3.params + conv4.params + h.params
				

class DCGAN(object):
    def __init__(self,
                 genParams = None, critParams = None,
                 zsize = 100,
                 rng = np.random.RandomState(),
                 critLosses = [],
                 genLosses = []
                ):
        self.rng = rng
        self.genParams = genParams
        self.critParams = critParams
        self.zsize = zsize
        
        self.critLosses = critLosses
        self.genLosses = genLosses
        
    def train(self, X, iters = 600000,
              alpha = .00005,
              momentum = .9, m = 64, ncrit = 5,
              optimizer = "Adam",
              verbose = False, runningSave = False,
              runningFile="Running Save/DCGAN"):
        
        print("Building model...")
         # allocate symbolic variables for the data
        z = T.matrix('z')  
        x = T.tensor4('x')
        
        #Generator setup
        gen = Generator(z, zsize=self.zsize, params=self.genParams, rng=self.rng)
        self.gen = gen
        self.genParams = gen.params
        
        #Critic setup
        critTarget = Critic(x, params=self.critParams, rng=self.rng)
        self.critTarget = critTarget
        #critTrue params must be the same as critG params
        critG = Critic(gen.output, params=critTarget.params, rng=self.rng)
        self.critG = critG
        self.critParams = critTarget.params
        
        print("Building optimizers...")
        genLoss = T.mean(T.log(1-self.critG.output))
        critLoss = T.mean(T.log(self.critTarget.output)) \
                    + T.mean(T.log(1-self.critG.output))
        if optimizer == "Adam":
            genUpdates = Adam(genLoss, self.genParams, lr=alpha, b1=momentum)
            critUpdates = Adam(-critLoss, self.critParams, lr=alpha, b1=momentum)
        elif optimizer == "RMSprop":
            genUpdates = RMSprop(genLoss, self.genParams, lr=alpha, rho=momentum)
            critUpdates = RMSprop(-critLoss, self.critParams, lr=alpha, rho=momentum)
        else:
            genUpdates = gradDesc(genLoss, genParams, lr=alpha)
            critUpdates = gradDesc(-critLoss, critParams, lr=alpha)
            optimizer = "Vanilla SGD"
        print("Using " + optimizer)
        
        #Training functions
        trainGen = theano.function(
            inputs = [z],
            outputs = genLoss,
            updates = genUpdates
        )
        trainCrit = theano.function(
            inputs = [z,x],
            outputs = critLoss,
            updates = critUpdates
        )
        print("Done!")
        
        print("\nBegin training...")
        def printIter(i):
             print("Iteration %i/%i (%.2f%%)" % 
                         (
                            i+1,
                            iters,
                            100*(i+1)/iters,
                         )
                     )
                
        #Begin training
        for i in log_progress(range(iters), every = 1, name = "Iteration"):
            #Print stuff
            if i == 0 or verbose or (i+1) % 500 == 0 or iters <= 500:
                printIter(i)
            if (i+1) % 500 == 0:
                #Plot loss
                self.plot()
                #Show images
                import itertools
                k = 10
                Z = np.random.uniform(-1,1,(k*k,100))
                Z = Z.astype("float32")
                gX = self.sample(None, Z)
                fig, axes = plt.subplots(k, k, figsize=(20,20))
                for row, col in itertools.product(range(k), range(k)):
                    gx = gX[row*k+col]
                    axes[row,col].get_yaxis().set_visible(False)
                    axes[row,col].get_xaxis().set_visible(False)
                    axes[row,col].imshow(rescale((gx.transpose(1,2,0)), invert=True).astype('uint8'))
                    fig.subplots_adjust(hspace=0, wspace=0)
                plt.show()
                #Save
                if runningSave:
                    self.save(runningFile + str(i+1))
                    
            #Update parameters
            for n in range(ncrit):
                #Generate critic samples:
                Zm = self.rng.uniform(low = -1, high = 1, 
                  size=(m,self.zsize)).astype(theano.config.floatX)
                Xm = X[self.rng.randint(X.shape[0], size=m)]
                
                #Update critic
                critLoss = trainCrit(Zm, Xm)
                
            #Generate generator samples:
            Zm = self.rng.uniform(low = -1, high = 1, 
              size=(m,self.zsize)).astype(theano.config.floatX)
            
            #Update generator
            genLoss = trainGen(Zm)
            self.genLosses.append(genLoss)
            self.critLosses.append(critLoss)
        self.plot()
    
    def sample(self, n=1, Z = None):
        if Z is None:
            Z = self.rng.uniform(low = -1, high = 1, 
                      size=(n,self.zsize)).astype(theano.config.floatX)
        
        gen = Generator(Z, zsize=self.zsize, params=self.genParams, rng=self.rng)
        
        return(gen.output.eval())
    
    def plot(self, kernel_size=101):
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(self.critLosses)
        plt.title("Critic loss " + str(self.critLosses[-1]))
        plt.xlabel("Generator iterations")
        plt.figure()
        plt.plot(self.genLosses)
        plt.title("Generator loss " + str(self.genLosses[-1]))
        plt.xlabel("Generator iterations")
        plt.show()
    
    def save(self, filename="DCGAN"):
        save_file = open("../saves/" + filename, 'wb')  # this will overwrite current contents
        tosave = [self.genParams, self.critParams, self.rng, self.critLosses, self.genLosses]
        for param in tosave:
            # the -1 is for HIGHEST_PROTOCOL
            cPickle.dump(param, save_file, -1)  
        save_file.close()
    
    def from_file(filename="DCGAN"):
        f = open("../saves/" + filename, 'rb')
        genParams = cPickle.load(f)
        critParams = cPickle.load(f)
        rng = cPickle.load(f)
        critLosses = cPickle.load(f)
        genLosses = cPickle.load(f)
        clf = DCGAN(
            genParams=genParams,
            critParams=critParams,
            rng = rng,
            critLosses=critLosses,
            genLosses=genLosses
        )
        return(clf)