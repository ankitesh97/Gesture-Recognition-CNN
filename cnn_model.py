import numpy as np
import json
import pickle
from img_preprocess import Preprocess
from scipy.optimize import minimize,fmin_tnc
import time
from joblib import Parallel, delayed
import multiprocessing

params_file  = open('params.json','r')
params = json.load(params_file)
params_file.close()
#architecture
#two black boxes i.e cnn->relu->max_pooling
#fully connected with two hidden layers
LAYER_1 = 0
LAYER_2 = 1
LAYER_3 = 2
start = time.time()


def backpropBodyParllel(self,X_pool,y_pool):

    weight_layer_3_grads = np.zeros(self.weights[LAYER_3].shape)
    weight_layer_2_grads = np.zeros(self.weights[LAYER_2].shape)
    weight_layer_1_grads = np.zeros(self.weights[LAYER_1].shape)
    kernel_layer_1_grads = np.zeros((self.n_filter_layer_1,self.filter_size_layer_1[0],self.filter_size_layer_1[1]))
    kernel_layer_2_grads = np.zeros((self.n_filter_layer_2,)+self.filter_size_layer_2)
    bias_layer_2_grads = np.zeros((self.n_filter_layer_2))
    bias_layer_1_grads = np.zeros((self.n_filter_layer_1))
    #get the batches
    J = 0
    for im_index in range(len(X_pool)):
        # print "im_index "+str(im_index)
        # raw_input()
        X_curr = X_pool[im_index]
        y_curr = y_pool[im_index]
        layer_1_conv_box, layer_1_pooling, layer_2_conv_box, layer_2_pooling, fully_connected  = self.feedForward(X_curr,y_curr,training=False) #out vector
        a_last_layer = fully_connected['a3']
        loss_curr = self.softmaxLoss(a_last_layer,y_curr)
        J += loss_curr
        #backprop in neural network

        # last layer error for softmax function and log likelihood
        d3 = (a_last_layer - y_curr).flatten()
        #update weight error matrix make a coulmn vector and multiply
        a2 = fully_connected['a2']
        weight_layer_3_grads += d3.reshape((d3.shape[0],1)) * a2
        #propogate the error
        z2 = fully_connected['z2']
        d2 = (np.dot(self.weights[LAYER_3].T,d3) * self.reluDerivative(z2))[1:] #350 X 1
        a1 = fully_connected['a1']
        weight_layer_2_grads += d2.reshape((d2.shape[0],1)) * a1
        # 1st hidden layer error
        z1 = fully_connected['z1']
        d1 = (np.dot(self.weights[LAYER_2].T,d2) * self.reluDerivative(z1))[1:]
        a0 = fully_connected['a0']
        weight_layer_1_grads += d1.reshape((d1.shape[0],1)) * a0
        z0 = fully_connected['z0']
        # error in the input layer i,e the layer after the convnet box
        d0 = (np.dot(self.weights[LAYER_1].T,d1) * self.reluDerivative(z0))[1:] #since the 1st is bias
        # now propogate to convnet layer
        # X_max_pooling_layer_2 this is nd matrix which contains last layer pixels
        # X_relu_layer_2 will contain relu layer pixels
        # max_x_pooling_layer_2,max_y_pooling_layer_2
        X_max_pooling_layer_2 = layer_2_pooling['pooling_val']
        X_relu_layer_2 = layer_2_conv_box['relu']
        d_pooling_layer_2 = d0.reshape(X_max_pooling_layer_2.shape)
        d_relu_layer_2 = np.zeros(X_relu_layer_2.shape)
        #for each channel
        max_x_pooling_layer_2 = layer_2_pooling['max_indexes_x']
        max_y_pooling_layer_2 = layer_2_pooling['max_indexes_y']
        for ch in range(d_relu_layer_2.shape[0]):
            d_relu_layer_2[ch,max_x_pooling_layer_2[ch],max_y_pooling_layer_2[ch]] = d_pooling_layer_2[ch]

        conv_layer_1_pooling_op = layer_1_pooling['pooling_val']
            #rotate dell and apply covolution with the previous layer output to get the errors
        conv_layer_1_pooling_op_shape = conv_layer_1_pooling_op[0].shape

        rotated_dell = np.flip(np.flip(d_relu_layer_2,-2),-1)
        bias_layer_2_grads = np.sum(rotated_dell,axis=(-1,-2))
        rotated_dell = rotated_dell.reshape((self.n_filter_layer_2,1,rotated_dell.shape[-2],rotated_dell.shape[-1]))
        conv_layer_1_pooling_op_reshaped = conv_layer_1_pooling_op.reshape((1,conv_layer_1_pooling_op.shape[0],conv_layer_1_pooling_op.shape[1],conv_layer_1_pooling_op.shape[2]))
        grads = self.convOpOpti(conv_layer_1_pooling_op_reshaped,rotated_dell,conv_layer_1_pooling_op_reshaped.shape[-2:],rotated_dell.shape[-2:],self.filter_size_layer_2[1:] ,backpass=1)
        grads = np.flip(np.flip(grads,-2),-1)
        kernel_layer_2_grads += grads
        #at this point we have all grads for all kernel of conv layer 2, now we have to propogate error backwards
        dell_pooled_layer_1 = np.zeros(conv_layer_1_pooling_op.shape)
        stacked_kernel_2 = np.array(self.filters_layer_2)
        stacked_kernel_2 = np.flip(np.flip(stacked_kernel_2,-2),-1)
        d_relu_layer_2_reshaped = d_relu_layer_2.reshape((d_relu_layer_2.shape[0],1)+d_relu_layer_2.shape[1:])

        dell_pooled_layer_1 = self.convOpOpti(d_relu_layer_2_reshaped,stacked_kernel_2, d_relu_layer_2_reshaped.shape[-2:],stacked_kernel_2.shape[-2:],-1,convType="full")
        conv_layer_1_pooling_op_non_activated = layer_1_pooling['pooled_non_activated']

        dell_pooling_layer_1 = dell_pooled_layer_1 * self.reluDerivative(conv_layer_1_pooling_op_non_activated)
        #at this point i have all the dell in the maxed_pooled layer now to propogate to layer before it
        X_relu_layer_1 = layer_1_conv_box['relu']
        dell_relu_layer_1 = np.zeros(X_relu_layer_1.shape)
        max_x_pooling_layer_1 = layer_1_pooling['max_indexes_x']
        max_y_pooling_layer_1 = layer_1_pooling['max_indexes_y']
        for ch in range(dell_relu_layer_1.shape[0]):
            dell_relu_layer_1[ch,max_x_pooling_layer_1[ch],max_y_pooling_layer_1[ch]] = dell_pooling_layer_1[ch]

        bias_layer_1_grads = np.sum(dell_relu_layer_1,axis=(-2,-1))
        #now to get change in weights
        rotated_dell = np.flip(np.flip(dell_relu_layer_1,-2),-1)
        # bias_layer_1_grads = np.sum(rotated_dell,axis=(-1,-2))
        X_curr_reshaped = X_curr.reshape((1,)+X_curr.shape)
        grads = self.convOpOpti(X_curr_reshaped,rotated_dell,self.input_dim,rotated_dell.shape[-2:],self.filter_size_layer_1)
        grads = np.flip(np.flip(grads,-2),-1)
        kernel_layer_1_grads += grads

    # at this point all the grads are calculated now just to stack it up into 1d array
    #will stack up in forward fashion
    all_grads = np.array([])
    # 1 st conv layer all kernel's biases, all_kernels
    all_grads = np.concatenate((all_grads,bias_layer_1_grads,kernel_layer_1_grads.flatten()))
    #2nd conv layer params
    all_grads = np.concatenate((all_grads,bias_layer_2_grads,np.array(kernel_layer_2_grads).flatten()))
    #fully connected now
    all_grads = np.concatenate((all_grads, weight_layer_1_grads.flatten(), weight_layer_2_grads.flatten(),weight_layer_3_grads.flatten()))
    self.gradientCheck("conv_2",X_pool,y_pool,kernel_layer_1_grads.flatten(),bias = bias_layer_1_grads)
    print "type any char to move forward"
    raw_input()
    return J, all_grads


class CNN:

    def __init__(self):
        self.obj = Preprocess() #this will return all images i.e X values and the expected output
        #layer 1 params
        self.input_dim = params['input_dim']['val']
        self.n_padding_bits = params['n_padding_bits']['val']
        self.n_filter_layer_1 = params['n_filter_layer_1']['val']
        self.filter_size_layer_1 = params['filter_size_layer_1']['val']
        self.filter_stride = params['filter_stride']['val']
        temp_dim = (self.input_dim[0] - self.filter_size_layer_1[0] + 2*self.n_padding_bits)/self.filter_stride + 1
        self.conv_op_layer_1_out_dim = [temp_dim, temp_dim] #output dimension after 1st layer convolution operation by default 50x50
        tmp_dim_pooling = (temp_dim - params['pooling_filter_size_layer_1']['val'][0])/params["pooling_stride_layer_1"]["val"] + 1
        self.pooling_layer_1_out_dim = [tmp_dim_pooling,tmp_dim_pooling]
        self.filters_layer_1 = []
        self.bias_layer_1 = []
        #layer 2 params
        self.n_filter_layer_2 = params['n_filter_layer_2']["val"]
        self.filter_size_layer_2 = tuple([self.n_filter_layer_1,params['filter_size_layer_2']['val'][0],params['filter_size_layer_2']['val'][1]])
        self.filters_layer_2 = []
        self.bias_layer_2 = []
        tmp_dim = (self.pooling_layer_1_out_dim[0] - self.filter_size_layer_2[1])/self.filter_stride + 1 #by default this will be 23*23
        self.conv_op_layer_2_out_dim =  [tmp_dim,tmp_dim]
        tmp_dim_pooling = (tmp_dim - params['pooling_filter_size_layer_2']['val'][0])/params["pooling_stride_layer_2"]["val"] + 1
        self.pooling_layer_2_out_dim = [tmp_dim_pooling,tmp_dim_pooling] #by default 22
        self.out_nodes_after_conv = self.pooling_layer_2_out_dim[0]*self.pooling_layer_2_out_dim[1]*self.n_filter_layer_2 #1452 by defailt
        #fully connected layers
        self.n_hidden_layers = params['n_hidden_layers']["val"]
        self.weights = [] #will contain the weights of the network total 3
        self.n_nodes_hidden_layer_1 = params["n_nodes_hidden_layer_1"]["val"]
        self.n_nodes_hidden_layer_2 = params["n_nodes_hidden_layer_2"]["val"]
        self.output_classes = params["output_classes"]["val"]
        self.dropout_percent_layer_1 = params["dropout_percent_layer_1"]["val"]
        self.dropout_percent_layer_2 = params["dropout_percent_layer_2"]["val"]
        self.intermediate_results = {}
        self.count = 0
        self.losses = []

    def train(self):
        #flow
        #train -> gradient descent -> backward_prop -> feed_forward to calculate values
        self.obj = self.obj.process()
        X_train,y_train = self.obj.X_train[:100], self.obj.y_train[:100]
        X_train = np.array([X[0:18,0:18] for X in X_train])
        print len(X_train)
        X_train = self.padBits(X_train,self.n_padding_bits) #will return the all images after padding
        #make random weights i.e filters
        self.randomFilterValues()
        #make params vector
        theta = self.makeThetaVector()
        # print theta, len(theta)
        print "training starting"
        # vals = self.gradientDescent(theta,X_train,y_train)
        n_epochs = params['n_epochs']
        learning_rate = params['learning_rate']
        mini_batch_size = params['batch_size']
        vals = self.MiniBatchGd(theta,X_train,y_train,n_epochs=n_epochs, mini_batch_size=mini_batch_size,learning_rate=learning_rate)
        self.fromThetaVectorToWeights(vals)
        #pickle the object




    @staticmethod
    def padBits(X,n_bits):
        #(before the number, after the number)
        #((along depth) , (along rows) , (along col))
        npad = ((0,0),(n_bits,n_bits),(n_bits,n_bits))
        X = np.pad(X, pad_width=npad, mode='constant', constant_values = 0)
        return X


    def randomFilterValues(self):
        mean = params["mean"]
        std = params["std"]
        #for conv layer box
        self.bias_layer_1 = list(np.random.normal(mean,0.00001,self.n_filter_layer_1)+1)
        #filter values for layer 1
        l1 = np.sqrt(2.0/np.product(self.filter_size_layer_1))
        for i in range(self.n_filter_layer_1):
            self.filters_layer_1.append(np.random.randn(self.filter_size_layer_1[0],self.filter_size_layer_1[1])*l1)
            #filter values for layer 2
        self.bias_layer_2 = list(np.random.normal(mean,0.00001,self.n_filter_layer_2)+1)
        l2 = np.sqrt(2.0/np.product(self.filter_size_layer_2))
        for i in range(self.n_filter_layer_2):
            self.filters_layer_2.append(np.random.randn(self.filter_size_layer_2[0],self.filter_size_layer_2[1],self.filter_size_layer_2[2])*l2)
        #for fully connected layers
        # +1 for bias
        shape_weight_layer_1 = (self.n_nodes_hidden_layer_1,self.out_nodes_after_conv+1)
        shape_weight_layer_2 = (self.n_nodes_hidden_layer_2,self.n_nodes_hidden_layer_1+1)
        shape_weight_layer_3 = (self.output_classes,self.n_nodes_hidden_layer_2+1)
        self.weights.append((1.0/np.sqrt(self.out_nodes_after_conv/2))*np.random.randn(shape_weight_layer_1[0],shape_weight_layer_1[1]))
        self.weights.append((1.0/np.sqrt(self.n_nodes_hidden_layer_1/2))*np.random.randn(shape_weight_layer_2[0],shape_weight_layer_2[1]))
        self.weights.append((1.0/np.sqrt(self.n_nodes_hidden_layer_2/2))*np.random.randn(shape_weight_layer_3[0],shape_weight_layer_3[1]))


    #this function does a feed forward
    def feedForward(self, X, y, training=False, g_check=False):
        #layer 1
        #this is to store intermediate results just for training thing
        layer_1_conv_box = {}
        X = self.convulationOp(X,layer=1) # 50X50X4
        layer_1_conv_box["convOp"] = X
        X = self.relu(X)
        layer_1_conv_box["relu"] = X
        layer_1_pooling = {}
        X,max_indexes_x, max_indexes_y = self.maxPooling(X,params["pooling_stride_layer_1"]["val"],params["pooling_filter_size_layer_1"]["val"])
        pooling_non_activated = np.zeros(X.shape)
        for ch in range(X.shape[0]):
            pooling_non_activated[ch] = layer_1_conv_box["convOp"][ch,max_indexes_x[ch],max_indexes_y[ch]]
        layer_1_pooling['pooled_non_activated'] = pooling_non_activated
        layer_1_pooling["pooling_val"] = X
        layer_1_pooling["max_indexes_x"] = max_indexes_x
        layer_1_pooling["max_indexes_y"] = max_indexes_y

        #layer 2
        layer_2_conv_box = {}
        X = self.convulationOp(X,layer=2) #of dimension 23X23X3
        layer_2_conv_box["convOp"] = X
        X = self.relu(X)
        layer_2_conv_box["relu"] = X
        layer_2_pooling = {}
        X,max_indexes_x_layer_2,max_indexes_y_layer_2 = self.maxPooling(X,params["pooling_stride_layer_2"]["val"],params["pooling_filter_size_layer_2"]["val"])
        layer_2_pooling["pooling_val"] = X
        layer_2_pooling["max_indexes_x"] = max_indexes_x_layer_2
        layer_2_pooling["max_indexes_y"] = max_indexes_y_layer_2
        z0_tensor =np.zeros(X.shape)
        # now i have total 22 X 22 X 3 image there total neurons = 1452
        for ch in range(X.shape[0]):
            z0_tensor[ch] =layer_2_conv_box["convOp"][ch,max_indexes_x_layer_2[ch],max_indexes_y_layer_2[ch]]

        fully_connected = {}
        fully_connected['z0'] =np.concatenate((np.array([1]),z0_tensor.flatten()))
        #fully connected
        X = self.flattenLayer(X) #this flattens the layer to make a column vector and adds 1 as a bias
        fully_connected['a0'] = X
        X = self.fullyConnected(X,layer=1)
        fully_connected['z1'] = np.concatenate((np.array([1]),X))
        X = self.relu(X)
        #at the 1st hidden layer
        X = self.flattenLayer(X) #add one as bias
        fully_connected['a1'] = X
        #perform dropout
        # if(training):
        #     X = self.dropout(X,self.dropout_percent_layer_1)

        X = self.fullyConnected(X,layer=2)
        fully_connected['z2'] = np.concatenate((np.array([1]),X))
        X = self.relu(X)
        X = self.flattenLayer(X) #add one as bias
        fully_connected['a2'] = X
        # if(training):
        #     X = self.dropout(X,self.dropout_percent_layer_2)

        X = self.fullyConnected(X,layer=3)
        fully_connected['z3'] = np.concatenate((np.array([1]),X))
        #now X contains the output now have to just squash the stuff
        X = self.softmax(X)
        fully_connected['a3'] = X
        if g_check:
            return X
        return layer_1_conv_box, layer_1_pooling, layer_2_conv_box, layer_2_pooling, fully_connected



    def convulationOp(self, X, layer):
        #apply layer 1 filters
        if(layer==1):
            #here there will be the naive image
            convolved_2d_img = []
            i=0
            #stack kernel
            k_shape = self.filters_layer_1[0].shape
            stacked_kernels = np.dstack(self.filters_layer_1)
            stacked_kernels = np.rollaxis(stacked_kernels,-1)
            shape = X.shape
            X = X.reshape((1,X.shape[0],X.shape[1]))
            #depth = #kernels
            bias = np.array(self.bias_layer_1)
            convolved = self.convOpOpti(X,stacked_kernels,shape,k_shape,self.conv_op_layer_1_out_dim)
            print convolved.shape, bias.reshape((bias.shape[0],1,1)).shape
            convolved = convolved + bias.reshape((bias.shape[0],1,1))
            return convolved
        #layer 2
        else:
            #here input will be 25X25X4 after pooling
            #filter here must be of same depth as that of input
            k_shape = self.filters_layer_2[0].shape[1:]
            shape = X.shape[1:]
            X = X.reshape((1,X.shape[0],X.shape[1],X.shape[2]))
            kernels = np.array(self.filters_layer_2)
            bias = np.array(self.bias_layer_2)
            convolved = self.convOpOpti(X,kernels,shape,k_shape,self.conv_op_layer_2_out_dim,layer=2)
            convolved = convolved + bias.reshape((bias.shape[0],1,1))
            return convolved


    @staticmethod
    def convOpOpti(X, kernel, x_dim, kernel_dim, output_dim,layer=1, backpass=0,convType="valid"):   #kernel will be all kernels

        padded_dim = np.array(x_dim) + np.array(kernel_dim) - 1
        output_dim = np.array(output_dim)

        fft_result = np.fft.fft2(X,padded_dim,axes=(-2,-1)) * np.fft.fft2(kernel, padded_dim, axes=(-2,-1))
        target = np.fft.ifft2(fft_result).real

        if(convType=="full"):
            print "aaaaaaaaaaaaaaaaaaaaaaaaaaa"
            print target.shape
            print "-------------aaaaaaaaaaaaaaaaaaaaaaaaaaa------"

            return np.sum(target,axis=0)

        start_i = (padded_dim -output_dim ) // 2
        end_i = start_i + output_dim

        if(layer==2):
            target = np.sum(target,axis=(1))
        if(backpass==1):
            return target[:,:,start_i[0]:end_i[0], start_i[1]:end_i[1]]
        return target[:,start_i[0]:end_i[0], start_i[1]:end_i[1]]


    @staticmethod
    def convOp(X, kernel, x_dim, kernel_dim, output_dim,conv_type='valid'):
        #to make the dimension equal
        padded_dim = np.array(x_dim) + np.array(kernel_dim) - 1
        output_dim = np.array(output_dim)
        #applying convolution theorem
        fft_result = np.fft.fft2(X, padded_dim) * np.fft.fft2(kernel, padded_dim)
        target = np.fft.ifft2(fft_result).real

        if(conv_type=='full'):
            return target
        #now to extract the convolution
        #here convolution is correlation with the flipped filter
        start_i = (padded_dim -output_dim ) // 2
        end_i = start_i + output_dim
        convolution = target[start_i[0]:end_i[0], start_i[1]:end_i[1]]
        return convolution

    def relu(self,X):
        X[X<0] = 0
        return X

    def maxPooling(self,X,stride,size):
        #layer 1, X = (50,50,4) and output will be after pooling (25X25X4)
        max_indexes_x = [] #this will be same as the dimension of output of max pooling, just to use in backprop
        max_indexes_y = []
        all_pooled_channels = []#to store all pooled stuffs
        ch_rows, ch_cols = X[0].shape #cols 50
        kernel_rows, kernel_cols = size
        output_rows = (ch_rows - kernel_rows)/stride + 1
        output_cols = (ch_cols - kernel_cols)/stride + 1
        for channel in X:
            #to store pooled values
            pooled_channel = np.zeros((output_rows,output_cols))
            max_i = np.zeros((output_rows,output_cols), dtype=int)
            max_j = np.zeros((output_rows,output_cols), dtype=int)
            curr_x, curr_y = 0,0 #variables to keep track of current pool value to be filled
            #channel is a 2 d stuff
            #loop over the matrix with a stride 2
            for i in range(0,ch_rows - kernel_rows + 1,stride):
                for j in range(0,ch_cols - kernel_cols + 1,stride):
                    start_i, start_j = i, j
                    end_i = start_i + kernel_rows
                    end_j = start_j + kernel_cols
                    patch = channel[start_i:end_i, start_j:end_j]
                    max_val_index_in_patch = np.argmax(patch)
                    #get the coordinates in the patch then shift the origin to get the actual coordinates
                    y_in_patch  = max_val_index_in_patch % kernel_cols
                    x_in_patch = int((max_val_index_in_patch - y_in_patch)/kernel_cols)
                    x_in_ch = start_i + x_in_patch
                    y_in_ch = start_j + y_in_patch
                    max_i[curr_x][curr_y] = x_in_ch
                    max_j[curr_x][curr_y] = y_in_ch
                    pooled_channel[curr_x][curr_y] = patch[x_in_patch][y_in_patch]
                    curr_x, curr_y = self.modifyCo(curr_x,curr_y,output_rows,output_cols)

            all_pooled_channels.append(pooled_channel)
            max_indexes_x.append(max_i)
            max_indexes_y.append(max_j)

        all_pooled_channels = np.rollaxis(np.dstack(all_pooled_channels),-1)
        max_indexes_x = np.rollaxis(np.dstack(max_indexes_x),-1)
        max_indexes_y = np.rollaxis(np.dstack(max_indexes_y),-1)
        return all_pooled_channels,max_indexes_x,max_indexes_y


    @staticmethod
    def modifyCo(curr_x, curr_y, rows, cols):
        if(curr_y == cols-1):
            return curr_x + 1, 0
        else:
            return curr_x,curr_y+1

    @staticmethod
    def flattenLayer(X):
        # also add bias
        return np.insert(X.flatten(),0,1)

    def fullyConnected(self,X,layer):
        Z = np.dot(self.weights[layer-1],X)
        return Z

    @staticmethod
    def dropout(X,p):
        mask = np.random.binomial(1,p,X.shape)
        return X*mask

    @staticmethod
    def softmax(X):
        X -= np.max(X) #for numeric stability
        expo = np.exp(X)
        return expo/np.sum(expo,axis=0)

    def makeThetaVector(self):
        all_theta = np.array([])
        #first bias of all kernels
        all_theta = np.concatenate((all_theta,np.array(self.bias_layer_1),np.array(self.filters_layer_1).flatten()))
        #2nd conv layer
        all_theta = np.concatenate((all_theta,np.array(self.bias_layer_2),np.array(self.filters_layer_2).flatten()))
        #fully connected
        all_theta = np.concatenate((all_theta, self.weights[0].flatten(), self.weights[1].flatten(), self.weights[2].flatten()))
        return all_theta


        pass

    def gradientCheck(self,layer,X,y,actual,bias=None):
        epsilon = params['epsilon']
        if layer== 'conv_2':
            curr_weight = np.array(self.filters_layer_1)
            shape = curr_weight.shape
            flattened = curr_weight.flatten()
            approx = []
            approx_bias = []
            for i in range(len(bias)):
                J1 = 0
                J2 = 0
                self.bias_layer_1[i] =   self.bias_layer_1[i] + epsilon
                for im in range(len(X)):
                    a = self.feedForward(X[im],y[im],g_check=True)
                    J1 += self.softmaxLoss(a,y[im],False)
                self.bias_layer_1[i] =   self.bias_layer_1[i] - epsilon
                self.bias_layer_1[i] =   self.bias_layer_1[i] - epsilon
                for im in range(len(X)):
                    J2 += self.softmaxLoss(self.feedForward(X[im],y[im],g_check=True),y[im],False)
                self.bias_layer_1[i] =   self.bias_layer_1[i] + epsilon
                approx_bias.append((1.0 * (J1-J2))/(2*epsilon))

            for i in range(len(flattened)):
                J1 = 0
                J2 = 0
                flattened[i] = flattened[i] + epsilon
                self.filters_layer_1 = flattened.reshape((shape))
                for im in range(len(X)):
                    a = self.feedForward(X[im],y[im],g_check=True)
                    # print "got here"
                    # print a
                    J1 += self.softmaxLoss(a,y[im],False)
                flattened[i] = flattened[i] - epsilon #make as the previous
                flattened[i] = flattened[i] - epsilon #modify
                self.filters_layer_1 = flattened.reshape((shape))
                for im in range(len(X)):
                    J2 += self.softmaxLoss(self.feedForward(X[im],y[im],g_check=True),y[im],False)
                flattened[i] = flattened[i] + epsilon #modify to previous state
                approx.append((1.0 * (J1-J2))/(2*epsilon))


        print "-------------------------------"
        print approx_bias
        print bias
        print "-------------------------------"
        approx = np.array(approx)
        print approx
        print actual
        nume = np.linalg.norm(approx-actual)
        deno = np.linalg.norm(actual) + np.linalg.norm(approx)
        print "ratio is " +  str(nume/deno)






    def gradientDescent(self,theta,X,y):
        fmin = minimize(fun=self.backprop,x0=theta,args=(X,y),method='L-BFGS-B',jac=True,options={"maxiter":200})
        return fmin

    def MiniBatchGd(self,theta,X,y,n_epochs,mini_batch_size,learning_rate):
        zipped = zip(X,y)
        for epoch in xrange(n_epochs):
            np.random.shuffle(zipped)
            X,y = zip(*zipped)
            X = np.array(X)
            y = np.array(y)
            loss_total = 0
            for i in xrange(0,X.shape[0],mini_batch_size):
                X_mini = X[i:i+mini_batch_size]
                y_mini = y[i:i+mini_batch_size]
                grads, loss = self.backprop(theta, X_mini, y_mini)
                loss_total += loss
                theta += learning_rate * grads
            print "iteration "+str(epoch+1)+" loss "+str(loss_total)

            self.losses.append(loss_total)

        return theta

    def backprop(self,theta,X,y):
        if self.count%5 == 0:
            print "-------------------------------------------------------"
            print(str(self.count)+" times the function is called time taken in seconds "+str(time.time()-start) )
            print "-------------------------------------------------------"
        self.count += 1
        # batch_co = np.random.choice(X.shape[0],size=params['batch_size'],replace=False)
        # # X_batch =
        # X = X[batch_co]
        # y = y[batch_co]
        #this function will make from one d to weights
        self.fromThetaVectorToWeights(theta) #now all weights loaded in the self object
        J = 0
        #initialze grad matrix
        weight_layer_3_grads = np.zeros(self.weights[LAYER_3].shape)
        weight_layer_2_grads = np.zeros(self.weights[LAYER_2].shape)
        weight_layer_1_grads = np.zeros(self.weights[LAYER_1].shape)
        kernel_layer_2_grads = [np.zeros(self.filter_size_layer_2) for k in range(self.n_filter_layer_2)]
        kernel_layer_1_grads = np.zeros((self.n_filter_layer_1,self.filter_size_layer_1[0],self.filter_size_layer_1[1]))
        bias_layer_2_grads = np.zeros((self.n_filter_layer_2))
        bias_layer_1_grads = np.zeros((self.n_filter_layer_1))

        #for all image
        pool_size = params['pool_size']
        # num_cores = multiprocessing.cpu_count()
        # ite = [delayed(backpropBodyParllel)(self,X[im:im+pool_size],y[im:im+pool_size]) for im in range(0,len(X),pool_size)]
        # all_return_values = Parallel(n_jobs=num_cores)(ite)
        print "batch "+str(self.count)
        all_return_values = []
        # for im in range(len(X)):
        print len(X)
        raw_input()
        all_return_values.append(backpropBodyParllel(self,X,y))


        J = 0
        all_grads = np.zeros(all_return_values[0][1].shape)
        for i in range(len(all_return_values)):
            J += all_return_values[i][0]
            all_grads += all_return_values[i][1]


        print "loss "+str(J)+" at iteration "+str(self.count)

        return J, all_grads

    def fromThetaVectorToWeights(self, theta):
        #transforming logic
        # kernel_layer_1,kernel_layer_2,biases_layer_1,biases_layer_2,weights variables
        #1st conv params
        self.bias_layer_1 = list(theta[0:self.n_filter_layer_1])
        self.filters_layer_1 = []
        elements_in_filter_1 = np.product(self.filter_size_layer_1)
        prev = self.n_filter_layer_1
        for i in range(self.n_filter_layer_1):
            get_curr_filter = theta[prev:prev+elements_in_filter_1]
            # print get_curr_filter
            self.filters_layer_1.append(np.reshape(get_curr_filter,self.filter_size_layer_1))
            prev = prev + elements_in_filter_1

        theta = theta[prev:]
        #2nd conv params
        self.bias_layer_2 = list(theta[0:self.n_filter_layer_2])
        self.filters_layer_2 = []
        prev = self.n_filter_layer_2
        elements_in_filter_2 = np.product(self.filter_size_layer_2)
        for i in range(self.n_filter_layer_2):
            get_curr_filter = theta[prev:prev+elements_in_filter_2]
            self.filters_layer_2.append(np.reshape(get_curr_filter,self.filter_size_layer_2))
            prev = prev + elements_in_filter_2

        theta = theta[prev:]
        #now get fully connected stuffs
        shape_weight_layer_1 = (self.n_nodes_hidden_layer_1,self.out_nodes_after_conv+1)
        shape_weight_layer_2 = (self.n_nodes_hidden_layer_2,self.n_nodes_hidden_layer_1+1)
        shape_weight_layer_3 = (self.output_classes,self.n_nodes_hidden_layer_2+1)
        self.weights = []
        self.weights.append(np.reshape(theta[:np.product(shape_weight_layer_1)],shape_weight_layer_1))
        theta = theta[np.product(shape_weight_layer_1):]
        self.weights.append(np.reshape(theta[:np.product(shape_weight_layer_2)],shape_weight_layer_2))
        theta = theta[np.product(shape_weight_layer_2):]
        self.weights.append(np.reshape(theta[:np.product(shape_weight_layer_3)],shape_weight_layer_3))
        #all params done


    @staticmethod
    def softmaxLoss(a,y_curr,prints=True):
        if prints:
            print "[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]"
            print a, y_curr
            print "[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]"

        return np.sum(-y_curr * np.log(a))

    @staticmethod
    def reluDerivative(z):
        z[z>0] = 1
        z[z<=0] = 0
        return z



if __name__ == '__main__':
    cnn = CNN()
    cnn.train()
    pickle_file_cnn_object = open('pickle_models/cnn_object_11_', 'w')
    pickle.dump(cnn, pickle_file_cnn_object)
    pickle_file_cnn_object.close()
    print("--- %s completed in seconds ---" % (time.time() - start))
