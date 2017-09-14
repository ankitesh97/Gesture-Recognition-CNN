
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

def backpropBodyParllel(self,X_curr,y_curr):

    weight_layer_3_grads = np.zeros(self.weights[LAYER_3].shape)
    weight_layer_2_grads = np.zeros(self.weights[LAYER_2].shape)
    weight_layer_1_grads = np.zeros(self.weights[LAYER_1].shape)
    kernel_layer_2_grads = [np.zeros(self.filter_size_layer_2) for k in range(self.n_filter_layer_2)]
    kernel_layer_1_grads = np.zeros((self.n_filter_layer_1,self.filter_size_layer_1[0],self.filter_size_layer_1[1]))
    bias_layer_2_grads = np.zeros((self.n_filter_layer_2))
    bias_layer_1_grads = np.zeros((self.n_filter_layer_1))
    layer_1_conv_box, layer_1_pooling, layer_2_conv_box, layer_2_pooling, fully_connected  = self.feedForward(X_curr,y_curr,training=True) #out vector
    a_last_layer = fully_connected['a3']
    loss_curr = self.softmaxLoss(a_last_layer,y_curr)
    J = loss_curr
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
    for kernel in range(self.n_filter_layer_2):
        rotated_dell = np.flip(np.flip(d_relu_layer_2[kernel],0),1)
        bias_layer_2_grads[kernel] += np.sum(rotated_dell)
        grads_for_kernel = []
        for ch in range(conv_layer_1_pooling_op.shape[0]):
            grads = self.convOp(conv_layer_1_pooling_op[ch],rotated_dell,conv_layer_1_pooling_op_shape,rotated_dell.shape,self.filter_size_layer_2[1:])
            grads_for_kernel.append(grads)
        grads_for_kernel = np.dstack(grads_for_kernel)
        grads_for_kernel = np.rollaxis(grads_for_kernel,-1)
        kernel_layer_2_grads[kernel] += grads_for_kernel

    #at this point we have all grads for all kernel of conv layer 2, now we have to propogate error backwards
    dell_pooled_layer_1 = np.zeros(conv_layer_1_pooling_op.shape)
    for kernel in range(self.n_filter_layer_2):
        dell2 = d_relu_layer_2[kernel]
        filter_curr = self.filters_layer_2[kernel]
        for ch in range(dell_pooled_layer_1.shape[0]):
             filter_ch_curr = filter_curr[ch]
             flipped_filter_ch_curr  = np.flip(np.flip(filter_ch_curr,0),1)
             val_change_curr_ch = self.convOp(flipped_filter_ch_curr,dell2,flipped_filter_ch_curr.shape,dell2.shape,-1,"full")
             dell_pooled_layer_1[ch] += val_change_curr_ch

    dell_pooling_layer_1 = dell_pooled_layer_1 * self.reluDerivative(conv_layer_1_pooling_op)
    #at this point i have all the dell in the maxed_pooled layer now to propogate to layer before it
    X_relu_layer_1 = layer_1_conv_box['relu']
    dell_relu_layer_1 = np.zeros(X_relu_layer_1.shape)
    max_x_pooling_layer_1 = layer_1_pooling['max_indexes_x']
    max_y_pooling_layer_1 = layer_1_pooling['max_indexes_y']
    for ch in range(dell_relu_layer_1.shape[0]):
        dell_relu_layer_1[ch,max_x_pooling_layer_1[ch],max_y_pooling_layer_1[ch]] = dell_pooling_layer_1[ch]

    #now to get change in weights
    for kernel in range(self.n_filter_layer_1):
        rotated_dell = np.flip(np.flip(dell_relu_layer_1[kernel],0),1)
        bias_layer_1_grads[kernel] += np.sum(rotated_dell)
        grads = self.convOp(X_curr,rotated_dell,self.input_dim,rotated_dell.shape,self.filter_size_layer_1)
        kernel_layer_1_grads[kernel] += grads


    # at this point all the grads are calculated now just to stack it up into 1d array
    #will stack up in forward fashion
    all_grads = np.array([])
    # 1 st conv layer all kernel's biases, all_kernels
    all_grads = np.concatenate((all_grads,bias_layer_1_grads,kernel_layer_1_grads.flatten()))
    #2nd conv layer params
    all_grads = np.concatenate((all_grads,bias_layer_2_grads,np.array(kernel_layer_2_grads).flatten()))
    #fully connected now
    all_grads = np.concatenate((all_grads, weight_layer_1_grads.flatten(), weight_layer_2_grads.flatten(),weight_layer_3_grads.flatten()))


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

    def train(self):
        #flow
        #train -> gradient descent -> backward_prop -> feed_forward to calculate values
        self.obj = self.obj.process()
        X_train,y_train = self.obj.X_train, self.obj.y_train
        print len(X_train)
        X_train = self.padBits(X_train,self.n_padding_bits) #will return the all images after padding
        #make random weights i.e filters
        self.randomFilterValues()
        #make params vector
        theta = self.makeThetaVector()
        # print theta, len(theta)
        print "training starting"
        params = self.gradientDescent(theta,X_train,y_train)
        self.fromThetaVectorToWeights(params[0])
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
        for i in range(self.n_filter_layer_1):
            self.filters_layer_1.append(np.random.normal(mean,std,tuple(self.filter_size_layer_1)))
            #filter values for layer 2
        self.bias_layer_2 = list(np.random.normal(mean,0.00001,self.n_filter_layer_2)+1)
        for i in range(self.n_filter_layer_2):
            self.filters_layer_2.append(np.random.normal(mean,std,self.filter_size_layer_2))
        #for fully connected layers
        # +1 for bias
        shape_weight_layer_1 = (self.n_nodes_hidden_layer_1,self.out_nodes_after_conv+1)
        shape_weight_layer_2 = (self.n_nodes_hidden_layer_2,self.n_nodes_hidden_layer_1+1)
        shape_weight_layer_3 = (self.output_classes,self.n_nodes_hidden_layer_2+1)
        self.weights.append(np.random.normal(mean,std,shape_weight_layer_1))
        self.weights.append(np.random.normal(mean,std,shape_weight_layer_2))
        self.weights.append(np.random.normal(mean,std,shape_weight_layer_3))


    #this function does a feed forward
    def feedForward(self, X, y, training=True):
        #layer 1
        #this is to store intermediate results just for training thing
        layer_1_conv_box = {}
        X = self.convulationOp(X,layer=1) # 50X50X4
        layer_1_conv_box["convOp"] = X
        X = self.relu(X)
        layer_1_conv_box["relu"] = X
        layer_1_pooling = {}
        X,max_indexes_x, max_indexes_y = self.maxPooling(X,params["pooling_stride_layer_1"]["val"],params["pooling_filter_size_layer_1"]["val"])
        layer_1_pooling["pooling_val"] = X
        layer_1_pooling["max_indexes_x"] = max_indexes_x
        layer_1_pooling["max_indexes_y"] = max_indexes_x

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
        if(training):
            X = self.dropout(X,self.dropout_percent_layer_1)

        X = self.fullyConnected(X,layer=2)
        fully_connected['z2'] = np.concatenate((np.array([1]),X))
        X = self.relu(X)
        X = self.flattenLayer(X) #add one as bias
        fully_connected['a2'] = X
        if(training):
            X = self.dropout(X,self.dropout_percent_layer_2)

        X = self.fullyConnected(X,layer=3)
        fully_connected['z3'] = np.concatenate((np.array([1]),X))
        #now X contains the output now have to just squash the stuff
        X = self.softmax(X)
        fully_connected['a3'] = X
        return layer_1_conv_box, layer_1_pooling, layer_2_conv_box, layer_2_pooling, fully_connected



    def convulationOp(self, X, layer):
        #apply layer 1 filters
        if(layer==1):
            #here there will be the naive image
            convolved_2d_img = []
            i=0
            for kernel in self.filters_layer_1:
                convolved_2d_img.append(self.convOp(X,kernel,X.shape,kernel.shape,self.conv_op_layer_1_out_dim)+self.bias_layer_1[i])
                i += 1

            convolved = np.dstack(convolved_2d_img)
            convolved = np.rollaxis(convolved, -1)
            return convolved
        #layer 2
        else:
            #here input will be 25X25X4 after pooling
            #filter here must be of same depth as that of input
            channels = X.shape[0]
            i=0
            convolved_img = [] #to append individual convolved stuff

            for kernel in self.filters_layer_2:
                convolved_in_2d_plane = np.zeros(tuple(self.conv_op_layer_2_out_dim))
                #for every channel
                for ch in range(channels):
                    convolved_in_2d_plane += self.convOp(X[ch],kernel[ch],X[ch].shape,kernel[ch].shape,self.conv_op_layer_2_out_dim)
                #ADD BIAS
                convolved_in_2d_plane += self.bias_layer_2[i]
                i += 1
                convolved_img.append(convolved_in_2d_plane)

            convolved = np.dstack(convolved_img)
            return np.rollaxis(convolved,-1)

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

    def gradientDescent(self,theta,X,y):
        fmin = fmin_tnc(func=self.backprop,x0=theta,args=(X,y),maxfun=10)
        return fmin




    def backprop(self,theta,X,y):
        if self.count%20 == 0:
            print "-------------------------------------------------------"
            print(str(self.count)+" times the function is called time taken in seconds "+str(time.time()-start) )
            print "-------------------------------------------------------"
        self.count += 1

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
        num_cores = multiprocessing.cpu_count()
        ite = [delayed(backpropBodyParllel)(self,X[im_index],y[im_index]) for im_index in range(5))]
        all_return_values = Parallel(n_jobs=num_cores)(ite)

        J = 0
        all_grads = np.zeros(all_return_values[0][1].shape)
        for i in range(len(all_return_values)):
            J += all_return_values[i][0]
            all_grads += all_return_values[i][1]


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
    def softmaxLoss(a,y_curr):
        return np.sum(-y_curr * np.log(a))

    @staticmethod
    def reluDerivative(z):
        z[z>0] = 1
        z[z<=0] = 0
        return z



if __name__ == '__main__':
    cnn = CNN()
    cnn.train()
    pickle_file_cnn_object = open('pickle_models/cnn_object_1', 'w')
    pickle.dump(cnn, pickle_file_cnn_object)
    pickle_file_cnn_object.close()
    print("--- %s completed in seconds ---" % (time.time() - start))
