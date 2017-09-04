
import numpy as np
import json
import pickle
from img_preprocess import Preprocess


params_file  = open('params.json','r')
params = json.load(params_file)
params_file.close()
#architecture
#two black boxes i.e cnn->relu->max_pooling
#fully connected with two hidden layers

class CNN:

    def __init__(self):
        self.obj = preProcess() #this will return all images i.e X values and the expected output
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
        self.filter_size_layer_2 = tuple(self.n_filter_layer_1,params['filter_size_layer_2']['val'][0],params['filter_size_layer_2']['val'][1])
        self.filters_layer_2 = []
        self.bias_layer_2 = []
        tmp_dim = (self.pooling_layer_1_out_dim[0] - self.filter_size_layer_2[1])/self.filter_stride + 1 #by default this will be 23*23
        self.conv_op_layer_2_out_dim =  [tmp_dim,tmp_dim]

    def train(self):
        X_train,y_train = self.obj.X_train, self.obj.y_train
        X_train = self.padBits(X_train,self.n_padding_bits) #will return the all images after padding
        #make random weights i.e filters
        randomFilterValues()
        for img_index in len(X_train):
            output = self.feedForward(X_train[img_index],y_train[img_index])



    @staticmethod
    def padBits(X,n_bits):
        #(before the number, after the number)
        #((along depth) , (along rows) , (along col))
        npad = ((0,0),(n_bits,n_bits),(n_bits,n_bits))
        X = np.pad(X, pad_width=npad, mode='constant', constant_values = 0)
        return X


    def randomFilterValues(self,layer):
        mean = params["mean"]
        std = params["std"]
        if layer == 1:
            self.bias_layer_1 = list(np.random.normal(mean,0.00001,self.n_filter_layer_1)+1)
            #filter values for layer 1
            for i in range(self.n_filter_layer_1):
                self.filters_layer_1.append(np.random.normal(mean,std,tuple(self.filter_size_layer_1)))
                #filter values for layer 2
        else:
            self.bias_layer_2 = list(np.random.normal(mean,0.00001,self.n_filter_layer_2)+1)
            for i in range(self.n_filter_layer_2):
                self.filters_layer_2.append(np.random.normal(mean,std,self.filter_size_layer_2))

    #this function does a feed forward
    def feedForward(self, X, y):
        #layer 1
        X = self.convulationOp(X,layer=1) # 50X50X4
        X = self.relu(X)
        X,max_indexes_x, max_indexes_y = self.maxPooling(X,params["pooling_stride_layer_1"]["val"],params["pooling_filter_size_layer_1"]["val"])
        #layer 2
        X = self.convulationOp(X,layer=2) #of dimension 23X23X3
        X = self.relu(X)
        X,max_indexes_x_layer_2,max_indexes_y_layer_2 = self.maxPooling(X,params["pooling_stride_layer_2"]["val"],params["pooling_filter_size_layer_2"]["val"])
        # now i have total 22 X 22 X 3 image there total neurons = 1452
        #fully connected





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
                convolved_img.append(convolved_2d_plane)

            convolved = np.dstack(convolved_img)
            return np.rollaxis(convolved,-1)

    @staticmethod
    def convOp(X, kernel, x_dim, kernel_dim, output_dim):
        #to make the dimension equal
        padded_dim = x_dim + kernel_dim - 1
        #applying convolution theorem
        fft_result = np.fft.fft2(X, padded_dim) * np.fft.fft2(kernel, padded_dim)
        target = np.fft.ifft2(fft_result).real

        #now to extract the convolution
        #here convolution is correlation with the flipped filter
        start_i = (padded_dim - output_dim) // 2
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
            curr_x,curr_y+1


if __name__ == '__main__':
    cnn = CNN()
    cnn.train()
    print cnn.X
