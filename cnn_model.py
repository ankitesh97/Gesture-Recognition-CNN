
import numpy as np
import json
import pickle


params_file  = open('params.json','r')
params = json.load(params_file)
params_file.close()
#architecture
#two black boxes i.e cnn->relu->max_pooling
#fully connected with two hidden layers

class CNN:

    def __init__(self):
        self.input_dim = params['input_dim']['val']
        self.n_padding_bits = params['n_padding_bits']['val']
        self.n_filter_layer_1 = params['n_filter_layer_1']['val']
        self.filter_size_layer_1 = params['filter_size_layer_1']['val']
        self.filter_stride = params['filter_stride']['val']
        temp_dim = (self.input_dim[0] - self.filter_size_layer_1[0] + 2*self.n_padding_bits)/self.filter_stride + 1
        self.conv_op_layer_1_out_dim = [temp_dim, temp_dim] #output dimension after 1st layer convolution operation by default 50x50
        self.X, self.y = preProcess() #this will return all images i.e X values and the expected output



    def train(self):
        self.X = self.padBits(self.n_padding_bits) #will return the image after padding



    @staticmethod
    def padBits(n_bits):
        pass

    def convulationOp(self, img, ):
        pass



if __name__ == '__main__':
    cnn = CNN()
    cnn.train()
    print cnn.X
