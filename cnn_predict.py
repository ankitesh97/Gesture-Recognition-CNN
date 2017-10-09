
import numpy as np
from img_preprocess import Preprocess
import pickle
from cnn_model import CNN
import json
import time

MODEL_FILE = 'pickle_models/cnn_object_15_'
CLASSES = ["FIST","HAND","ONE","PEACE"] #change here if you want to add new gesture
params_file  = open('params.json','r')
params = json.load(params_file)
params_file.close()
start = time.time()

class CNNLite():
    def __init__(self):
        self.cnn_obj = pickle.load(open(MODEL_FILE,'r'))


    #this function does a feed forward
    def feedForward(self, X, training=True):
        #layer 1
        #this is to store intermediate results just for training thing
        X = self.cnn_obj.convulationOp(X,layer=1) # 50X50X4
        X = self.cnn_obj.relu(X)
        X,max_indexes_x, max_indexes_y = self.cnn_obj.maxPooling(X,params["pooling_stride_layer_1"]["val"],params["pooling_filter_size_layer_1"]["val"])
        #layer 2
        X = self.cnn_obj.convulationOp(X,layer=2) #of dimension 23X23X3
        X = self.cnn_obj.relu(X)
        X,max_indexes_x_layer_2,max_indexes_y_layer_2 = self.cnn_obj.maxPooling(X,params["pooling_stride_layer_2"]["val"],params["pooling_filter_size_layer_2"]["val"])
        X = self.cnn_obj.flattenLayer(X) #this flattens the layer to make a column vector and adds 1 as a bias
        X = self.cnn_obj.fullyConnected(X,layer=1)
        X = self.cnn_obj.relu(X)
        X = self.cnn_obj.flattenLayer(X) #add one as bias
        #perform dropout
        if(training):
            X = self.cnn_obj.dropout(X,self.cnn_obj.dropout_percent_layer_1)
        X = self.cnn_obj.fullyConnected(X,layer=2)
        X = self.cnn_obj.relu(X)
        X = self.cnn_obj.flattenLayer(X) #add one as bias
        if(training):
            X = self.cnn_obj.dropout(X,self.cnn_obj.dropout_percent_layer_2)
        X = self.cnn_obj.fullyConnected(X,layer=3)
        X = self.cnn_obj.softmax(X)
        return X


    #give one image point
    def predict(self,X):
        probabilities = self.feedForward(X,training=False)
        print probabilities
        max_index = np.argmax(probabilities)
        print CLASSES[max_index]
        return CLASSES[max_index],probabilities[max_index]


    def test(self,X,y):
        accuracy = []
        print "testing started"
        for im_index in range(len(X)):
            if(im_index%10==0):
                print "done for images "+str(im_index)
                print "time taken ------------------- "+str(time.time()-start)
            predicted_class, predicted_probab = self.predict(X[im_index])
            # for row in X[im_index]:
                # print row
            print y[im_index]

            if predicted_class == CLASSES[np.argmax(y[im_index])]:
                accuracy.append(1)
            else:
                accuracy.append(0)



        return (1.0 * sum(accuracy)/len(accuracy))*100




def main():
    process_obj = Preprocess()
    cnn_obj = CNNLite()
    process_obj = process_obj.process()
    X,y = process_obj.X_validation, process_obj.y_validation
    acc =  cnn_obj.test(X[:100],y[:100])
    print("accuracy is, "+str(acc)+" %")
    print "time taken ------------------- "+str(time.time()-start)



if __name__ == '__main__':
    main()
