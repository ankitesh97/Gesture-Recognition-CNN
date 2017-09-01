
import numpy as np
import cv2
import random
import json
import os
import pickle


params_file  = open('params.json','r')
params = json.load(params_file)
params_file.close()
preprocess_params = params['preprocess']
BASE_FOLDER = preprocess_params['BASE']
FIST = preprocess_params['FIST']
HAND = preprocess_params['HAND']
NONE = preprocess_params['NONE']
ONE = preprocess_params['ONE']
PEACE = preprocess_params['PEACE']
ALL_IMAGES_PATHS = [FIST,HAND,ONE,PEACE, NONE] #change here if you want to add new gesture

random.seed(0)

class Preprocess():

    def __init__(self):
        self.images = []    #this will contain list of all images preprocessed
        self.n_types = preprocess_params['n_types']
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_validation = None
        self.y_validation = None
        #total in normal arrays



    #controller function this will return preprocessed value of X and also their outputs
    #data type is train or test or validation
    def process(self):
        #if the sampled data exists then simply read and return
        try:
            pickle_file_sampled_data = open('pickle_models/sampled_data','r')
            processed_obj = pickle.load(pickle_file_sampled_data)
            pickle_file_sampled_data.close()
            return processed_obj
        #basically this will sample and save it then read from there and return it
        except Exception as e:
            print "the pickled file not found pickling now ......."
        tX_train = []
        tX_test = []
        tX_validation = []
        ty_train = []
        ty_test = []
        ty_validate = []
        #for all types
        for curr_type in range(self.n_types):
            local_X, local_y = self.loadAllImages(ALL_IMAGES_PATHS[curr_type],curr_type)
            lX_train,lX_test,lX_validate,ly_train,ly_test,ly_validate = self.shuffleAndSplit(local_X,local_y)
            #adds to main array
            [tX_train.append(x) for x in lX_train]
            [tX_test.append(x) for x in lX_test]
            [tX_validation.append(x) for x in lX_validate]
            [ty_train.append(x) for x in ly_train]
            [ty_test.append(x) for x in ly_test]
            [ty_validate.append(x) for x in ly_validate]


        #convert into numpy array
        self.X_train = np.dstack(tX_train)
        self.X_test = np.dstack(tX_test)
        self.X_validation = np.dstack(tX_validation)
        self.y_train = np.dstack(ty_train)
        self.y_test = np.dstack(ty_test)
        self.y_validation = np.dstack(ty_validate)

        #rotate the axis
        self.X_train = np.rollaxis(self.X_train,-1)
        self.X_test = np.rollaxis(self.X_test,-1)
        self.X_validation = np.rollaxis(self.X_validation,-1)
        self.y_train = np.rollaxis(self.y_train,-1)
        self.y_test = np.rollaxis(self.y_test,-1)
        self.y_validation = np.rollaxis(self.y_validation,-1)

        #now just shuffle
        self.shuffleAll()

        return None



    def shuffleAll(self):
        train = np.random.permutation(len(self.X_train))
        self.X_train, self.y_train = self.X_train[train],self.y_train[train]
        test = np.random.permutation(len(self.X_test))
        self.X_test, self.y_test = self.X_test[test],self.y_test[test]
        validation = np.random.permutation(len(self.X_validation))
        self.X_validation, self.y_validation = self.X_validation[validation],self.y_validation[validation]


    #returns all images in a given folder
    @staticmethod
    def loadAllImages(path,curr_type):

        file_y = open(path+'y.txt','r')
        y_to_append = map(int,file_y.read().strip().split(' '))
        file_y.close()
        temp_X = []
        temp_y = []
        for f in os.listdir(path):
            if(f!='y.txt'):
                img = cv2.imread(path+str(f),cv2.COLOR_BGR2GRAY)
                ret,binary_img = cv2.threshold(img,120,255,cv2.THRESH_BINARY)
                binary_img[binary_img==255] = 1 #replace 255 with 1 for easy calc, 1 means white , 0 means black
                temp_X.append(binary_img)
                temp_y.append(y_to_append)


        return temp_X, temp_y

    @staticmethod
    def shuffleAndSplit(X,y):
        l = len(X)
        random.shuffle(X)
        random.shuffle(y)
        l_train = int(preprocess_params['train']*l)
        l_test = int(preprocess_params['test']*l)
        upper = l_train + l_test
        return X[0:l_train], X[l_train:upper], X[upper:], y[0:l_train], y[l_train:upper], y[upper:]




if __name__ == '__main__':
    process_obj = Preprocess()
    obj = process_obj.process()
    if(obj == None):
        pickle_file_sampled_data = open('pickle_models/sampled_data','w')
        pickle.dump(process_obj,pickle_file_sampled_data)
        pickle_file_sampled_data.close()
    else:
        print("stuff loaded")
        # img =  obj.X_validation[0]
        # img[img == 1] = 255
        # for i in range(50):
        #     print img[i]
        #     print ""
        # print obj.y_validation[0]
        # cv2.imshow("image",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
