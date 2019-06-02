# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 19:13:23 2019

@author: Nidhal
"""

import matplotlib.pyplot as plt
import pylab as pl
#%matplotlib inlinie
from sklearn.datasets import load_digits
import random
from sklearn import ensemble

class RandomForest:
    def __init__(self, data):
        self.data = data
        self.classifier = ensemble.RandomForestClassifier()
        self.m = len(self.data.images)
        self.X = self.data.images.reshape((self.m,-1))
        self.y = self.data.target
        self.split_data()
    
    def split_data(self):
        self.train_index = random.sample(range(self.m), int(self.m/5)) # return m/5 numbers of indexes for a list of nums between 0 and m
        print("length of sample_index ",len(self.train_index))
        self.valid_index = [i for i in range(self.m) if i not in self.train_index]
        print("length of valid_index ",len(self.valid_index)) 
        #sample and validation images:
        self.train_images = [self.X[i] for i in self.train_index]
        self.valid_images = [self.X[i] for i in self.valid_index]
        
        #sample and validation target:
        self.train_target = [self.y[i] for i in self.train_index]
        self.valid_target = [self.y[i] for i in self.valid_index]
    
    def train(self):
        self.classifier.fit(self.valid_images, self.valid_target)
        
    def accuracy(self):
        self.score = self.classifier.score(self.train_images, self.train_target)
        print(f"Accuracy of the Random Forest Classifier is : {self.score}")
        
    def predict(self,index):
        pl.gray()
        pl.matshow(random_forest.data.images[index])
        pl.show()
        prediction = self.classifier.predict([self.X[index]])
        print(f"prediction is => {prediction} ")
        
random_forest = RandomForest(data=load_digits())
random_forest.train()
random_forest.accuracy()


random_forest.predict(90)
#digits = load_digits()
##print(digits.images[0])
#
#
##pl.gray()
##pl.matshow(digits.images[1])
##pl.show()
#
#m = len(digits.images) # 1797 pictures
#
#x = digits.images.reshape((m,-1)) # size => 1797*64
#y = digits.target # size => 1797*1
#
##create random Indices:
#sample_index = random.sample(range(m), int(m/5)) # return m/5 numbers of indexes for a list of nums between 0 and m
#print("length of sample_index ",len(sample_index))
#valid_index = [i for i in range(m) if i not in sample_index]
#print("length of valid_index ",len(valid_index))
#
##sample and validation images:
#sample_images = [x[i] for i in sample_index]
#valid_images = [x[i] for i in valid_index]
#
##sample and validation target:
#sample_target = [y[i] for i in sample_index]
#valid_target = [y[i] for i in valid_index]
#
##using the random tree classifier:
#classifier = ensemble.RandomForestClassifier()
#
##fit model with sample data:
#classifier.fit(sample_images, sample_target)
#
##Attempt to predict validation data:
#score = classifier.score(valid_images, valid_target)
#print("Random Tree Classifier: \n ")
#print("score of the Algorithm => ", str(score))
#
#i = 150
#pl.gray()
#pl.matshow(digits.images[i])
#pl.show()
    

    


    
