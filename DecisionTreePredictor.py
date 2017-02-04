#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 23:30:01 2017

@author: cammilligan
"""

from sklearn import tree

#Sample data X is body measurements
X = [[181,80,44], [177,70,43],[160,60,38], [154, 54, 37],[166,65,40],[190,90,47],
     [175,64,39],[177,70,40],[159,55,37],[171,76,52],[181,85,43]]

#Sample data Y is related X's gender
Y = ['male','female','female','female','male','male','male','female','male','female','male']

#DecisionTreeClassifier trains on the data set
clf = tree.DecisionTreeClassifier()

clf = clf.fit(X,Y)

#using predict to load in a sample X
prediction = clf.predict([190,70,43])

#predict outputs the Y value
print(prediction)