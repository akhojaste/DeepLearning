###Implementation of Logistic Regression


import numpy as np
from numpy.ma.core import dot
from encodings.punycode import T
import math
import sys
from math import sin, cos, radians
import time
import matplotlib

sample_size = 50
feature_size = 3

X = np.random.rand(feature_size, sample_size )
w = np.zeros((feature_size, 1))
Y = np.random.rand(1, sample_size)  ##The actual output
##Y = np.ravel(Y)

##
#X.reshape(feature_size, sample_size)
#Y.reshape(1, sample_size)


b = 0
alpha = 0.5 ##learning rate

A = np.zeros((1, sample_size))

for i in range(1, 10000):
    #Demostration of Broadcasting in Python i.e. here the parameter b will be padded 
    #to a matrix of the same dimension of np.dot(w.T, X), just like MATLAB babe
    Z = np.dot(w.T, X) + b; 
    ##Change the matrix to be 1D instead of 2D
    #Z = np.ravel(Z)

    ##Sigmoid Function sigm(z) = 1/ (1 + e^(-z))
    ##The predicted output i.e. a or y_hat
    A = 1 / (1 + np.exp(-Z))

    ##Loss function, np.log is the natural logarithm or the lograrithm in base
    J = np.sum(-Y * np.log(A) + (1 - Y) * np.log(1 - A) );
    J /= sample_size

    if(np.abs(J) < 0.001) :
        break


    #print("Loss value: " + str(Loss))

    ## dz = dJ(w, b) / dz * dJ / da * da / dZ
    dZ = A - Y 
    ##dw = dJ(w, b) / da * da / dz * dz / dw
    ##dw1 = x1 * dz for each sample. The matrix below: dW contains the followin:
    ##dW[1] = x1*dz(1 or first observation) + x1*dz(2)  + x1*dz(3) + .... x1*dz(M or Mth observation) i.e.
    ##dW[i] = sum(x(i) * dz) summation is over the training samples or observations
    dW = np.dot(X, dZ.T) 
    ## OR dW = no.matmul(X, dZ.T)

    #w = np.ravel(w)

    db = np.sum(dZ) / sample_size
    w = w - alpha * dW
    b = b - alpha * db


##This line will change the shape of the array to be so called ranked 1 array
##So this line of code will fail : assert(Y.shape == (1, sample_size))
Y = np.ravel(Y)
A = np.ravel(A)

##To change the format of the matrix of rank 1 back to 2D
#Y = Y.reshape(1, sample_size)

for i in range(sample_size - 1):
    print("Actual output: " + str(Y[i]))
    print("Predicted output: " + str(A[i]) )
    

###Sample of doing an explicit for loop and vectorization. the results are amazing!

#m1 = np.zeros(sample_size)
#m2 = np.zeros(sample_size)
#c = 0
#tic = time.time()
#for i in range(sample_size):
#    c += m1[i] * m2[i]
#toc = time.time()
#print("Vectorized dot product took " + str(1000 * (toc - tic)) + " ms. Result: " + str(c))
#tic = time.time()
#c = 0
#for i in range(sample_size):
#    c += m1[i] * m2[i]
#toc = time.time()
#print("Compiled dot product took " + str(1000 * (toc - tic)) + " ms. Result: " + str(c))

##Python Broadcasting
a = np.random.randn(4, 3)
b = np.random.randn(3, 2)
c = a + b

##np.dot(a, b) is matrix multiplication of a and b whereas a * b is element wise multiplication
