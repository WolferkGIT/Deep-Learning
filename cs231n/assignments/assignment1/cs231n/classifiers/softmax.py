from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax(XS): # input is a vector, output is a vector aswell
    return np.exp(XS) / sum(np.exp(XS))

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]  # X: A numpy array of shape (N, D) containing a minibatch of data. num_train = N
    num_class = W.shape[1]  # W: A numpy array of shape (D, C) containing weights. num_class = D
    vec_loss = np.zeros([num_train,1])  # tensor of losses N x 1
    diff_W = np.zeros([num_train,num_class]) 
    
    for i in range(num_train):
      ten_scores = np.dot(X[i], W)  # apply our minibatch piece, vector [D] multiplied by [D,C], result is tensor [1xC]
      vec_val = ten_scores[y[i]]  # y[i] is a scalar, vec_val contains weights for the i-th dimension as [C]
      # dW as agradient also should be DxC
      
      for j in range(num_class):
        diff_W[j] += softmax [ten_scores[j]] @ X[i].transpose() #that would be our differntial [Dx1][1xC] = DxC ?
      
      vec_loss[i] = -np.log(softmax(vec_val))

    # pass
    loss = np.sum(vec_loss)/num_train + reg * np.sum(W*W)
    #average  for the batch + someregularisation
    dW = diff_W / num_train  + 2 * reg * W
    #average for the batch + some regularisation

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]  # X: A numpy array of shape (N, D) containing a minibatch of data. num_train = N
    num_class = W.shape[1]
    vec_val = np.dot(X,W) # NxD dot DxC = NxC
    vec_softmax = softmax(vec_val) # Getting normalized NxC tensor
    
    vec_row = vec_softmax[np.arange(num_train), y] #losses for each minibatch shaped into a vector 
    
    loss = - np.sum( np.log(vec_row))/num_train #averaging the loss
    diff_W = np.dot(X,vec_row)
    dW += diff_W / num_train  + 2 * reg * W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
