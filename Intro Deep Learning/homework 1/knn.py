import math
import numpy as np  
from download_mnist import load
import operator  
import time
# classify using kNN  
#x_train = np.load('../x_train.npy')
#y_train = np.load('../y_train.npy')
#x_test = np.load('../x_test.npy')
#y_test = np.load('../y_test.npy')
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000,28,28)
x_test  = x_test.reshape(10000,28,28)
x_train = x_train.astype(float)
x_test = x_test.astype(float)
def kNNClassify(newInput, dataSet, labels, k): 
    result=[]
    ########################
    # Input your code here #
    ########################
    
    # Loop over each test point
    for test_point in newInput:
         # Calculate Euclidean distances between test_point and all points in dataSet
         # dataSet shape: (N, 28, 28), test_point shape: (28, 28)
         # diff shape: (N, 28, 28) due to broadcasting
        diff = dataSet - test_point
        sq_diff = diff ** 2
        # Sum over the last two dimensions (pixels)
        sq_distances = sq_diff.sum(axis=(1, 2))
        distances = sq_distances ** 0.5
        
        # Sort distances and get indices of k nearest neighbors
        sorted_dist_indices = distances.argsort()
        k_nearest_indices = sorted_dist_indices[:k]
        
        # Get labels of k nearest neighbors
        class_votes = {}
        for idx in k_nearest_indices:
            vote_label = labels[idx]
            class_votes[vote_label] = class_votes.get(vote_label, 0) + 1
            
        # Find the label with the most votes
        sorted_votes = sorted(class_votes.items(), key=lambda item: item[1], reverse=True)
        result.append(sorted_votes[0][0])

    ####################
    # End of your code #
    ####################
    return result

start_time = time.time()
outputlabels=kNNClassify(x_test[0:20],x_train,y_train,10)
result = y_test[0:20] - outputlabels
result = (1 - np.count_nonzero(result)/len(outputlabels))
print ("---classification accuracy for knn on mnist: %s ---" %result)
print ("---execution time: %s seconds ---" % (time.time() - start_time))
