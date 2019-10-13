import numpy as np
import kmeans
import common
import naive_em
import em


X = np.loadtxt("toy_data.txt")

# TODO: Your code here
K = np.array([1,2,3,4])
seeds = np.array([0,1,2,3,4])
for i in seeds:
    mixture = common.init(X, K[3], i)[0]
    post = common.init(X, K[3], i)[1]
    [mixture, post, cost] = kmeans.run(X, mixture, post)
#    common.plot(X, mixture, post)
    
    
     
      

