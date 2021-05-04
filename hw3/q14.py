##for question 15
import numpy as np
import random
import py_compile
import matplotlib.pyplot as plt

#main
data = np.genfromtxt("hw3_train.dat.txt")

result = []
X = data[:, : -1]
y = data[:, -1]
y = y.reshape([X.shape[0], 1])
X = np.c_[(1 * np.ones(X.shape[0]), X)]
X_p = np.linalg.pinv(X)
w_lin = X_p.dot(y)

E_in_sqr = 0
for i in range(X.shape[0]):
    x_i = np.array(X[i, :])
    x_i = x_i.reshape((1, X.shape[1]))
    E_in_sqr += (np.dot(x_i,w_lin) - y[i]) * (np.dot(x_i,w_lin) - y[i])

E_in_sqr /= X.shape[0]
print(E_in_sqr)


for i in range(1000):
    w = np.zeros((X.shape[1], 1), np.float)
    t = 0 #count the number of iteration
   
    while True:
        t += 1
        k = random.randint(0, X.shape[0]-1)
        x_k = np.array(X[k, :])
        x_k = x_k.reshape((1, X.shape[1]))

        w_gradient=np.zeros(shape=(X.shape[1], 1))
        prediction=np.dot(x_k, w)
        w_gradient = (2)*(y[k]-(prediction)) * x_k.T
        w +=  0.001 * w_gradient
        

        E_in = 0
        for i in range(X.shape[0]):
            x_i = np.array(X[i, :])
            x_i = x_i.reshape((1, X.shape[1]))
            E_in += (np.dot(x_i,w) - y[i]) * (np.dot(x_i,w) - y[i])

        E_in /= X.shape[0]
        #print("current MSE :" ,E_in)
        
        if E_in <= 1.01 * E_in_sqr:
            print(t)
            break
                      
    result.append(t)
                  
plt.hist(result)
plt.show()
print("The average number of iteration is : ", np.mean(result))






