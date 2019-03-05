import numpy as np


y = np.ones((4, 4))
y_pad = np.pad(y, (1, 1), 'constant', constant_values = 0)

w = np.ones((3, 3))
w = 0.1*w

output = np.zeros_like(y)
K = int((w.shape[0] - 1)/2)

y_next = np.zeros_like(y)

for i in range(y_next.shape[0]):
    for j in range(y_next.shape[1]):
        y_next[i, j] = i + j 

for i in range(y.shape[0]):
    for j in range(y.shape[1]):
        temp = 0
        for n in range(w.shape[0]):
            for m in range(w.shape[1]):
                temp += y_pad[i + n, j + m]*w[n, m]  
        output[j, i] = temp
y_next = y_next[np.newaxis,:]
y_next = y_next[np.newaxis,:]
print(y_next)

y_next_flipped = np.flip(y_next, axis = 2) 
print(y_next_flipped)

print(np.flip(y_next_flipped, axis = 3))



