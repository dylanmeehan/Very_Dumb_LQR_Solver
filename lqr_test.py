import numpy as np
from pydrake.all import LinearQuadraticRegulator

A = np.array([[0, 1, 0], [0, 0, 1], [0,0,0]])
B = np.array([[0], [0], [1]])
Q = np.eye(3)
R = np.eye(1)

(K, S) = LinearQuadraticRegulator(A, B, Q, R)
print("K = " + str(K))
print("S = " + str(S))