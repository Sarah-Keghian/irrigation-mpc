import numpy as np
import control
import matplotlib.pyplot as plt

a = 0.9  # Sytem specific parameters representing natural moisture loss
b = 0.8  # Sytem specific parameters representing water absorption capacity
d = 2  # delay
# x_eq_initial = np.array([0 for _ in range(d + 1)]) # Retained for context, though overwritten
# x_eq_initial[d - 1] = 0.5 # Retained for context, though overwritten

def build_A(d):
    A = np.zeros((d + 1, d + 1))
    if d < 2:
        return
    if d >= 2:
        for i in range(1, d):
            A[i, 0] = 1.0

        A[-1, -1] = a
        A[-1, -2] = b
        return A


A = build_A(d)

B = np.zeros((d + 1, 1))
B[0] = 1
print(B.shape)
print(A)

diag_Q = [0.0 for _ in range(d + 1)]
diag_Q[d] = 1

Q = np.diag(diag_Q)
R = np.array([[1.0]]) # R should be 1x1 for scalar input 'u' for dlqr
K,V,L=control.dlqr(A, B, Q, R)
print(K)
print(Q)
x = np.array([[0.0 for _ in range(d + 1)]])
x[-1] = 0.7

x_eq=np.array([[0],[0],[1.4]]) # The desired equilibrium point, as a column vector

x=x.T # Transpose x to be a column vector (d+1, 1)
xk =[]
xk.append(x.flatten()) # Append the flattened 1D state vector
# ueq = np.linalg.pinv(B) @ (np.eye(d + 1) - A) @ x_eq

for k in np.arange(0,100):
  u = -K @ (x - x_eq)
  x = A @ x + B @ u
  xk.append(x.flatten()) # Append the flattened 1D state vector
#print(xk)

xk_spam = np.array(xk)
#print(xk_spam[:, -1]) # Use xk_spam for indexing after conversion to numpy array
plt.plot(xk_spam[:, -1]) # Plot using xk_spam
plt.show()