import numpy as np
import matplotlib.pyplot as plt

A = np.array([
        [-1, 3],
        [3, 2]
    ], dtype=np.dtype(float))

b = np.array([7, 1], dtype=np.dtype(float))

print("Matrix A:")
print(A)
print("\nArray b:")
print(b)

x = np.linalg.solve(A, b)         # Solve the equation

print(f"Solution: {x}")

d = np.linalg.det(A)

print(f"Determinant of matrix A: {d:.2f}")

A = np.array([
        [4, -3, 1],
        [2, 1, 3],
        [-1, 2, -5]
    ], dtype=np.dtype(float))

b = np.array([-10, 0, 17], dtype=np.dtype(float))

print("Matrix A:")
print(A)
print("\nArray b:")
print(b)

print(f"Shape of A: {np.shape(A)}")
print(f"Shape of b: {np.shape(b)}")

x = np.linalg.solve(A, b)

print(f"Solution: {x}")

# np.linalg.solve(A, b) will show an error: Singular Matrix if matrix is singular i.e it's determinant is 0 that it has no unique solution