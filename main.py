import numpy as np
import solving_methods as sm
import time as time
import matplotlib.pyplot as plt


def create_equation_system_matrix(a1,a2,a3,N):
    A1 = np.diag([a1]*N,0)
    A2 = np.diag([a2]*(N-1),1)
    A3 = np.diag([a3]*(N-2),2)
    A = A1 + A2 + A3 + np.transpose(A2) + np.transpose(A3)  
    return A


#Task A

index = [1,7,5,7,5,7]
c = index[-2]
d = index[-1]
e = index[3]
f = index[2]

a1 = 5 + e
a2 = a3 = -1
N = 9*c*d

A = create_equation_system_matrix(a1,a2,a3,N)

b = [0]*N
for i in range(N):
    b[i] = np.sin((i)*(f+1))


#Task B
print("\nZadanie B:")

jac_it = sm.jacobi(A,b)
if (jac_it > -1):
    print("Ilość iteracji w metodzie Jacobiego: " + str(jac_it))
else:
    print("Metoda Jacobiego nie zbiega się dla podanego układu równań")

gs_it = sm.gauss_siedel(A,b)
if (gs_it > -1):
    print("Ilość iteracji w metodzie Gaussa-Seidla: " + str(jac_it))
else:
    print("Metoda Gaussa-Seidla nie zbiega się dla podanego układu równań")


#Task C
print("\nZadanie C:")

a1 = 3
a2 = a3 = -1
N = 9*c*d

A = create_equation_system_matrix(a1,a2,a3,N)

jac_it = sm.jacobi(A,b)
if (jac_it > -1):
    print("Ilość iteracji w metodzie Jacobiego: " + str(jac_it))
else:
    print("Metoda Jacobiego nie zbiega się dla podanego układu równań")

gs_it = sm.gauss_siedel(A,b)
if (gs_it > -1):
    print("Ilość iteracji w metodzie Gaussa-Seidla: " + str(gs_it))
else:
    print("Metoda Gaussa-Seidla nie zbiega się dla podanego układu równań")


#Task D
print("\nZadanie D:")
lu_norm_res = sm.lu(A, b)
print("Norma z residuum po rozwiązaniu układu równań metodą faktoryzacji LU: " + str(lu_norm_res))


#Methods comparison

jacobi_times = [0]*10
gauss_siedel_times = [0]*10
lu_times = [0]*10
matrix_sizes = [0]*10


b = [0]*2000
for i in range(400):
    b[i] = np.sin((i)*(f+1))

for i in range(10):
    N = (i+1)*40
    A = create_equation_system_matrix(a1,a2,a3,N)

    start_time = time.time()
    sm.jacobi(A, b[:N])
    jacobi_times[i] = time.time()-start_time

    start_time = time.time()
    sm.gauss_siedel(A, b[:N])
    gauss_siedel_times[i] = time.time()-start_time

    start_time = time.time()
    sm.lu(A, b[:N])
    lu_times[i] = time.time()-start_time

    matrix_sizes[i] = N

plt.plot(matrix_sizes, jacobi_times, label="metoda Jacobiego")
plt.plot(matrix_sizes, gauss_siedel_times, label="metoda Gaussa-Seidla")
plt.plot(matrix_sizes, lu_times, label="faktoryzacja LU")
plt.legend()
plt.xlabel("rozmiar macierzy")
plt.ylabel("czas obliczeń [s]")
plt.title("Porównanie szybkości działania trzech metod")
plt.show()


    