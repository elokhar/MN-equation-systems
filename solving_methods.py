import numpy as np

def jacobi(A,b):  
    max_it = 1000
    D = np.diag(np.diag(A))
    U = np.triu(A,1)
    L = np.tril(A,-1)
    r = [1]*len(b)
    Db = forward_sub(D,b)
    for i in range(max_it):
        Dr = forward_sub(D, r)
        DLUr = np.matmul(L+U, Dr)
        r = np.subtract(Db, DLUr)
        
        Ar = np.matmul(A,r)   #ok
        res = np.subtract(Ar, b)
        norm_res = vector_norm(res)
        if (norm_res) < (10**(-9)):
            break   
        elif norm_res > 10**6:
            return -1     
    return i

def gauss_siedel(A,b):
    max_it = 1000
    D = np.diag(np.diag(A))
    U = np.triu(A,1)
    L = np.tril(A,-1)
    r = [1]*len(b)
    DLb = forward_sub(D+L,b)
    for i in range(max_it):
        Ur = np.matmul(U,r)
        DLUr = forward_sub(D+L,Ur)
        r = np.subtract(DLb, DLUr)

        Ar = np.matmul(A,r)   #ok
        res = np.subtract(Ar, b)
        norm_res = vector_norm(res)
        if norm_res < (10**(-9)):
            break
        elif norm_res > 10**6:
            return -1     
    return i

def lu(A,b):
    m = len(b)
    U = np.copy(A).astype(float)
    L = np.eye(m).astype(float)
    for k in range(0,m-1):
        for j in range(k+1,m):
            L[j][k] = U[j][k]/U[k][k]
            C3 = np.multiply(L[j][k],U[k][k:m])
            C2 = U[j][k:m] - C3
            U[j][k:m] = C2.copy()
        
    y = forward_sub(L, b)
    x = backward_sub(U, y)

    res = np.matmul(A,x)-b

    return vector_norm(res)    

def forward_sub(L, b):  #ok
    x  = [0]*len(b)
    for i in range (len(x)):
        x[i]=b[i]
        for j in range (i):
            x[i]=x[i]-L[i][j]*x[j]
        x[i] = x[i]/L[i][i]
    return x

def backward_sub(U, b):  #ok
    x = [0]*len(b)
    for i in reversed(range(len(x))):
        x[i]=b[i]
        for j in range(i+1,len(x)):
            x[i]=x[i]-U[i][j]*x[j]
        x[i] = x[i]/U[i][i]
    return x



def vector_norm(x): #ok
    norm = 0
    for elem in x:
        norm += elem**2
    norm = np.sqrt(norm)
    return norm

