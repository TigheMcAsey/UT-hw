import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.animation as animation

#Input data
X = np.array([1/np.sqrt(2),1])
R = 500

#Write Generators for SL(2,Z), we could hope more generally to compute the generators of a given lattice.
S = np.array([[0, -1], [1, 0]])
T = np.array([[1, 1], [0, 1]])
Tinv = np.array([[1, -1], [0, 1]])

def norm(A):
    return np.linalg.norm(A)

def rot(t):
    return np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])

def gen_lattice(R): #Returns all lattice elements with norm less than R, namely takes as input a positive real number and returns a list of matrices
    N = math.floor(np.sqrt(R))
    lattice = []
    for i in range(-N-2,N+2):
        for j in range(-N-2,N+2):
            for k in range(-N-2,N+2):
                for l in range(-N-2,N+2):
                    A = np.array([[i,j],[k,l]])
                    if norm(A) <= R and i*l - j*k == 1:
                        lattice.append(A)
    return lattice

def h(s):
    return np.array([[1,s],[0,1]])

def mobius(A,X): #Returns the vector given by A acting on X via mobius transform
    a, b = A[0]
    c,d = A[1]
    x1, x2 = X
    denom = np.square(c*x1 + d) + np.square(c*x2)
    renum = a*c*np.square(norm(X)) + a*d*x1 + b*c*x1 + b*d
    Re = renum/denom
    Im = x2/denom
    return np.array([Re, Im])

def findtranslation(X): #Finds n, such that Re(T^n X) is in (-1/2,1/2]
    x1, x2 = X
    idx = 1
    n = 0
    if x1 > 0:
        idx = -1
    while x1 <= -0.5 or x1 >0.5:
        n += idx
        x1 += idx
    return n

def quotient(A):
    G = np.array([[1,0],[0,1]])
    X = np.array([0,1])
    Y = mobius(A,X)
    while Y[0] > 0.5 or Y[0] <= -0.5 or norm(Y) < 1:
        if Y[0] > 0.5 or Y[0] <= -0.5:
            n = findtranslation(Y)
            G = np.array([[1, n],[0,1]])@G
            Y = mobius(np.array([[1, n],[0,1]]),Y)
        else:
            G = S@G
            Y = mobius(S,Y)
    return G@A

def psi(X): #returns the matrix Psi(X)
    x1,x2 = X * 1/norm(X)
    t = np.arctan2(x2,x1)
    R = rot(t)
    Psi = R@np.array([[norm(X),0],[0,1/norm(X)]])
    return Psi

def global_plot(R,X):
    lattice = gen_lattice(R)
    arr1 = list(map(lambda A: A@psi(X), lattice))
    arr1x = list(map(lambda A: A[0], arr1))
    arr1y = list(map(lambda A: A[1], arr1))
    arr2 = list(map(lambda A: psi(A@X), lattice))
    arr2x = list(map(lambda A: A[0], arr2))
    arr2y = list(map(lambda A: A[1], arr2))
    plt.scatter(arr1x,arr1y, color = 'red')
    plt.scatter(arr2x,arr2y, color = 'blue')
    plt.show()

def quot_plot(R,X):
    lattice = gen_lattice(R)
    arr1 = list(map(lambda A: mobius(quotient(A@psi(X)),np.array([0,1])), lattice))
    arr1x = list(map(lambda A: A[0], arr1))
    arr1y = list(map(lambda A: A[1], arr1))
    arr2 = list(map(lambda A: mobius(quotient(psi(A@X)),np.array([0,1])), lattice))
    arr2x = list(map(lambda A: A[0], arr2))
    arr2y = list(map(lambda A: A[1], arr2))
    plt.scatter(arr2x,arr2y, color = 'blue', s = 1)
    plt.scatter(arr1x,arr1y, color = 'red', s = 1)
    x = np.linspace(-0.5,0.5,200)
    plt.plot(x,np.sqrt(1-x**2), color = 'black')
    plt.vlines(x=-0.5, ymin = np.sqrt(3)/2,ymax = 50, color = 'black')
    plt.vlines(x=0.5, ymin = np.sqrt(3)/2,ymax = 50, color = 'black')
    plt.xlim(-0.55,0.55)
    plt.ylim(0.8,2.5)
    plt.show()

def quot_plot2(X):
    val = mobius(quotient(psi(X)),np.array([0,1]))
    plt.scatter(val[0],val[1], color = 'red',s = 2)
    x = np.linspace(-0.5,0.5,200)
    plt.plot(x,np.sqrt(1-x**2), color = 'black')
    plt.vlines(x=-0.5, ymin = np.sqrt(3)/2,ymax = 50, color = 'black')
    plt.vlines(x=0.5, ymin = np.sqrt(3)/2,ymax = 50, color = 'black')
    plt.xlim(-0.55,0.55)
    plt.ylim(0.8,2.5)
    plt.show()

def quot_flow(A,svec):
    return list(map(lambda s: mobius(quotient(A@h(s)),np.array([0,1])), svec))

def update_flow(frame):
    #line.set_xdata(list(map(lambda Z: Z[0], Zvec[:frame])))
    #line.set_ydata(list(map(lambda Z: Z[1], Zvec[:frame])))
    data = np.stack([list(map(lambda Z: Z[0], Zvec[:frame])), list(map(lambda Z: Z[1], Zvec[:frame]))]).T
    scat.set_offsets(data)
    return scat

quot_plot(100,np.array([1/np.sqrt(2),1]))

t = np.linspace(0,R,math.floor(500*R))
Zvec = quot_flow(psi(X),t)
fig, ax = plt.subplots()
x = np.linspace(-0.5,0.5,200)
ax.plot(x,np.sqrt(1-x**2), color = 'black')
ax.vlines(x=-0.5, ymin = np.sqrt(3)/2,ymax = 50, color = 'black')
ax.vlines(x=0.5, ymin = np.sqrt(3)/2,ymax = 50, color = 'black')
ax.set(xlim= (-0.55,0.55), ylim = (0.8,10))
#line = ax.plot(mobius(quotient(psi(X)),[1,0]))[0]
scat = ax.scatter(mobius(quotient(psi(X)),[1,0])[0],mobius(quotient(psi(X)),[1,0])[1])
ani = animation.FuncAnimation(fig=fig, func=update_flow, frames=math.floor(10*R), interval=30)
plt.show()