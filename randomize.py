import numpy as np
import copy

from numba import jit, prange

steps = int(1e6)
#steps = int(1e2)
trials = 20
folder = 'randomized/'
seed = np.random.randint(1e6)

@jit(nopython=True)
def rewire(A):
    
    N = len(A)
    links = A.nonzero()
    n_links = len(links[0])
    
    # choose at random two links
    l1, l2 = np.random.choice(np.arange(n_links), size=2, replace=False)
    u1, u2 = links[0][l1], links[1][l1]
    u3, u4 = links[0][l2], links[1][l2]
    
    # if connection not already exist -> switch
    if A[u1,u4]==0 and A[u2,u3]==0:
        A[u1,u4], A[u2, u3] = A[u1,u2], A[u3, u4] # switch
        A[u4,u1], A[u3, u2] = A[u1,u4], A[u2, u3] # symmetrize
        
    return A

@jit(nopython=True)
def maslov_sneppen(A, steps):
    for _ in range(steps):
        A = rewire(A)
        
    return A

# Load matrix
Aij = np.loadtxt('connectome.txt')
Q = copy.deepcopy(Aij)

print('START RANDOMIZING...')
# Run many trials
for i in range(trials):
    print('[*] trial '+str(i+1))
    Q = maslov_sneppen(Q, steps=steps)
    # Save randomized matrix
    np.save(folder+str(seed)+'_'+str(i)+'.txt', Q)