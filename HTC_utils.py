import numpy as np
import matplotlib.pyplot as plt
import time
import os

from scipy import signal

import igraph as ig

from statsmodels.tsa.stattools import acf

from tqdm.auto import tqdm
from IPython.display import clear_output

from numba import jit, prange
parallel = False

# ----------------- USEFUL FUNCTIONS -----------------
def normalize(W):
    ''' Normalize each entry in a matrix by the sum of its row'''
    return W / np.sum(W, axis=1)[:,None]

@jit(nopython=True, parallel=parallel)
def init_state(N, x0, y0, runs):
    '''
    Initialize the state of the system
    '''
    
    # create shuffled array
    states = np.zeros((runs,N))
    for i in prange(runs):
        # generate two random numbers
        n_act, n_ref = np.ceil( np.random.random(2) * np.array([x0,y0]) * N )
        # create array
        ss = np.zeros(N)
        ss[:n_act] = 1.
        ss[-n_ref:] = -1.
        # shuffle array
        states[i] = np.random.choice(ss, len(ss), replace=False)
        
    return states

@jit(nopython=True)
def update_state_single(S, T, r1, r2, dt):
    
    N = len(S)
    probs = np.random.random(N)                 # generate probabilities
    s = (S==1).astype(np.float64)               # get active nodes
    pA = ( r1 + (1.-r1) * ( np.mean(s)>T ) ) * dt    # prob. to become active

    # update state vector
    newS = ( (S==0)*(probs<pA)                  # I->A with prob pA
         + (S==1)*-1*(probs<dt)                 # A->R with prob dt
         + (S==1)*(probs>=dt)                   # remain A with prob 1-dt
         + (S==-1)*(probs>=r2*dt)*-1 )          # R->I (remain R with prob 1-r2*dt)

    return newS

@jit(nopython=True)
def update_state_single_matrix(S, W, T, r1, r2, dt):
    
    N = len(S)
    probs = np.random.random(N)                 # generate probabilities
    s = (S==1).astype(np.float64)               # get active nodes
    pA = ( r1 + (1.-r1) * ( (W@s)>T ) ) * dt    # prob. to become active

    # update state vector
    newS = ( (S==0)*(probs<pA)                  # I->A with prob pA
         + (S==1)*-1*(probs<dt)                 # A->R with prob dt
         + (S==1)*(probs>=dt)                   # remain A with prob 1-dt
         + (S==-1)*(probs>=r2*dt)*-1 )          # R->I (remain R with prob 1-r2*dt)

    return newS

@jit(nopython=True, parallel=parallel)
def update_state(S, W, T, r1, r2, dt):
    '''
    Update state of each runs
    '''
    runs = S.shape[0]
    newS = np.zeros((S.shape[0], S.shape[1]), dtype=np.float64)
    
    # Simulation step in parallel        
    for i in prange(runs):
        if W is None:
            newS[i] = update_state_single(S[i], T, r1, r2, dt)
        else:
            newS[i] = update_state_single_matrix(S[i], W, T, r1, r2, dt)
            
    return (newS, (newS==1).astype(np.int64))
    
@jit(nopython=True)
def run_htc_single(W, x0, y0, T, r1, r2, 
            dt, steps, N=0,
            runs=10, eq_steps=0):
    
    if W is not None:
        N = W.shape[0]
    else:
        if N==0:
            raise Exception('N is missing')
        
    # Create empty vector to store time-series
    x, y = np.zeros((runs, steps)), np.zeros((runs, steps))
    
    # Initial state
    S = init_state(N, x0, y0, runs=runs)
        
    if eq_steps>0:
        for i in prange(eq_steps):
            S, s = update_state(S, W, T, r1, r2, dt)
    
    # Loop over time steps
    for i in prange(steps):
        x[:,i], y[:,i] = np.sum((S==1), axis=1)/N, np.sum((S==-1), axis=1)/N
        S, s = update_state(S, W, T, r1, r2, dt)
        
    return x, y

def run_htc_hysteresis(W, dt, steps, Tmin, Tmax, N=None,
            r1=0.1, r2=0.1, runs=50, nperseg=1000, nT=40, eq_steps=int(1e3)):
    
    start = time.time()
    
    ### Initialize variables
    if W is not None:
        N = W.shape[0]
    else:
        if N is None:
            raise Exception('Insert N')
    Tminus = r1 * r2 / (r1 + r2 + r1*r2)
    Tplus = r2 / (2*r2 +1)

    xplus = Tplus
    yplus = Tplus / r2

    xminus = Tminus
    yminus = Tminus / r2
        
    ### Define Trange
    Trange = np.linspace(Tmin, Tmax, nT, endpoint=True)
    Trange = np.concatenate((Trange, Trange[:-1][::-1]))
    
    ### Initialize empty arrays
    A, sigmaA = np.zeros(len(Trange)), np.zeros(len(Trange))
    acorr, spectra = [], [] 
    
    # LOOP OVER Ts
    for i,T in enumerate(Trange):
        clear_output(wait=True)
        print('\n'+str(i+1) + '/'+ str(len(Trange)) + ' - T = ' +  str(round(T/Tplus, 2)) + ' * T+' )
        print('Simulating activity...')
        
        # Create empty array to store activity over time
        Aij = np.zeros((runs, steps, N))
        
        # Init activity only at the beginning
        if i==0:
            S = init_state(N, xplus, yplus, runs)
        
        # LOOP OVER TIME STEPS
        # Initial equilibration
        if eq_steps>0:
            print('Running equilibration...')
            for t in tqdm(range(eq_steps)):
                S, s = update_state(S, W, T, r1, r2, dt)
        
        print('Running simulation...')
        for t in tqdm(range(steps)):
            S, s = update_state(S, W, T, r1, r2, dt)
            Aij[:,t] = s
            
        # COMPUTE AVERAGES
        # Activity
        print('Computing activity...')
        At = np.mean(Aij, axis=2)    # node average <A(t)>
        A[i], sigmaA[i] = np.mean(At), np.mean( np.std(At, axis=1) )
        
        # AUTOCORRELATION
        print('Computing autocorrelation...')
        tmp_acorr = []
        for i, Ai in enumerate(At):
            single_acorr = acf(Ai, nlags = int(1e3))
            tmp_acorr.append(single_acorr)
        
        tmp_acorr = np.stack(tmp_acorr, axis=0) # from list to array
        tmp_acorr = np.mean(tmp_acorr, axis=0) # average over runs
        acorr.append(tmp_acorr)
        
        # POWER SPECTRUM
        print('Computing power spectrum...')
        spectrum = []
        for i, Ai in enumerate(At):
            f, ss = signal.welch(Ai, nperseg = nperseg, fs = 1/dt, scaling = 'density')
            spectrum.append(ss)
        
        spectrum = np.stack(spectrum, axis=0) # from list to array
        spectrum = np.mean(spectrum, axis=0) # average over runs
        spectra.append(spectrum)
        
    clear_output(wait=True)
    print('End simulating activity')
    print('Total computation time: {:.2f}s'.format(time.time()-start))
    
    return (Trange, A, sigmaA, np.stack(acorr, axis=0), f, np.stack(spectra, axis=0))


def save_results(name, results):
    folder = 'data/'
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    print('Saving results for '+str(name)+'...\n')
    
    Trange, A, sigmaA, acorr, f, spectra = results
    
    np.save(folder+str(name)+'_Trange.npy', Trange)
    np.save(folder+str(name)+'_A.npy', A)
    np.save(folder+str(name)+'_sigmaA.npy', sigmaA)
    np.save(folder+str(name)+'_acorr.npy', acorr)
    np.save(folder+str(name)+'_f.npy', f)
    np.save(folder+str(name)+'_spectra.npy', spectra)
    
    
# ----------------- MAIN -----------------
r1 = 0.1
r2 = 0.1

Tminus = r1 * r2 / (r1 + r2 + r1*r2)
Tplus = r2 / (2*r2 +1)

xplus = Tplus
yplus = Tplus / r2

xminus = Tminus
yminus = Tminus / r2

'''
ps = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8]
dt = 0.05
steps = int(5e4)
N = int(1e4)

# Run fully connected
#tmp = run_htc_hysteresis(None, dt, steps, N=N, Tmin=0.03, Tmax=0.10, nT=40, runs=1)
#save_results('N_'+str(N)+'_p_1', tmp)

for p in ps:
    # Generate graphs
    print('Generating topology for p={:.1f}'.format(p))
    top = ig.Graph.Erdos_Renyi(n=N, p=p).get_adjacency_sparse().toarray()
    # Homeostatic normalization
    top = normalize(top)
    # Run model
    tmp = run_htc_hysteresis(top, dt, steps, N=N, Tmin=0.03, Tmax=0.10, nT=40, runs=1)
    save_results('N_'+str(N)+'_p_{:.1f}'.format(p), tmp)
'''

'''
#Ns = [5e2, 1e3, 5e3, 1e4, 2e4, 3e4, 4e4, 5e4]
Ns = [3e3]
dt = 0.05
steps = int(5e4)

for N in Ns:
    tmp = run_htc_hysteresis(None, dt, steps, N=int(N), Tmin=0.03, Tmax=0.10, nT=40, runs=1)
    save_results('N_'+str(int(N))+'_p_1_r1_'+str(r1)+'_r2_'+str(r2), tmp)
'''

'''
r1 = 0.001
r2 = 0.2

for N in Ns:
    tmp = run_htc_hysteresis(None, dt, steps, N=int(N), Tmin=0., Tmax=0.2, nT=40, runs=1, r1=r1, r2=r2)
    save_results('N_'+str(int(N))+'_p_1_r1_'+str(r1)+'_r2_'+str(r2), tmp)
'''