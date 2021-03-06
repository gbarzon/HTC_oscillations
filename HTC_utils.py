import numpy as np
import matplotlib.pyplot as plt
import time
import os

from scipy.stats import truncnorm
from scipy import signal
from statsmodels.tsa.stattools import acf

from tqdm.auto import tqdm
from IPython.display import clear_output

from numba import jit, prange
parallel = False

# ----------------- USEFUL FUNCTIONS -----------------
def generate_random(pdf, N):
    if pdf == 'uniform':
        mat = np.random.uniform(size=(N,N))
    elif pdf == 'exp':
        lmbd = 12.5
        mat = np.random.exponential(scale=1/lmbd, size=(N,N))
    elif pdf == 'normal':
        mat = truncnorm.rvs(0, 1, size=(N,N))
    
    # Symmetrize
    mat = np.triu(mat, k=1)
    mat += mat.T
    
    return mat

def normalize(W):
    ''' Normalize each entry in a matrix by the sum of its row'''
    return W / np.sum(W, axis=1)[:,None]

@jit(nopython=True)
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

@jit(nopython=True)
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

def run_htc_hysteresis(W, dt, steps, Tmin, Tmax, r1, r2, 
                       N=None, runs=50, nperseg=1000, Tdiv_log=False,
                       nT=40, eq_steps=int(1e3), display=False, hist=True):
    
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
    if Tdiv_log:
        Trange = np.logspace(np.log10(Tmin), np.log10(Tmax), nT, endpoint=True)
    else:
        Trange = np.linspace(Tmin, Tmax, nT, endpoint=True)
        
    if hist:
        Trange = np.concatenate((Trange, Trange[:-1][::-1]))
    
    ### Initialize empty arrays
    A, sigmaA, errA = np.zeros(len(Trange)), np.zeros(len(Trange)), np.zeros(len(Trange))
    spectra, sigma_spectra = [], [] 
    acorr = []
    
    # LOOP OVER Ts
    for i,T in enumerate(Trange):
        if display:
            clear_output(wait=True)
        print('\n'+str(i+1) + '/'+ str(len(Trange)) + ' - T = ' +  str(round(T/Tplus, 2)) + ' * T+' )
        print('Simulating activity...')
        
        # Create empty array to store activity over time
        Aij = np.zeros((runs, steps, N))
        
        # Init activity only at the beginning
        if hist==True:
            if i==0:
                S = init_state(N, xplus, yplus, runs)
        else:
            S = init_state(N, xplus, yplus, runs)
        
        # LOOP OVER TIME STEPS
        # Initial equilibration
        if eq_steps>0:
            print('Running equilibration...')
            if display:
                for t in tqdm(range(eq_steps)):
                    S, s = update_state(S, W, T, r1, r2, dt)
            else:
                for t in range(eq_steps):
                    S, s = update_state(S, W, T, r1, r2, dt)
        
        print('Running simulation...')
        if display:
            for t in tqdm(range(steps)):
                S, s = update_state(S, W, T, r1, r2, dt)
                Aij[:,t] = s
        else:
            for t in range(steps):
                S, s = update_state(S, W, T, r1, r2, dt)
                Aij[:,t] = s
            
        # COMPUTE AVERAGES
        # Activity
        print('Computing activity...')
        At = np.mean(Aij, axis=2)    # node average <A(t)>
        A[i], sigmaA[i], errA[i] = np.mean(At), np.mean( np.std(At, axis=1) ), np.sqrt( np.mean( np.std(At, axis=1)**2 ) )
        
        # AUTOCORRELATION
        print('Computing autocorrelation...')
        tmp_acorr = []
        for i, Ai in enumerate(At):
            single_acorr = acf(Ai, nlags = int(1e3))
            tmp_acorr.append(single_acorr)
        
        tmp_acorr = np.stack(tmp_acorr, axis=0) #??from list to array
        tmp_acorr = np.mean(tmp_acorr, axis=0) # average over runs
        acorr.append(tmp_acorr)
        
        '''
        # POWER SPECTRUM
        print('Computing power spectrum...')
        spectrum = []
        for i, Ai in enumerate(At):
            f, ss = signal.welch(Ai, nperseg = nperseg, fs = 1/dt, scaling = 'density')
            spectrum.append(ss)
        
        spectrum = np.stack(spectrum, axis=0) #??from list to array
        spectrum, sigma_spectrum = np.mean(spectrum, axis=0), np.std(spectrum, axis=0) # average over runs
        spectra.append(spectrum)
        sigma_spectra.append(sigma_spectrum)
        '''
    
    if display:
        clear_output(wait=True)
    print('End simulating activity')
    print('Total computation time: {:.2f}s'.format(time.time()-start))
    
    #return (Trange, A, sigmaA, np.stack(acorr, axis=0), f, np.stack(spectra, axis=0))
    return (Trange, A, sigmaA, acorr)


def save_results(name, results):
    folder = 'data/'
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    print('Saving results for '+str(name)+'...\n')
    
    #Trange, A, sigmaA, acorr, f, spectra = results
    Trange, A, sigmaA, errA = results
    
    np.save(folder+str(name)+'_Trange.npy', Trange)
    np.save(folder+str(name)+'_A.npy', A)
    np.save(folder+str(name)+'_sigmaA.npy', sigmaA)
    np.save(folder+str(name)+'_errA.npy', sigmaA)
    #np.save(folder+str(name)+'_acorr.npy', acorr)
    #np.save(folder+str(name)+'_f.npy', f)
    #np.save(folder+str(name)+'_spectra.npy', spectra)