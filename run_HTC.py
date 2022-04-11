import numpy as np
import igraph as ig
from HTC_utils import *

# ----------------- MAIN -----------------
r1 = 0.1
r2 = 0.1

Tminus = r1 * r2 / (r1 + r2 + r1*r2)
Tplus = r2 / (2*r2 +1)

xplus = Tplus
yplus = Tplus / r2

xminus = Tminus
yminus = Tminus / r2


ps = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8]
pdfs = ['uniform', 'exp', 'normal']
dt = 0.05
steps = int(5e3)
N = int(1e4)

for pdf in pdfs:    
    # Generate graphs
    print('[*] Generating topology for pdf = ' + pdf)
    top = generate_random(pdf, N)
    # Homeostatic normalization
    top = normalize(top)
    
    # Run model
    tmp = run_htc_hysteresis(top, dt, steps, N=N, Tmin=0.03, Tmax=0.10, nT=40, runs=1)
    save_results('N_'+str(N)+'_p_1_pdf_'+str(pdf), tmp)

'''
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