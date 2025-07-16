
# We generalize and implement the formula for the transfer matrix from Chapter 
# 2 of Wave Propagation by Markos and Soukoulis. 
# See equations (2.17) to (2.21) for details
# Each barrier has a width of 2*a_n and a height of v_n while the edge-to-edge
# distance between the n-th and (n+1)-th barrier is s_n
# for n = 1,...,N_barrier

import numpy as np
import matplotlib.pyplot as plt
from numpy import random

from scipy.constants import Planck as hbar
from scipy.constants import m_e, eV, pi

# print(f"Planck constant (hbar) is {hbar}") 
# print(f"Electron mass (m_e) is {m_e}")
# print(f"Electron-volt (eV) is {eV}")

nm = 1E-9 # definition of nanometer

# These are simulation parameters that can be tuned
E_min = 0.00*eV # lower bound of energy spectrum
E_max = 2.00*eV # upper bound of energy spectrum
N_barrier = 21 # number of barriers, cannot be too large, limit to 200 max
a_mean = 1.0*nm # average half width of barrier
delta_a = 0.0*nm # fluctuation in half width of barrier
s_mean = 3.0*nm # average interbarrier (edge-to-edge) distance
delta_s = 0.5*nm # fluctuation in interbarrier distance
v_mean = 0.50*eV # average barrier height
delta_v = 0.00*eV # fluctuation in barrier height

def Ekdispersion(v_in,a_in,s_in):
    # Compute and plot energy dispersion for infinite periodic array of identical barriers 
    Nbands = 4 # number of bands to plot
    Nx = 200
    a_latt = s_in + 2*a_in
    x = np.linspace(-0.5*a_latt,0.5*a_latt,Nx+1)     
    x = x[:-1]
    dx = x[1]-x[0] # grid spacing
    Vx = np.zeros_like(x)
    Vx[(x<1.0*a_in)*(x>-1.0*a_in)] = v_in
    V = np.diag(Vx)
    
    Ekdata = np.array([])
    
    K = 2*np.eye(Nx,dtype='complex') - np.roll(np.eye(Nx),1,axis=0) - \
        np.roll(np.eye(Nx),-1,axis=0)
    
    for nk,k in enumerate( np.linspace(0.00,1.00,101)*np.pi/a_latt ):        
        Hk = hbar*hbar/(2*m_e)/dx/dx*K        
        Hk[0,Nx-1] = Hk[0,Nx-1]*np.exp(1j*k*a_latt)
        Hk[Nx-1,0] = Hk[Nx-1,0]*np.exp(-1j*k*a_latt)

        [Dk, Uk] = np.linalg.eig(Hk)
        Dk.real.sort()
        Ek = Dk.real[0:Nbands]
        plt.plot(k*a_latt/2/np.pi,np.array([Ek])/eV,'b.',ms=2)
        if nk==0:            
            plt.plot(k*a_latt/2/np.pi,np.array([Ek[0]])/eV,'b.',\
                    ms=2,label='Free')

        Hk = Hk + V        
        [Dk, Uk] = np.linalg.eig(Hk)
        Dk.real.sort()
        Ek = Dk.real[0:Nbands]        
        plt.plot(k*a_latt/2/np.pi,np.array([Ek])/eV,'r.',ms=2)
        if nk==0:
            plt.plot(k*a_latt/2/np.pi,np.array([Ek[0]])/eV,'r.',\
                     ms=2,label='Infinite array')
            
    plt.xlim([0,0.5])
    plt.ylim([E_min/eV,E_max/eV])
    plt.xlabel('Normalized wave number')
    plt.ylabel('Energy (eV)')
    plt.legend(loc='upper left')    
    return Ek 

plt.figure()
plt.subplot(1,2,1)
Ek = Ekdispersion(v_mean,a_mean,s_mean)

# %%
import time

all_t_list = []

t0 = time.perf_counter()


# Feel free to modify a_list, s_list and v_list to tune the properties of the 
# potential barriers 

for nsample in range(20): # loop over samples for obtaining averages        
    a_list = a_mean - delta_a \
        + 2*delta_a*random.rand(N_barrier) # list of barrier half widths
    a_list = a_list \
        + (a_mean-a_list.mean()) # ensures the mean half widths is exactly a_mean            
    s_list = s_mean - delta_s \
        + 2*delta_s*random.rand(N_barrier) # list of interbarrier distances
    s_list = s_list \
        + (s_mean-s_list.mean()) # ensures the mean distance is exactly s_mean
    # s_list[0] is the distance between 0th and 1st barrier    
    v_list = v_mean - delta_v \
        + 2*delta_v*random.rand(N_barrier) # list of barrier heights
    v_list = v_list \
        + (v_mean-v_list.mean()) # ensures the mean height is exactly v_mean
    # v_list[0] is the height of the 0th barrier      
    E_list = np.linspace(E_min,E_max,1001,dtype='float') \
        + 1E-10*eV # energy spectrum
    t_list = np.zeros_like(E_list) # transmission spectrum    


    for idx, E in enumerate(E_list):    
        Ptotal = np.eye(2)
        InvP = np.eye(2)
        for n in range(N_barrier): # note the reversed order 
            V = v_list[n] # barrier height
            s = s_list[n] # distance to next barrier on right, \
                # doesn't matter for rightmost barrier
            a = a_list[n]
                        
            if E>V:
                k_prime = np.sqrt(2*m_e*(E-V))/hbar
            else:
                k_prime = 1j*np.sqrt(2*m_e*(V-E))/hbar
            k = np.sqrt(2*m_e*(E-0))/hbar
            eta = k/k_prime
            eps_plus = 0.5*(eta+1/eta)
            eps_minus = 0.5*(eta-1/eta)
            M11 = (np.cos(2*k_prime*a_list[n]) + \
                   1j*eps_plus*np.sin(2*k_prime*a))            
            M12 = -1j*eps_minus*np.sin(2*k_prime*a)
            M22 = M11.conjugate()
            M21 = M12.conjugate()
            M = np.array([[M11, M12],[M21, M22]]) 
            # This is the transfer matrix through the n-th barrier        
            P = np.array([[np.exp(1j*k*s), 0],[0, np.exp(-1j*k*s)]])@M
            # Multiply by phase factors aasociated with interbarrier region            
            Ptotal = P@Ptotal                
            
        t = 1.0/np.abs(Ptotal[0,0])**2 # transmission coefficient
        t_list[idx] = t
            
    t_list[np.isnan(t_list)] = 1e-200 # replace nan data at low energies

    if nsample==0:
        all_t_list = np.array([t_list])
    else:        
        all_t_list = np.vstack((all_t_list,t_list))
    t = time.perf_counter()
    print(f"Sample {nsample+1} finished at time = {t-t0} s")

t_list = np.mean(all_t_list,axis=0)

plt.subplot(1,2,2)
# Plot transmission data
plt.plot(t_list, E_list/eV, linewidth=1, marker='none',color=[0.0, 0.0, 0.5])
plt.xlim([0,1.0])
plt.ylim([E_min/eV,E_max/eV])
# plt.ylabel('Energy (eV)')
plt.xlabel('Average Transmission')
plt.show()


