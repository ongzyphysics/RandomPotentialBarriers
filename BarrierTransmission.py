
# We generalize and implement the formula for the transfer matrix in
# D J Griffiths and C A Steinke, "Waves in locally periodic media", 
# Am. J. Phys. 69, 137–154 (2001)
# The key equations are (19) to (23), (41) to (42)

import numpy as np
import matplotlib.pyplot as plt
from numpy import random

from scipy.constants import Planck as hbar
from scipy.constants import m_e, eV, pi

# print(f"Planck constant (hbar) is {hbar}") 
# print(f"Electron mass (m_e) is {m_e}")
# print(f"Electron-volt (eV) is {eV}")

nm = 1E-9 # definition of nanometer

E_min = 0.00*eV # lower bound of energy spectrum
E_max = 2.00*eV # upper bound of energy spectrum

N_barrier = 20 # number of barriers, cannot be too large, limit to 200 max
a_barrier = 1.0*nm # half width of barrier
s_mean = 5.0*a_barrier # average interbarrier distance
delta_s = 0.0*a_barrier # fluctuation in interbarrier distance
v_mean = 0.50*eV # average barrier height
delta_v = 0.00*eV # fluctuation in barrier height

def Ekdispersion(v_in,a_in,s_in):
    Nbands = 4 # number of bands to plot
    Nx = 200
    a_latt = s_in + 2*a_in
    x = np.linspace(-0.5*a_latt,0.5*a_latt,Nx+1)     
    x = x[:-1]
    dx = x[1]-x[0] # grid spacing
    Vx = 0*x
    Vx[(x<1.0*a_in)*(x>-1.0*a_in)] = v_in
    V = np.diag(Vx)
    # plt.plot(x,Vx)
    # print(len(x))
    
    Ekdata = np.array([])
    
    K = 2*np.eye(Nx,dtype='complex') - np.roll(np.eye(Nx),1,axis=0) - \
        np.roll(np.eye(Nx),-1,axis=0)
    
    for k in np.linspace(0.00,1.00,101)*np.pi/a_latt:
        # k = 0.01*np.pi/s_in
        Hk = hbar*hbar/(2*m_e)/dx/dx*K        
        Hk[0,Nx-1] = Hk[0,Nx-1]*np.exp(1j*k*a_latt)
        Hk[Nx-1,0] = Hk[Nx-1,0]*np.exp(-1j*k*a_latt)

        [Dk, Uk] = np.linalg.eig(Hk)
        Dk.real.sort()
        Ek = Dk.real[0:Nbands]
        plt.plot(k*a_latt/2/np.pi,np.array([Ek])/eV,'b.')

        Hk = Hk + V        
        [Dk, Uk] = np.linalg.eig(Hk)
        Dk.real.sort()
        Ek = Dk.real[0:Nbands]
        plt.plot(k*a_latt/2/np.pi,np.array([Ek])/eV,'r.')
        plt.xlim([0,0.5])
        plt.ylim([0,2.0])
        plt.ylabel('Energy (eV)')
    # plt.show()    
    # f = hbar*hbar/(2*m_e)/dx/dx
    return Ek 

# fig, axs = plt.subplots(1, 2, figsize=(8, 6))
plt.subplot(1,2,1)

Ek = Ekdispersion(v_mean,a_barrier,s_mean)
# [Dk, Uk] = np.linalg.eig(Hk)
# print( Ekdispersion(v_mean,a_barrier,s_mean) )

# %%

s_list = s_mean - delta_s \
    + 2*delta_s*random.rand(N_barrier) # list of interbarrier distances
s_list = s_list \
    + (s_mean-s_list.mean()) # ensure that the mean distance is exactly s_mean
# s_list[0] is the distance between 0th and 1st barrier
v_list = v_mean - delta_v \
    + 2*delta_v*random.rand(N_barrier) # list of barrier heights
v_list = v_list \
    + (v_mean-v_list.mean()) # ensure that the mean height is exactly v_mean
# v_list[0] is the height of the 0th barrier
E_list = np.linspace(E_min,E_max,1001,dtype='float') \
    + 1E-10*eV # energy spectrum
M_list = np.zeros([N_barrier,2,2],dtype='complex')
t_list = np.zeros_like(E_list) # transmission spectrum

for idx, E in enumerate(E_list):    
    Ptotal = np.eye(2)
    InvP = np.eye(2)
    for n in range(N_barrier): # note the reversed order 
        V = v_list[n] # barrier height
        s = s_list[n] # distance to next barrier on right, \
            # doesn't matter for rightmost barrier
        # s = s -2*a_barrier
        if E>V:
            k_prime = np.sqrt(2*m_e*(E-V))/hbar
        else:
            k_prime = 1j*np.sqrt(2*m_e*(V-E))/hbar
        k = np.sqrt(2*m_e*(E-0))/hbar
        eta = k/k_prime
        eps_plus = 0.5*(eta+1/eta)
        eps_minus = 0.5*(eta-1/eta)
        M11 = (np.cos(2*k_prime*a_barrier) + \
               1j*eps_plus*np.sin(2*k_prime*a_barrier))            
        M12 = -1j*eps_minus*np.sin(2*k_prime*a_barrier)
        M22 = M11.conjugate()
        M21 = M12.conjugate()
        M = np.array([[M11, M12],[M21, M22]])        
        P = np.array([[np.exp(1j*k*s), 0],[0, np.exp(-1j*k*s)]])@M
        Ptotal = P@Ptotal        
        # M_list[n,:,:] = M 
        
    t = 1.0/np.abs(Ptotal[0,0])**2 # transmission coefficient
    t_list[idx] = t
    # print(E/eV,np.abs(np.linalg.det(P)))
    
t_list[np.isnan(t_list)] = 1e-200 # replace nan data at low energies
    

plt.subplot(1,2,2)
# Plot transmission data
plt.plot(t_list, E_list/eV, linewidth=1, marker='none',color=[0.0, 0.0, 0.5])
# plt.plot(E_list/eV, t_list, linewidth=1, marker='none',color=[0.0, 0.0, 0.5])
plt.xlim([0,1.0])
plt.ylim([0,2.0])
plt.ylabel('Energy (eV)')
plt.xlabel('Transmission')
plt.show()


