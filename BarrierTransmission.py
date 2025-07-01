"""
We generalize and implement the formula for the transfer matrix in 
D J Griffiths and C A Steinke, "Waves in locally periodic media", 
Am. J. Phys. 69, 137–154 (2001)
The key equations are (19) to (23), (41) to (42)
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import random

from scipy.constants import Planck as hbar
from scipy.constants import m_e, eV, pi

# print(f"Planck constant (hbar) is {hbar}") 
# print(f"Electron mass (m_e) is {m_e}")
# print(f"Electron-volt (eV) is {eV}")

nm = 1E-9 # definition of nanometer

E_min = 0.0*eV # lower bound of energy spectrum
E_max = 2.0*eV # upper bound of energy spectrum

N_barrier = 20 # number of barriers, cannot be too large, limit to 200 max
a_barrier = 1.0*nm # half width of barrier
s_mean = 5.0*a_barrier # average interbarrier distance
delta_s = 0.0*a_barrier # fluctuation in interbarrier distance
v_mean = 0.50*eV # average barrier height
delta_v = 0.00*eV # fluctuation in barrier height

s_list = s_mean - delta_s + 2*delta_s*random.rand(N_barrier) # list of interbarrier distances
s_list = s_list + (s_mean-s_list.mean()) # ensure that the mean distance is exactly s_mean
# s_list[0] is the distance between 0th and 1st barrier
v_list = v_mean - delta_v + 2*delta_v*random.rand(N_barrier) # list of barrier heights
v_list = v_list + (v_mean-v_list.mean()) # ensure that the mean height is exactly v_mean
# v_list[0] is the height of the 0th barrier
E_list = np.linspace(E_min,E_max,1001,dtype='float') + 1E-10*eV # energy spectrum
M_list = np.zeros([N_barrier,2,2],dtype='complex')
t_list = np.zeros_like(E_list) # transmission spectrum

for idx, E in enumerate(E_list):    
    Ptotal = np.eye(2)
    InvP = np.eye(2)
    for n in reversed(range(N_barrier)): # note the reversed order 
        V = v_list[n] # barrier height
        s = s_list[n] # distance to next barrier on right, doesn't matter for rightmost barrier
        if E>V:
            k_prime = np.sqrt(2*m_e*(E-V))/hbar
        else:
            k_prime = -1j*np.sqrt(2*m_e*(V-E))/hbar
        k = np.sqrt(2*m_e*(E-0))/hbar
        eta = k/k_prime
        eps_plus = 0.5*(eta+1/eta)
        eps_minus = 0.5*(eta-1/eta)
        M11 = (np.cos(2*k_prime*a_barrier) - \
               1j*eps_plus*np.sin(2*k_prime*a_barrier))*np.exp(2*1j*k*a_barrier)
        M12 = 1j*eps_minus*np.sin(2*k_prime*a_barrier)
        M22 = M11.conjugate()
        M21 = M12.conjugate()
        M = np.array([[M11, M12],[M21, M22]])        
        P = M@np.array([[np.exp(-1j*k*s), 0],[0, np.exp(1j*k*s)]])
        Ptotal = P@Ptotal        
        # M_list[n,:,:] = M 
        
    t = 1.0/np.abs(Ptotal[0,0]) # transmission coefficient
    t_list[idx] = t
    # print(E/eV,np.abs(np.linalg.det(P)))
    
t_list[np.isnan(t_list)] = 1e-200 # replace nan data at low energies
    

# Plot transmission data
plt.plot(E_list/eV, t_list, linewidth=1, marker='none',color=[0.0, 0.0, 0.5])
plt.xlabel('Energy (eV)')
plt.ylabel('Transmission')
plt.show()


