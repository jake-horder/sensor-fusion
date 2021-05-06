#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 13:13:21 2021

@author: jjh
"""

import numpy as np
import matplotlib.pyplot as plt
from allantools import oadev



#%%

# Ground Truth
t_end = 600     # Total signal duration [s]
fS = 100        # sample frequency [Hz]
LS = fS*t_end
tS = np.linspace(1/fS, t_end, LS)
tS = np.round(tS*fS, 0)/fS          # remove floating point excess

### uncomment to select a particular ground truth signal
# Impulse + ramp
z_step = 1 - np.exp(-(tS-219))
z_step[z_step<0] = 0
zS = 2*np.exp(-(tS-219)**2*50) + z_step

# # Stationary
# zS = np.zeros(LS)

# # Sinusoidal
# zS = 0.1*np.sin(3*tS*2*np.pi/3600)

# Adding some process noise to the ground truth
np.random.seed(8)
std_rw_C = 0.0003
rwC = np.cumsum(std_rw_C*np.random.randn(LS))
zS = zS + rwC



# Classical 100 Hz
fC = 100
LC = fC*t_end
tC = np.linspace(1/fC, t_end, LC)

np.random.seed(0)
std_wn_C = 0.001
wnC = std_wn_C*np.random.randn(LC)

np.random.seed(1)
std_rw_C = 0.0006
rwC = np.cumsum(std_rw_C*np.random.randn(LC))

dr = 0.2                # drift rate [ms^-2/hr]
drC = (dr/3600)*tC

zC = zS + wnC + 1*rwC + 1*drC



# Classical 1 Hz
fC1 = 1
LC1 = fC1*t_end
tC1 = np.linspace(1/fC1, t_end, LC1)

zC_reshaped = zC.reshape((t_end, fC))
zC1 = np.mean(zC_reshaped[:,-5:], 1)



# Quantum 1 Hz
fQ = 1
LQ = fQ*t_end
tQ = np.linspace(1/fQ, t_end, LQ)

np.random.seed(2)
std_wn_Q = 0.01
wnQ = std_wn_Q*np.random.randn(LS)

zQ100 = zS + wnQ
zQ_reshaped = zQ100.reshape((t_end, fC))
zQ = np.mean(zQ_reshaped[:,-5:], 1)




# Show raw signals
plt.figure(figsize=(9,12))

t1 = 217
t2 = 221
plt.subplot(311)
plt.plot(tC[t1*fC:t2*fC], zC[t1*fC:t2*fC], 'C0')
plt.plot(tC1[t1*fC1-1:t2*fC1], zC1[t1*fC1-1:t2*fC1], 'C4', marker='o', markersize=12, linestyle='none')
plt.plot(tQ[t1*fQ-1:t2*fQ], zQ[t1*fQ-1:t2*fQ], 'C1', marker='*', markersize=12, linestyle='none')
plt.plot(tS[t1*fS:t2*fS], zS[t1*fS:t2*fS], 'C3')
plt.xlabel('Time [s]')
plt.ylabel('Acc. [ms^-2]')
plt.grid()
plt.legend(('Classical (100 Hz)', 'Classical (1 Hz)', 'Quantum (1 Hz)', 'Ground Truth'))

plt.subplot(312)
plt.plot(tC, zC, 'C0')
plt.plot(tC1, zC1, 'C4')
plt.plot(tQ, zQ, 'C1')
plt.plot(tS, zS, 'C3')
plt.xlabel('Time [s]')
plt.ylabel('Acc. [ms^-2]')
plt.grid()
plt.legend(('Classical (100 Hz)', 'Classical (1 Hz)', 'Quantum (1 Hz)', 'Ground Truth'))

t1 = 590
t2 = 600
plt.subplot(313)
plt.plot(tC[t1*fC:t2*fC], zC[t1*fC:t2*fC], 'C0')
plt.plot(tC1[t1*fC1-1:t2*fC1], zC1[t1*fC1-1:t2*fC1], 'C4', marker='o', markersize=12, linestyle='none')
plt.plot(tQ[t1*fQ-1:t2*fQ], zQ[t1*fQ-1:t2*fQ], 'C1', marker='*', markersize=12, linestyle='none')
plt.plot(tS[t1*fS:t2*fS], zS[t1*fS:t2*fS], 'C3')
plt.xlabel('Time [s]')
plt.ylabel('Acc. [ms^-2]')
plt.grid()
plt.legend(('Classical (100 Hz)', 'Classical (1 Hz)', 'Quantum (1 Hz)', 'Ground Truth'))

plt.tight_layout()



#%%
# Hybrid 100 Hz

# Initialise
zH = np.zeros(LC)
t_lin_reg = 5       # duration of "look-back" time to average over [s]
tQ2 = np.hstack((np.linspace(-t_lin_reg+1, 0, t_lin_reg), tQ))

diff_vec = np.zeros(t_lin_reg)
m = 0
b = 0

m_vec = np.zeros(LQ)
b_vec = np.zeros(LQ)

# Hybridise in real-time
for i in range(LS):
    t = tS[i]
    
    # Estimate accumulated drift
    drift_est = m*t + b
    
    # Re-zero raw classical datum
    zH[i] = zC[i] - drift_est
    
    # Recalculate drift coefficients (every 1 second)
    if t%1==0:
        j = int(t)
        d_j = zC1[j-1] - zQ[j-1]
        diff_vec = np.hstack((diff_vec[1:], np.array([d_j])))
        
        m, b = np.polyfit(tQ2[j:j+t_lin_reg], diff_vec, 1)
        m_vec[j-1] = m
        b_vec[j-1] = b



# Show hybrid signal and Allan deviation of residuals
plt.figure(figsize=(9,12))

plt.subplot(311)
plt.title('Time series')
plt.plot(tC, zC, 'C0')
plt.plot(tC1, zC1, 'C4')
plt.plot(tQ, zQ, 'C1')
plt.plot(tC, zH, 'C2')
plt.plot(tS, zS, 'C3')
plt.xlabel('Time [s]')
plt.ylabel('Acc. [ms^-2]')
plt.grid()
plt.legend(('Classical (100 Hz)', 'Classical (1 Hz)', 'Quantum (1 Hz)', 'Hybrid (100 Hz)', 'Ground Truth'))

plt.subplot(312)
plt.title('Residual')
plt.plot(tC, zS-zC, 'C0')
plt.plot(tC, zS-zH, 'C2')
plt.xlabel('Time [s]')
plt.ylabel('Acc. [ms^-2]')
plt.grid()
plt.legend(('Classical', 'Hybrid'))

taus100 = np.logspace(np.log(1/fC), np.log(LC/2/fC), 500)
taus1 = np.logspace(np.log(1/fQ), np.log(LQ/2/fQ), 500)

adeC = oadev(zS-zC, rate=fC, data_type='freq', taus=taus100)
adeQ = oadev(zS[int(fC/fQ)-1::int(fC/fQ)]-zQ, rate=fQ, data_type='freq', taus=taus1)
adeH = oadev(zS-zH, rate=fC, data_type='freq', taus=taus100)

plt.subplot(313)
plt.title('Allan deviation')
plt.loglog(adeC[0], adeC[1], 'C0')
plt.loglog(adeQ[0], adeQ[1], 'C1')
plt.loglog(adeH[0], adeH[1], 'C2')
plt.xlabel('Cluster time [s]')
plt.ylabel('ADEV [ms^-2]')
plt.grid()
plt.legend(('Classical', 'Quantum', 'Hybrid'))

plt.tight_layout()



#%%

plt.figure(figsize=(9,12))

t1 = 215
t2 = 218
plt.subplot(311)
plt.plot(tC[t1*fC:t2*fC], zC[t1*fC:t2*fC], 'C0')
plt.plot(tC1[t1*fC1-1:t2*fC1], zC1[t1*fC1-1:t2*fC1], 'C4', marker='o', markersize=12, linestyle='none')
plt.plot(tQ[t1*fQ-1:t2*fQ], zQ[t1*fQ-1:t2*fQ], 'C1', marker='*', markersize=12, linestyle='none')
plt.plot(tS[t1*fS:t2*fS], zS[t1*fS:t2*fS], 'C3')
plt.plot(tS[t1*fS:t2*fS], zH[t1*fS:t2*fS], 'C2')
plt.xlabel('Time [s]')
plt.ylabel('Acc. [ms^-2]')
plt.grid()
plt.legend(('Classical (100 Hz)', 'Classical (1 Hz)', 'Quantum (1 Hz)', 'Ground Truth', 'Hybrid'))

plt.subplot(312)
plt.plot(tC, zC, 'C0')
plt.plot(tC1, zC1, 'C4')
plt.plot(tQ, zQ, 'C1')
plt.plot(tS, zS, 'C3')
plt.plot(tS, zH, 'C2')
plt.xlabel('Time [s]')
plt.ylabel('Acc. [ms^-2]')
plt.grid()
plt.legend(('Classical (100 Hz)', 'Classical (1 Hz)', 'Quantum (1 Hz)', 'Ground Truth', 'Hybrid'))

t1 = 500
t2 = 600
plt.subplot(313)
plt.plot(tC[t1*fC:t2*fC], zC[t1*fC:t2*fC], 'C0')
plt.plot(tC1[t1*fC1-1:t2*fC1], zC1[t1*fC1-1:t2*fC1], 'C4', marker='o', markersize=12, linestyle='none')
plt.plot(tQ[t1*fQ-1:t2*fQ], zQ[t1*fQ-1:t2*fQ], 'C1', marker='*', markersize=12, linestyle='none')
plt.plot(tS[t1*fS:t2*fS], zS[t1*fS:t2*fS], 'C3')
plt.plot(tS[t1*fS:t2*fS], zH[t1*fS:t2*fS], 'C2')
plt.xlabel('Time [s]')
plt.ylabel('Acc. [ms^-2]')
plt.grid()
plt.legend(('Classical (100 Hz)', 'Classical (1 Hz)', 'Quantum (1 Hz)', 'Ground Truth', 'Hybrid'))

plt.tight_layout()