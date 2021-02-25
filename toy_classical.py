#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 17:59:21 2021

@author: jjh
"""


import numpy as np
import matplotlib.pyplot as plt
import allantools

'''Plots:
    White noise
    Brown noise
    WN & BN
    
    Drift
    WN & drift
    BN & drift
    
    Sinusoidal - varying periods
    WN & sino
    BN & sino
    
    Sinusoidal - varying amplitude
    WN & sino
    BN & sino
    
    WN & drift & sino
    WN, BN, drift, sino
'''

#%% Constants
freq = 100              # sensor rate [Hz]
t0 = 1/freq             # data start [s]
t1 = 100                # data stop [s]
L = t1*freq             # data size [-]
t = np.linspace(t0, t1, L)          # signal time [s]

# dtype = 'phase'
dtype = 'freq'
fsize = (10, 8)

taus = np.logspace(
    np.log(1/freq), np.log(L/2/freq), 500) # ADEV evaluation domain



#%% White noise
data_wn = allantools.noise.white(L, fs=1/freq)
(tau_p, adev_p, errors, ns) = allantools.oadev(
    data_wn, rate=freq, data_type='phase', taus=taus)
(tau_f, adev_f, errors, ns) = allantools.oadev(
    data_wn, rate=freq, data_type='freq', taus=taus)

plt.figure(figsize=fsize)
plt.suptitle('White noise', fontsize=20)

plt.subplot(2, 1, 1)
plt.plot(t, data_wn)
plt.xlabel('Time [s]')
plt.ylabel('Accel. [L/s^2]')

plt.subplot(2, 1, 2)
plt.loglog(tau_p, adev_p, color='C0', linestyle='dashed')
plt.loglog(tau_f, adev_f, color='C0', linestyle='solid')
plt.xlabel('Cluster time [s]')
plt.ylabel('OADEV [L/s^2]')
plt.legend(('phase', 'freq'))

plt.savefig('white_noise.pdf')




#%% Brownian "brown" noise
data_bn = 1e-4*allantools.noise.brown(L, fs=1/freq)
(tau_bn, adev_bn, errors, ns) = allantools.oadev(
    data_bn, rate=freq, data_type=dtype, taus=taus)

(tau_wn, adev_wn, errors, ns) = allantools.oadev(
    data_wn, rate=freq, data_type=dtype, taus=taus)

plt.figure(figsize=fsize)
plt.suptitle('Brown noise', fontsize=20)

plt.subplot(2, 1, 1)
plt.plot(t, data_wn, alpha=0.5)
plt.plot(t, data_bn, )
plt.xlabel('Time [s]')
plt.ylabel('Accel. [L/s^2]')

plt.subplot(2, 1, 2)
plt.loglog(tau_wn, adev_wn, tau_bn, adev_bn)
plt.xlabel('Cluster time [s]')
plt.ylabel('OADEV [L/s^2]')
plt.legend(('WN', 'BN'), loc='lower center')

plt.savefig('brown_noise.pdf')




#%% WN & BN
data_wn_bn = data_wn + data_bn
(tau, adev, errors, ns) = allantools.oadev(
    data_wn_bn, rate=freq, data_type=dtype, taus=taus)

plt.figure(figsize=fsize)
plt.suptitle('White & Brown noise', fontsize=20)

plt.subplot(2, 1, 1)
plt.plot(t, data_wn_bn)
plt.xlabel('Time [s]')
plt.ylabel('Accel. [L/s^2]')

plt.subplot(2, 1, 2)
plt.loglog(tau, adev)
plt.xlabel('Cluster time [s]')
plt.ylabel('OADEV [L/s^2]')

plt.savefig('wn_bn.pdf')




#%% Drift
data_drift = np.zeros((L, 5))
drift = np.linspace(0, 1, L)

for i in range(5):
    data_drift[:,i] = drift*(i+1)

plt.figure(figsize=fsize)
plt.suptitle('Drift', fontsize=20)

plt.subplot(2, 1, 1)
plt.plot(t, data_drift)
plt.xlabel('Time [s]')
plt.ylabel('Accel. [L/s^2]')

plt.subplot(2, 1, 2)
for i in range(5):
    (tau, adev, errors, ns) = allantools.oadev(
        data_drift[:,i], rate=freq, data_type=dtype, taus=taus)
    plt.loglog(tau, adev)
plt.xlabel('Cluster time [s]')
plt.ylabel('OADEV [L/s^2]')

plt.savefig('drift.pdf')




#%% WN & Drift
data_wn_drift = np.zeros((L, 5))
drift = np.linspace(0, 1, L)

for i in range(5):
    data_wn_drift[:,i] = data_wn + drift*(i+1)

plt.figure(figsize=fsize)
plt.suptitle('White noise & drift', fontsize=20)

plt.subplot(2, 1, 1)
plt.plot(t, data_wn_drift)
plt.xlabel('Time [s]')
plt.ylabel('Accel. [L/s^2]')

plt.subplot(2, 1, 2)
for i in range(5):
    (tau, adev, errors, ns) = allantools.oadev(
        data_wn_drift[:,i], rate=freq, data_type=dtype, taus=taus)
    plt.loglog(tau, adev)
plt.xlabel('Cluster time [s]')
plt.ylabel('OADEV [L/s^2]')

plt.savefig('wn_drift.pdf')




#%% BN & Drift
data_bn_drift = np.zeros((L, 5))
drift = 0.5*np.linspace(0, 1, L)

for i in range(5):
    data_bn_drift[:,i] = data_bn + drift*(i+1)

plt.figure(figsize=fsize)
plt.suptitle('Brown noise & drift', fontsize=20)

plt.subplot(2, 1, 1)
plt.plot(t, data_bn_drift)
plt.xlabel('Time [s]')
plt.ylabel('Accel. [L/s^2]')

plt.subplot(2, 1, 2)
for i in range(5):
    (tau, adev, errors, ns) = allantools.oadev(
        data_bn_drift[:,i], rate=freq, data_type=dtype, taus=taus)
    plt.loglog(tau, adev)
plt.xlabel('Cluster time [s]')
plt.ylabel('OADEV [L/s^2]')

plt.savefig('bn_drift.pdf')




#%% Sinusoidal - varying periods
y = np.linspace(0, L, L)

if dtype == 'phase':
    A = 0.25
elif dtype == 'freq':
    A = 0.05

kvec = [2, 10]

k = kvec[0]   # number of periods over L
data_sino0 = A*np.sin(2*np.pi*k*y/L)
(tau0, adev0, errors, ns) = allantools.oadev(
    data_sino0, rate=freq, data_type=dtype, taus=taus)

k = kvec[1]   # number of periods over L
data_sino1 = A*np.sin(2*np.pi*k*y/L)
(tau1, adev1, errors, ns) = allantools.oadev(
    data_sino1, rate=freq, data_type=dtype, taus=taus)

plt.figure(figsize=fsize)
plt.suptitle('Sinusoidal: Period variation', fontsize=20)

plt.subplot(2, 2, 1)
plt.plot(t, data_sino0)
plt.title('k = %s, A = %s' % (kvec[0], A))
plt.xlabel('Time [s]')
plt.ylabel('Accel. [L/s^2]')

plt.subplot(2, 2, 3)
plt.loglog(tau0, adev0)
plt.xlabel('Cluster time [s]')
plt.ylabel('OADEV [L/s^2]')

plt.subplot(2, 2, 2)
plt.title('k = %s, A = %s' % (kvec[1], A))
plt.plot(t, data_sino1)
plt.xlabel('Time [s]')
plt.ylabel('Accel. [L/s^2]')

plt.subplot(2, 2, 4)
plt.loglog(tau1, adev1)
plt.xlabel('Cluster time [s]')
plt.ylabel('OADEV [L/s^2]')

plt.savefig('sino_var_period.pdf')




#%% WN & Sinusoidal - varying periods
y = np.linspace(0, L, L)

if dtype == 'phase':
    A = 0.25
elif dtype == 'freq':
    A = 0.05

kvec = [2, 10]

k = kvec[0]   # number of periods over L
data_wn_sino0 = data_wn + A*np.sin(2*np.pi*k*y/L)
(tau0, adev0, errors, ns) = allantools.oadev(
    data_wn_sino0, rate=freq, data_type=dtype, taus=taus)

k = kvec[1]   # number of periods over L
data_wn_sino1 = data_wn + A*np.sin(2*np.pi*k*y/L)
(tau1, adev1, errors, ns) = allantools.oadev(
    data_wn_sino1, rate=freq, data_type=dtype, taus=taus)

plt.figure(figsize=fsize)
plt.suptitle('White noise & Sinusoidal: Period variation', fontsize=20)

plt.subplot(2, 2, 1)
plt.plot(t, data_wn_sino0)
plt.title('k = %s, A = %s' % (kvec[0], A))
plt.xlabel('Time [s]')
plt.ylabel('Accel. [L/s^2]')

plt.subplot(2, 2, 3)
plt.loglog(tau0, adev0)
plt.xlabel('Cluster time [s]')
plt.ylabel('OADEV [L/s^2]')

plt.subplot(2, 2, 2)
plt.title('k = %s, A = %s' % (kvec[1], A))
plt.plot(t, data_wn_sino1)
plt.xlabel('Time [s]')
plt.ylabel('Accel. [L/s^2]')

plt.subplot(2, 2, 4)
plt.loglog(tau1, adev1)
plt.xlabel('Cluster time [s]')
plt.ylabel('OADEV [L/s^2]')

plt.savefig('sino_var_period_wn.pdf')




#%% BN & Sinusoidal - varying periods
y = np.linspace(0, L, L)

if dtype == 'phase':
    A = 0.25
elif dtype == 'freq':
    A = 0.2

kvec = [2, 10]

k = kvec[0]   # number of periods over L
data_bn_sino0 = data_bn + A*np.sin(2*np.pi*k*y/L)
(tau0, adev0, errors, ns) = allantools.oadev(
    data_bn_sino0, rate=freq, data_type=dtype, taus=taus)

k = kvec[1]   # number of periods over L
data_bn_sino1 = data_bn + A*np.sin(2*np.pi*k*y/L)
(tau1, adev1, errors, ns) = allantools.oadev(
    data_bn_sino1, rate=freq, data_type=dtype, taus=taus)

plt.figure(figsize=fsize)
plt.suptitle('Brown noise & Sinusoidal: Period variation', fontsize=20)

plt.subplot(2, 2, 1)
plt.plot(t, data_bn_sino0)
plt.title('k = %s, A = %s' % (kvec[0], A))
plt.xlabel('Time [s]')
plt.ylabel('Accel. [L/s^2]')

plt.subplot(2, 2, 3)
plt.loglog(tau0, adev0)
plt.xlabel('Cluster time [s]')
plt.ylabel('OADEV [L/s^2]')

plt.subplot(2, 2, 2)
plt.title('k = %s, A = %s' % (kvec[1], A))
plt.plot(t, data_bn_sino1)
plt.xlabel('Time [s]')
plt.ylabel('Accel. [L/s^2]')

plt.subplot(2, 2, 4)
plt.loglog(tau1, adev1)
plt.xlabel('Cluster time [s]')
plt.ylabel('OADEV [L/s^2]')

plt.savefig('sino_var_period_bn.pdf')




#%% Sinusoidal - varying amplitude
y = np.linspace(0, L, L)

if dtype == 'phase':
    Avec = [1, 100]
elif dtype == 'freq':
    Avec = [0.2, 20]

k = 6

A = Avec[0]   # number of periods over L
data_sino0 = A*np.sin(2*np.pi*k*y/L)
(tau0, adev0, errors, ns) = allantools.oadev(
    data_sino0, rate=freq, data_type=dtype, taus=taus)

A = Avec[1]   # number of periods over L
data_sino1 = A*np.sin(2*np.pi*k*y/L)
(tau1, adev1, errors, ns) = allantools.oadev(
    data_sino1, rate=freq, data_type=dtype, taus=taus)

plt.figure(figsize=fsize)
plt.suptitle('Sinusoidal: Amplitude variation', fontsize=20)

plt.subplot(2, 2, 1)
plt.plot(t, data_sino0)
plt.title('A = %s, k = %s' % (Avec[0], k))
plt.xlabel('Time [s]')
plt.ylabel('Accel. [L/s^2]')

plt.subplot(2, 2, 3)
plt.loglog(tau0, adev0)
plt.xlabel('Cluster time [s]')
plt.ylabel('OADEV [L/s^2]')

plt.subplot(2, 2, 2)
plt.title('A = %s, k = %s' % (Avec[1], k))
plt.plot(t, data_sino1)
plt.xlabel('Time [s]')
plt.ylabel('Accel. [L/s^2]')

plt.subplot(2, 2, 4)
plt.loglog(tau1, adev1)
plt.xlabel('Cluster time [s]')
plt.ylabel('OADEV [L/s^2]')

plt.savefig('sino_var_amp.pdf')




#%% WN & Sinusoidal - varying amplitude
y = np.linspace(0, L, L)

if dtype == 'phase':
    Avec = [1, 100]
elif dtype == 'freq':
    Avec = [0.2, 20]

k = 6

A = Avec[0]   # number of periods over L
data_wn_sino0 = data_wn + A*np.sin(2*np.pi*k*y/L)
(tau0, adev0, errors, ns) = allantools.oadev(
    data_wn_sino0, rate=freq, data_type=dtype, taus=taus)

A = Avec[1]   # number of periods over L
data_wn_sino1 = data_wn + A*np.sin(2*np.pi*k*y/L)
(tau1, adev1, errors, ns) = allantools.oadev(
    data_wn_sino1, rate=freq, data_type=dtype, taus=taus)

plt.figure(figsize=fsize)
plt.suptitle('White noise & Sinusoidal: Amplitude variation', fontsize=20)

plt.subplot(2, 2, 1)
plt.plot(t, data_wn_sino0)
plt.title('A = %s, k = %s' % (Avec[0], k))
plt.xlabel('Time [s]')
plt.ylabel('Accel. [L/s^2]')

plt.subplot(2, 2, 3)
plt.loglog(tau0, adev0)
plt.xlabel('Cluster time [s]')
plt.ylabel('OADEV [L/s^2]')

plt.subplot(2, 2, 2)
plt.title('A = %s, k = %s' % (Avec[1], k))
plt.plot(t, data_wn_sino1)
plt.xlabel('Time [s]')
plt.ylabel('Accel. [L/s^2]')

plt.subplot(2, 2, 4)
plt.loglog(tau1, adev1)
plt.xlabel('Cluster time [s]')
plt.ylabel('OADEV [L/s^2]')

plt.savefig('sino_var_amp_wn.pdf')




#%% BN & Sinusoidal - varying amplitude
y = np.linspace(0, L, L)

if dtype == 'phase':
    Avec = [1, 100]
elif dtype == 'freq':
    Avec = [0.2, 20]

k = 6

A = Avec[0]   # number of periods over L
data_bn_sino0 = data_bn + A*np.sin(2*np.pi*k*y/L)
(tau0, adev0, errors, ns) = allantools.oadev(
    data_bn_sino0, rate=freq, data_type=dtype, taus=taus)

A = Avec[1]   # number of periods over L
data_bn_sino1 = data_bn + A*np.sin(2*np.pi*k*y/L)
(tau1, adev1, errors, ns) = allantools.oadev(
    data_bn_sino1, rate=freq, data_type=dtype, taus=taus)

plt.figure(figsize=fsize)
plt.suptitle('Brown noise & Sinusoidal: Amplitude variation', fontsize=20)

plt.subplot(2, 2, 1)
plt.plot(t, data_bn_sino0)
plt.title('A = %s, k = %s' % (Avec[0], k))
plt.xlabel('Time [s]')
plt.ylabel('Accel. [L/s^2]')

plt.subplot(2, 2, 3)
plt.loglog(tau0, adev0)
plt.xlabel('Cluster time [s]')
plt.ylabel('OADEV [L/s^2]')

plt.subplot(2, 2, 2)
plt.title('A = %s, k = %s' % (Avec[1], k))
plt.plot(t, data_bn_sino1)
plt.xlabel('Time [s]')
plt.ylabel('Accel. [L/s^2]')

plt.subplot(2, 2, 4)
plt.loglog(tau1, adev1)
plt.xlabel('Cluster time [s]')
plt.ylabel('OADEV [L/s^2]')

plt.savefig('sino_var_amp_bn.pdf')




#%% WN & drift, sinusoidal
data_wn_drift_sino = np.zeros((L, 5))
drift = np.linspace(0, 1, L)

y = np.linspace(0, L, L)

if dtype == 'phase':
    A = 1
elif dtype == 'freq':
    A = 0.5

k = 6

for i in range(5):
    data_wn_drift_sino[:,i] = data_wn + drift*(i+1) + A*np.sin(2*np.pi*k*y/L)

plt.figure(figsize=fsize)
plt.suptitle('White noise & drift, sinusoidal', fontsize=20)

plt.subplot(2, 1, 1)
plt.plot(t, data_wn_drift_sino)
plt.xlabel('Time [s]')
plt.ylabel('Accel. [L/s^2]')

plt.subplot(2, 1, 2)
for i in range(5):
    (tau, adev, errors, ns) = allantools.oadev(
        data_wn_drift_sino[:,i], rate=freq, data_type=dtype, taus=taus)
    plt.loglog(tau, adev)
plt.xlabel('Cluster time [s]')
plt.ylabel('OADEV [L/s^2]')

plt.savefig('wn_drift_sino.pdf')




#%% WN, BN, drift, sinusoidal
data_wn_bn_drift_sino = np.zeros((L, 5))
drift = 0.4*np.linspace(0, 1, L)

y = np.linspace(0, L, L)

if dtype == 'phase':
    A = 1
elif dtype == 'freq':
    A = 0.25

k = 6

for i in range(5):
    data_wn_bn_drift_sino[:,i] = data_wn + 2*data_bn + drift*(i+1) + A*np.sin(2*np.pi*k*y/L)

plt.figure(figsize=fsize)
plt.suptitle('White noise, Brown noise, drift, sinusoidal', fontsize=20)

plt.subplot(2, 1, 1)
plt.plot(t, data_wn_bn_drift_sino)
plt.xlabel('Time [s]')
plt.ylabel('Accel. [L/s^2]')

plt.subplot(2, 1, 2)
for i in range(5):
    (tau, adev, errors, ns) = allantools.oadev(
        data_wn_bn_drift_sino[:,i], rate=freq, data_type=dtype, taus=taus)
    plt.loglog(tau, adev)
plt.xlabel('Cluster time [s]')
plt.ylabel('OADEV [L/s^2]')

plt.savefig('wn_bn_drift_sino.pdf')