import numpy as np
import matplotlib.pyplot as plt

tvec_MOOSE, Ivec_MOOSE, Vmvec_MOOSE = np.load('tvec_Ivec_Vmvec_MOOSE.npy')
tvec_NEURON, Ivec_NEURON, Vmvec_NEURON = np.load('tvec_Ivec_Vmvec_NEURON.npy')

fig1,ax1 = plt.subplots()
ax1.plot(tvec_MOOSE, Ivec_MOOSE, label='Injected current MOOSE', c='wheat')
ax1.plot(tvec_NEURON, Ivec_NEURON, linestyle='--', label='Injected current NEURON', c='lightsteelblue')

ax2 = ax1.twinx()
ax2.plot(tvec_MOOSE, Vmvec_MOOSE, label='Vm MOOSE', c='orange')
ax2.plot(tvec_NEURON, Vmvec_NEURON, linestyle='--', label='Vm NEURON', c='blue')

ax1.legend()
ax2.legend()
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Injected current (A)')
ax2.set_ylabel('Vm (V)')
plt.show()