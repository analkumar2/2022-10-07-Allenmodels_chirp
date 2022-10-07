from neuron import h
import numpy as np
from neuron.units import s, V, m
import time
from tqdm import tqdm
import brute_curvefit
import matplotlib.pyplot as plt
h.load_file("stdrun.hoc")
h.load_file('import3d.hoc')
# h.celsius = 34

h.tstop = 1e3
h.dt = 5e-2

def getVm(t=[1,2,3], cm = 1, g=1/30e3):
    soma = h.Section(name="soma")
    soma.nseg = 1
    soma.L = 100e-6*m
    soma.diam = 100e-6*m
    sm_area = np.pi*soma.L*soma.diam
    soma.cm = cm
    soma.insert("pas")
    soma(0.5).pas.g = g
    soma(0.5).pas.e = -65

    # h.v_init = -65e-3*V

    iclamp = h.IClamp(soma(0.5))
    iclamp.delay = 0.2e3 #0.28125 *s
    iclamp.dur = 0.5e3 #0.500*s
    iclamp.amp = -0.025

    v = h.Vector().record(soma(0.5)._ref_v)  # Membrane potential vector
    t = h.Vector().record(h._ref_t)  # Time stamp vector

    h.finitialize(h.v_init)
    h.continuerun(h.tstop)

    vv = np.array(v)
    tt = np.array(t)

    # plt.figure()
    # plt.plot(tt*1e-3, vv*1e-3)
    # plt.show()

    def chargingm25_1(t, R,tau):
        t = np.array(t)
        return (
            E_rest
            - R * 25e-12 * (1 - np.exp(-t / tau))
        )

    E_rest = soma(0.5).pas.e*1e-3
    stim_start = iclamp.delay*1e-3
    tempv = vv[(tt*1e-3 >= stim_start) & (tt*1e-3 <= stim_start + 0.2)]*1e-3

    RCfitted_chm25, errorm25 = brute_curvefit.brute_scifit(
        chargingm25_1,
        np.linspace(0, 0.2, len(tempv)),
        tempv,
        restrict=[[5e6, 0], [1000e6, 0.1]],
        ntol=1000,
        printerrors=False,
        parallel=False,
    )
    Rin, Cin = RCfitted_chm25[0], RCfitted_chm25[1]/RCfitted_chm25[0]

    # print(f'{Cin=}', f'{Rin=}', f'{RCfitted_chm25=}', f'{errorm25=}',)

    # plt.figure()
    # plt.plot(np.linspace(0, 0.2, len(tempv)), tempv)
    # plt.plot(np.linspace(0, 0.2, len(tempv)), chargingm25_1(np.linspace(0, 0.2, len(tempv)), *RCfitted_chm25))


    return [Rin*1e-6, Cin*1e12]


# fitted, error = brute_curvefit.brute_scifit(
#     getVm,
#     [1,2,3],
#     [168,138],
#     restrict=[[0.01, 1e-6], [10, 1e-3]],
#     ntol=1000,
#     printerrors=True,
#     parallel=True,
# )

# print(fitted)
print(getVm(t=[1,2,3], cm = 4.38794138e-01, g=1.89470170e-05))