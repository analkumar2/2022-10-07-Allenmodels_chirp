# from neuron import h
import numpy as np
import matplotlib.pyplot as plt
import brute_curvefit
from scipy.signal import hilbert, chirp
from scipy.fft import fft, fftfreq, fftshift
# from neuron.units import s, V, m
import moose
import rdesigneur as rd
import time
import sys

stim_start = 0.2
stimamp = 170e-12
elecDt = 5e-5
elecPlotDt = 5e-5
Em = -82.34514617919922e-3
tstop = 1
gbarscaling = 100
print(gbarscaling)
rdes = rd.rdesigneur(
	elecDt = elecDt,
	elecPlotDt = elecPlotDt,
    cellProto = [['reconstruction.swc', 'elec']],
    passiveDistrib = [['soma#', 'CM', '1e-2', 'RA', '118.7e-2', 'RM', '0.1169', 'Em', f'{Em}', 'initVm', f'{Em}'],
    	['axon#', 'CM', '1.000e-2', 'RA', '118.7e-2', 'RM', '0.105', 'Em', f'{Em}', 'initVm', f'{Em}'],
    	['dend#', 'CM', '3.315e-2', 'RA', '118.7e-2', 'RM', '14.68', 'Em', f'{Em}', 'initVm', f'{Em}'],
    	['apic#', 'CM', '3.315e-2', 'RA', '118.7e-2', 'RM', '1.381', 'Em', f'{Em}', 'initVm', f'{Em}']],
    chanProto = [
            # ['make_HH_Na()', 'Na'], 
            # ['make_HH_K()', 'K'],
            ["../../Compilations/Kinetics/Na_T_Chan_Hay2011_exact.Na_T_Chan()", "Na_T"],
            ["../../Compilations/Kinetics/K_31_Chan_Hay2011_exact.K_31_Chan()", "K_31"],
            ["../../Compilations/Kinetics/K_P_Chan_Hay2011_exact.K_P_Chan()", "K_P"],
            # ["../../Compilations/Kinetics/Na_Chan_Custom4.Na_Chan()", "Na_Chan"],
            # ["../../Compilations/Kinetics/K_DR_Chan_Custom3.K_DR_Chan()", "K_DR_Chan"]
            ],
    chanDistrib = [
            # ['Na', 'soma', 'Gbar', '1200' ],
            # ['K', 'soma', 'Gbar', '200' ],
            ['Na_T', 'soma', 'Gbar', f'{1.7058e4*gbarscaling}'],
            ['K_31', 'soma', 'Gbar', f'{0.04407e4*gbarscaling}'],
            ['K_P', 'soma', 'Gbar', f'{0.23628e4*gbarscaling}'],
            # ['Na_Chan', 'soma', 'Gbar', '2000' ],
            # ['K_DR_Chan', 'soma', 'Gbar', '700' ],
            ],
    stimList = [['soma', '1', '.', 'inject', f'(t>={stim_start} && t<{stim_start+0.5}) * {stimamp}']],
    # stimList = [['soma', '1', '.', 'inject', f'(t>{stim_start}) * sin(3.14159265359*(t-{stim_start})^4) * {stimamp}']],
    plotList = [['soma', '1', '.', 'Vm', 'Soma membrane potential'],
            ['soma', '1', '.', 'inject', 'Stimulus current']],
)
 # Setup clock table to record time
clk = moose.element("/clock")
moose.Neutral("Graphs")
plott = moose.Table("/Graphs/plott")
moose.connect(plott, "requestOut", clk, "getCurrentTime")

rdes.buildModel()
Time = time.time()
moose.reinit()
moose.start( tstop )
print(f'Time to run actual simulation = {time.time()-Time}')
Vmvec = moose.element("/model/graphs/plot0").vector
tvec = moose.element("/Graphs/plott").vector
Ivec = moose.element("/model/graphs/plot1").vector

# fig1, ax1 = plt.subplots()
# ax1.plot(tvec, Ivec, c='lightgreen', linestyle='--', label='Injected current MOOSE')
# ax2 = ax1.twinx()
# ax2.plot(tvec, Vmvec, c='forestgreen', label='Vm MOOSE')
# np.save('tvec_Ivec_Vmvec_MOOSE', [tvec,Ivec,Vmvec])
rdes.display()

#################################################

# h.load_file("stdrun.hoc")
# h.load_file('import3d.hoc')
# h.celsius = 34

# ###Loading the swc and setting up morphology#####
# cell = h.Import3d_SWC_read()
# cell.input('reconstruction.swc')
# i3d = h.Import3d_GUI(cell, 0)
# i3d.instantiate(None)

# # for sec in h.allsec():
# # 	if sec.name()[:4] == "axon":
# # 		h.delete_section(sec=sec)

# # axon = [h.Section(name="axon[0]"), h.Section(name="axon[1]"), h.Section(name="axon[2]")]

# # for sec in axon:
# #     sec.L = 30
# #     sec.diam = 1
# #     sec.nseg = 1 + 2 * int(sec.L / 40.0)
# # axon[0].connect(h.soma[0], 0.5, 0.0)
# # axon[1].connect(axon[0], 1.0, 0.0)
# #################################################

# #####Setting parameters #########################
# soma = h.soma[0]
# for sec in h.allsec():
# 	if 'soma' in sec.name():
# 		sec.cm = 1.0
# 		sec.Ra = 118.68070828445255
# 		sec.insert("pas")
# 		sec(0.5).pas.g = 0.00085515989842588011
# 		sec(0.5).pas.e = Em*V
# 		sec.insert('NaTs')
# 		sec.insert('K_P')
# 		sec.insert('Kv3_1')
# 		sec.ena = 53.0
# 		sec.ek = -107
# 		sec(0.5).NaTs.gbar = 1.7057586476320825
# 		sec(0.5).K_P.gbar = 0.044066924433216317
# 		sec(0.5).Kv3_1.gbar = 0.23628414673253201
# 	elif 'axon' in sec.name():
# 		sec.cm = 1.0
# 		sec.Ra = 118.68070828445255
# 		sec.insert("pas")
# 		sec(0.5).pas.g = 0.00095157282638217886
# 		sec(0.5).pas.e = Em*V
# 	elif 'dend' in sec.name():
# 		sec.cm = 3.314938540399873
# 		sec.Ra = 118.68070828445255
# 		sec.insert("pas")
# 		sec(0.5).pas.g = 6.8117062501849471e-06
# 		sec(0.5).pas.e = Em*V
# 	elif 'apic' in sec.name():
# 		sec.cm = 3.314938540399873
# 		sec.Ra = 118.68070828445255
# 		sec.insert("pas")
# 		sec(0.5).pas.g = 7.2396479517848947e-05
# 		sec(0.5).pas.e = Em*V

# for sec in h.allsec():
#     sec.nseg = 1 + 2 * int(sec.L / 40.0)

# #################################################

# ###Setting neuron h parameters###################
# h.dt = elecDt*s
# h.v_init = Em*V
# h.tstop = tstop*s
# #################################################

# #########Setting up Iclamp########################################
# iclamp = h.IClamp(soma(0.5))
# iclamp.delay = 0
# iclamp.dur = h.tstop #0.500*s
# # iclamp.amp = -0.025
# #################################################

# #######Setting up chirp stimulus##########################################
# ChirpAmp=stimamp*1e9
# t_temp = np.arange(0,h.tstop, h.dt)
# # chirpstim = ChirpAmp*np.sin(2*np.pi*t_temp**2/2e6)
# chirpstim = ChirpAmp*chirp(t_temp, 0, h.tstop, 10e-3, phi=-90)
# chirpstim = np.append(np.zeros(int(50/h.dt)), chirpstim)[:len(t_temp)]
# injfreq = t_temp/2e3
# #################################################

# #####Plotting chirp stimulus###########################################
# # fig, ax1 = plt.subplots()
# ax1.plot(np.linspace(0, h.tstop*1e-3, len(chirpstim)), chirpstim*1e-12, c='darkorchid', linestyle='--', label='Injected current NEURON')
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('Injected current (s)')
# # plt.title('Injected chirp stimulus (0Hz to 20Hz in 20s)')
# ################################################

# #####Recording vectors and runnin the simulation############################################
# chirpstim = h.Vector(chirpstim)
# chirpstim.play(iclamp._ref_amp, h.dt)

# v = h.Vector().record(soma(0.5)._ref_v)  # Membrane potential vector
# t = h.Vector().record(h._ref_t)  # Time stamp vector


# # h.run()
# h.finitialize(h.v_init)
# h.continuerun(h.tstop)
# vv = np.array(v)
# tt = np.array(t)
# #################################################

# #####Plotting Vm############################################
# # plt.figure()
# # fig, ax2 = plt.subplots()
# # ax2 = ax1.twinx()
# # analytic_signal = hilbert(vv*1e-3 - np.mean(vv*1e-3))
# ax2.plot(tt*1e-3,vv*1e-3, c='mediumblue', label='Vm NEURON')
# # plt.plot(tt*1e-3, np.abs(analytic_signal) + np.mean(vv*1e-3))
# ax2.set_xlabel('Time (s)')
# ax2.set_ylabel('Membrane Potential (V)')
# #################################################

# ###### Plotting psd###########################################
# # plt.figure()
# # sp = np.fft.fft(vv*1e-3)
# # freq = np.fft.fftfreq(tt.shape[-1], d=5e-5)
# # plt.plot(freq, sp.real, freq, sp.imag)
# # # plt.psd(x=vv*1e-3, Fs=int(1/5e-5), NFFT=2**17)
# #################################################

# ####### Plotting Impedance vs frequency##########################################
# fig_fft, ax_fft = plt.subplots()
# vvv = Vmvec - np.mean(Vmvec)
# Vmsp = fftshift(fft(vvv))
# Isp = fftshift(fft(Ivec))
# freq = fftshift(fftfreq(tvec.shape[-1], d=elecDt))
# Impedance = np.abs(Vmsp)/np.abs(Isp)
# # ax_fft.plot(freq[freq>=0], Impedance[freq>=0], label='Full morphology')
# ax_fft.plot(freq[freq>=0], np.abs(Isp[freq>=0]))
# ax_fft.set_xlabel('Frequency (Hz)')
# ax_fft.set_ylabel('Impedance amplitude (ohms)')
# #################################################

# # #### With different dt#############################################
# # h.dt = 5e-5*s
# # h.v_init = Em*V
# # h.finitialize(Em*V)

# # ChirpAmp=0.05
# # t_temp = np.arange(0,h.tstop, h.dt)
# # chirpstim = ChirpAmp*np.sin(2*np.pi*t_temp**2/2e6)
# # chirpstim = np.append(np.zeros(int(50/h.dt)), chirpstim)[:len(t_temp)]
# # injfreq = t_temp/2e3

# # chirpstim = h.Vector(chirpstim)
# # chirpstim.play(iclamp._ref_amp, h.dt)

# # v = h.Vector().record(soma(0.5)._ref_v)  # Membrane potential vector
# # t = h.Vector().record(h._ref_t)  # Time stamp vector


# # h.run()
# # vv = np.array(v)
# # tt = np.array(t)

# # plt.plot(tt*1e-3,vv*1e-3, label=f'{h.dt = }')
# # #################################################

# # #########Deleting all sections excet soma for single compartment model#################################################
# # for sec in h.allsec():
# # 	if sec.name()[:4] != "soma":
# # 		h.delete_section(sec=sec)
# # 	else:
# # 		sec.cm = 27
# # 		sec(0.5).pas.g = 0.00085515989842588011*2

# # h.finitialize(Em*V)
# # h.run()
# # vv = np.array(v)
# # vvv = vv - np.mean(vv)
# # tt = np.array(t)
# # Vmsp = fftshift(fft(vvv[:len(chirpstim)]*1e-3))
# # Isp = fftshift(fft(chirpstim*1e-9))
# # freq = fftshift(fftfreq(tt[:len(chirpstim)].shape[-1], d=h.dt*1e-3))
# # Impedance = np.abs(Vmsp)/np.abs(Isp)
# # # plt.plot(injfreq, vv[:len(chirpstim)]*1e-3/chirpstim/1e-9)
# # ax_fft.plot(freq[freq>=0], Impedance[freq>=0], label='Only soma')
# # plt.legend()

# # fig_fft, ax_fftchirp = plt.subplots()
# # ax_fftchirp.plot(freq, np.abs(Isp))
# # ax_fftv = ax_fftchirp.twinx()
# # ax_fftv.plot(freq, np.abs(Vmsp), c='orange', linestyle='--')


# # fig2, ax3 = plt.subplots()
# # ax3.plot(np.linspace(0, h.tstop*1e-3, len(chirpstim)), chirpstim)
# # ax3.set_xlabel('Time (s)')
# # ax3.set_ylabel('Injected current (s)')
# # ax4 = ax3.twinx()
# # # analytic_signal = hilbert(vv*1e-3 - np.mean(vv*1e-3))
# # ax4.plot(tt*1e-3,vv*1e-3, label=f'{h.dt = }', c='orange', linestyle='--')
# # # plt.plot(tt*1e-3, np.abs(analytic_signal) + np.mean(vv*1e-3))
# # ax4.set_xlabel('Time (s)')
# # ax4.set_ylabel('Membrane Potential (V)')
# # #################################################

# # #### Calculate input resistance and capacitance#############################################
# # E_rest = h.v_init*1e-3
# # stim_start = iclamp.delay*1e-3
# # # def chargingm25(t, R1, R2, tau1, tau2):
# # #     return (
# # #         E_rest
# # #         - R1 * 25e-12 * (1 - np.exp(-t / tau1))
# # #         - R2 * 25e-12 * (1 - np.exp(-t / tau2))
# # #     )
# # def chargingm25_1(t, R,tau):
# #     return (
# #         E_rest
# #         - R * 25e-12 * (1 - np.exp(-t / tau))
# #     )
# # tempv = vv[(tt*1e-3 >= stim_start) & (tt*1e-3 <= stim_start + 0.2)]*1e-3
# # # RCfitted_chm25, errorm25 = brute_curvefit.brute_scifit(
# # #     chargingm25,
# # #     np.linspace(0, 0.1, len(tempv)),
# # #     tempv,
# # #     restrict=[[5e6, 5e6, 0, 0], [1000e6, 1000e6, 0.1, 0.1]],
# # #     ntol=1000,
# # #     printerrors=False,
# # # )
# # # Rin = RCfitted_chm25[0] + RCfitted_chm25[1]
# # # if RCfitted_chm25[2] > RCfitted_chm25[3]:
# # #     Cin = RCfitted_chm25[2] / RCfitted_chm25[0]
# # # else:
# # #     Cin = RCfitted_chm25[3] / RCfitted_chm25[1]

# # RCfitted_chm25, errorm25 = brute_curvefit.brute_scifit(
# #     chargingm25_1,
# #     np.linspace(0, 0.2, len(tempv)),
# #     tempv,
# #     restrict=[[5e6, 0], [1000e6, 0.1]],
# #     ntol=1000,
# #     printerrors=False,
# # )
# # Rin, Cin = RCfitted_chm25[0], RCfitted_chm25[1]/RCfitted_chm25[0]

# # print(f'{Cin=}', f'{Rin=}', f'{RCfitted_chm25=}', f'{errorm25=}',)

# # plt.figure()
# # plt.plot(np.linspace(0, 0.2, len(tempv)), tempv)
# # plt.plot(np.linspace(0, 0.2, len(tempv)), chargingm25_1(np.linspace(0, 0.2, len(tempv)), *RCfitted_chm25))
# # #################################################

# ax1.legend()
# ax2.legend()
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('Injected current (A)')
# ax2.set_ylabel('Vm (V)')
# ax_fft.legend()
# plt.show()

# np.save('fullmorphochirp.npy', [tvec, Ivec, Vmvec])