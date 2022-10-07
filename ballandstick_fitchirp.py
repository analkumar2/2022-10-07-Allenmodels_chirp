import matplotlib.pyplot as plt
import numpy as np
import moose
import rdesigneur as rd
import brute_curvefit
from sklearn import preprocessing
# import sys
# sys.path.insert(1,'../../Compilations/Kinetics')
# import Na_T_Chan_Hay2011
# Na_T_Chan_Hay2011.l_vhalf_inf = -0.066+0.1


numDendSegments = 10
numBranches = 1
comptLen = 50e-6
comptDia = 4e-6
RM_soma = 0.11696
CM_soma = 1e-2
RA_soma = 1.18
RM_dend = 1.86794368
CM_dend = 0.02411617
RA_dend = 1.18
Em = -82.34514617919922e-3
initVm = Em
stimamp = 50e-12
stim_start = 0.050
elecDt = 1e-4
elecPlotDt = 1e-4
tstop = 7

tvec_fM, Ivec_fM, Vmvec_fM = np.load('fullmorphochirp.npy')

def get_Vmsinglecompt():
    if moose.exists('model'):
        moose.delete('model')
    if moose.exists("Graphs"):
        moose.delete("Graphs")
    if moose.exists("library"):
        moose.delete("library")

    rdes = rd.rdesigneur(
        elecDt = elecDt,
        elecPlotDt = elecPlotDt,
        passiveDistrib = [['soma', 'Cm', f'{138e-12}', 'Rm', f'{168e6}', 'Em', f'{Em}', 'initVm', f'{Em}'],],
        stimList = [['soma', '1', '.', 'inject', f'(t>{stim_start}) * sin(3.14159265359*(t-{stim_start})^4) * {stimamp}']],
        plotList = [['soma', '1', '.', 'Vm', 'Soma membrane potential'],
            ['soma', '1', '.', 'inject', 'Stimulus current']],
    )

    # Setup clock table to record time
    clk = moose.element("/clock")
    moose.Neutral("Graphs")
    plott = moose.Table("/Graphs/plott")
    moose.connect(plott, "requestOut", clk, "getCurrentTime")

    rdes.buildModel()
    moose.reinit()
    moose.start( tstop )
    Vmvec = moose.element("/model/graphs/plot0").vector
    tvec = moose.element("/Graphs/plott").vector
    Ivec = moose.element("/model/graphs/plot1").vector

    return [tvec, Ivec, Vmvec]

def makeBranchingneuronProto(numBranches=1, RM_dend=1.86794368, CM_dend=0.02411617, RA_dend=1.18):
    BNeuron = moose.Neuron( '/library/BNeuron' )
    soma = rd.buildCompt( BNeuron, 'soma', RM = RM_soma, RA = RA_soma, CM = CM_soma, dia = 10.075e-6, x = 0.0, y = 0.0, z = 0.0, dx = 10.075e-6, dy = 0.0, dz = 0.0, Em = Em, initVm = initVm)

    for dendbranch in range(numBranches):
        prev = soma
        dx = np.cos(np.pi*dendbranch/2/numBranches - np.pi/4) * comptLen
        dy = np.sin(np.pi*dendbranch/2/numBranches - np.pi/4) * comptLen
        for seg in range(numDendSegments):
            x = np.cos(np.pi*dendbranch/2/numBranches - np.pi/4) * (seg*comptLen + 15e-6)
            y = np.sin(np.pi*dendbranch/2/numBranches - np.pi/4) * (seg*comptLen + 15e-6)
            
            compt = rd.buildCompt( BNeuron, f'dend_{dendbranch}_{seg}', RM = RM_dend, RA = RA_dend, CM = CM_dend, dia = comptDia, x = x, y = y, z = 0.0, dx = dx, dy = dy, dz = 0.0, Em = Em, initVm = initVm)
            moose.connect(prev, 'axial', compt, 'raxial')
            prev = compt
            x = x+dx
            y = y+dy

    return BNeuron

def get_Vm(numBranches=1, RM_dend=1.86794368, CM_dend=0.02411617, RA_dend=1.18):
    if numBranches==0:
        tvec, Ivec, Vmvec = get_Vmsinglecompt()
        return [tvec, Ivec, Vmvec]

    if moose.exists('model'):
        moose.delete('model')
    if moose.exists("Graphs"):
        moose.delete("Graphs")
    if moose.exists("library"):
        moose.delete("library")

    moose.Neutral( '/library' )
    makeBranchingneuronProto(numBranches,RM_dend, CM_dend, RA_dend)
    gbarscaling = 1
    rdes = rd.rdesigneur(
        elecDt = elecDt,
        elecPlotDt = elecPlotDt,
        cellProto = [['elec','BNeuron']],
        # chanProto = [
        #         # ['make_HH_Na()', 'Na'], 
        #         # ['make_HH_K()', 'K'],
        #         ["../../Compilations/Kinetics/Na_T_Chan_Hay2011.Na_T_Chan()", "Na_T"],
        #         ["../../Compilations/Kinetics/K_31_Chan_Hay2011.K_31_Chan()", "K_31"],
        #         ["../../Compilations/Kinetics/K_P_Chan_Hay2011.K_P_Chan()", "K_P"],
        #         # ["../../Compilations/Kinetics/Na_Chan_Custom4.Na_Chan()", "Na_Chan"],
        #         # ["../../Compilations/Kinetics/K_DR_Chan_Custom3.K_DR_Chan()", "K_DR_Chan"]
        #         ],
        # chanDistrib = [
        #         # ['Na', 'soma', 'Gbar', '1200' ],
        #         # ['K', 'soma', 'Gbar', '200' ],
        #         ['Na_T', 'soma', 'Gbar', f'{1.7058e4*1.5*gbarscaling}'],
        #         ['K_31', 'soma', 'Gbar', f'{0.04407e4*0.5*gbarscaling}'],
        #         ['K_P', 'soma', 'Gbar', f'{0.23628e4*0.5*gbarscaling}'],
        #         # ['Na_Chan', 'soma', 'Gbar', '2000' ],
        #         # ['K_DR_Chan', 'soma', 'Gbar', '700' ],
        #         ],
        stimList = [['soma', '1', '.', 'inject', f'(t>{stim_start}) * sin(3.14159265359*(t-{stim_start})^4) * {stimamp}']],
        plotList = [['soma', '1', '.', 'Vm', 'Soma membrane potential'],
            ['soma', '1', '.', 'inject', 'Stimulus current']],
        # moogList = [['#', '1', '.', 'Vm', 'Vm (mV)']]
        )

    # Setup clock table to record time
    clk = moose.element("/clock")
    moose.Neutral("Graphs")
    plott = moose.Table("/Graphs/plott")
    moose.connect(plott, "requestOut", clk, "getCurrentTime")

    rdes.buildModel()
    moose.reinit()
    moose.start( tstop )
    Vmvec = moose.element("/model/graphs/plot0").vector
    tvec = moose.element("/Graphs/plott").vector
    Ivec = moose.element("/model/graphs/plot1").vector

    return [tvec, Ivec, Vmvec]


def get_scaledpasproperties(ttt=[1,2,3], RM_dend=RM_dend, CM_dend=CM_dend, RA_dend=RA_dend):
    tvec, Ivec, Vmvec = get_Vm(numBranches=8, RM_dend=RM_dend, CM_dend=CM_dend, RA_dend=RA_dend)

    #### Calculate input resistance and capacitance#############################################
    E_rest = np.median(Vmvec[tvec>0.6])
    def chargingm25_1(t, R,tau):
        return (
            E_rest - R * 25e-12 * (1 - np.exp(-t / tau))
        )
    tempv = Vmvec[(tvec >= stim_start) & (tvec <= stim_start + 0.2)]

    RCfitted_chm25, errorm25 = brute_curvefit.brute_scifit(
        chargingm25_1,
        np.linspace(0, 0.2, len(tempv)),
        tempv,
        restrict=[[5e4, 0], [170e6, 0.1]],
        ntol=1000,
        printerrors=False,
        parallel=False,
    )
    Rin, Cin = RCfitted_chm25[0], RCfitted_chm25[1]/RCfitted_chm25[0]

    print(f'{RM_dend=}',f'{RA_dend=}', f'{Cin=}', f'{Rin=}', f'{RCfitted_chm25=}', f'{errorm25=}',)

    # plt.figure()
    # plt.plot(np.linspace(0, 0.2, len(tempv)), tempv)
    # plt.plot(np.linspace(0, 0.2, len(tempv)), chargingm25_1(np.linspace(0, 0.2, len(tempv)), *RCfitted_chm25))
    # ################################################

    return [Rin*1e-6, Cin*1e12]
    # return [Cin*1e12]

def normalize(x):
    maxx = max(x)
    minx = min(x - maxx)
    return (x - maxx)/minx


# print(get_scaledpasproperties(ttt=[1,2,3], RM_dend=14.92, CM_dend=0.003, RA_dend=9.44))

# tvec0, Ivec0, Vmvec0 =  get_Vm(numBranches=0)
tvec1, Ivec1, Vmvec1 =  get_Vm(numBranches=1, RM_dend=1.87, CM_dend=0.024, RA_dend=1.18)
# tvec2, Ivec2, Vmvec2 =  get_Vm(numBranches=2, RM_dend=3.73, CM_dend=0.012, RA_dend=2.36)
# tvec4, Ivec4, Vmvec4 =  get_Vm(numBranches=4, RM_dend=7.46, CM_dend=0.006, RA_dend=4.72)
# tvec8, Ivec8, Vmvec8 =  get_Vm(numBranches=8, RM_dend=14.92, CM_dend=0.003, RA_dend=9.44)

plt.plot(tvec_fM, Vmvec_fM, label='full morpho')
# plt.plot(tvec0, Vmvec0, label='singlecomt')
plt.plot(tvec1, Vmvec1, label='1 branch')
# plt.plot(tvec2, Vmvec2, label='2 branches')
# plt.plot(tvec4, Vmvec4, label='4 branches', linestyle='--')
# plt.plot(tvec8, Vmvec8, label='8 branches', linestyle='--')

plt.legend()
plt.show()




# act_pas = [168, 138]
# # get_Vm(RA_soma=RA_soma, RM_dend=RM_dend,CM_dend=CM_dend,RA_dend=RA_dend)
# # get_Vm(RA_soma=RA_soma, RM_dend=RM_dend,CM_dend=CM_dend,RA_dend=RA_dend)
# fitted, error = brute_curvefit.brute_scifit(
#         get_scaledpasproperties,
#         [1,2,3],
#         act_pas,
#         restrict=[[0.5, 0.5], [4,4]],
#         ntol=100,
#         returnnfactor=0.02,
#         maxfev=1000,
#         printerrors=True,
#         parallel=True,
#         savetofile=False,
#     )

# print(f'fitted = {fitted}', f'error={error}')

# # fitted = [1e-5, 1.96,0.02,1e-5] ##If the RA is too low 
# # fitted = [0.02]
# # print(get_scaledpasproperties([1,2,3], *fitted))
# print('Rin, Cin = ', get_scaledpasproperties([1,2,3], *fitted))
# tvec, Ivec, Vmvec = get_Vm(numBranches=2, RM_dend=fitted[0], CM_dend=0.012, RA_dend=fitted[1])
# tvec_o, Ivec_o, Vmvec_o = get_Vm(numBranches=0)
# plt.plot(tvec, Vmvec, label='fitted')
# plt.plot(tvec_o, Vmvec_o, label='full morphology single compt equivalent')
# plt.legend()
# plt.show()