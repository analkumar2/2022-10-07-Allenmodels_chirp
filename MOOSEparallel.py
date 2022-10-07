import moose
import rdesigneur as rd
from tqdm import tqdm


for i in tqdm(range(10)):
    if moose.exists('model'):
        moose.delete('model')
    if moose.exists("Graphs"):
        moose.delete("Graphs")
    if moose.exists("library"):
        moose.delete("library")

    rdes = rd.rdesigneur(
    elecDt = 5e-5,
    stimList = [['soma', '1', '.', 'inject', '(t>0.1 && t<0.2) * 2e-8']],
    plotList = [['soma', '1', '.', 'Vm', 'Soma membrane potential']],
    )
    rdes.buildModel()
    moose.reinit()
    moose.start(100)
    # rdes.display()