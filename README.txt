Here, we fit ballandstick chirp response to full morphology chirp response (0Hz to 1000Hz in 100s) in MOOSE

FIrst we check if MOOSE is faster than NEURON (same dt). 
LOL. 
NEURON is faster than MOOSE. By 10x. NEURON took 3.1s. MOOSE took 31s. for 5e-5s dt. for a 10s simulation of full morphology passive

BUT, for single compartment simulations, MOOSE is faster than NEURON by 2x. It took 25s for NEURON for 10 runs of 100s each. MOOSE took only 11s

##########################
In the allensdk model, it is not neccessary to delete the axon segments and then add axon segment again. But do remember to decrease the injected current slightly to compensate if you are not going that way.

###########################
Its decided then. I'll do a parameter search in a balland stick model in MOOSE. Soma passive and geometry is fixed. t^4 for 7s works with dt=1e-4s. Goes till 660Hz

###########################
Check if changing cm changes anything in chirp responses. My intuition says it changes theta