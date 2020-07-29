import BoltzmannMachine as bm
import QHO as qho
import numpy as np

# Init the quantum gas at temperature T, frequency w and
# N particles
N = 28*28
T = 0.00001
w = 1000000

# hyper-params
hidden_units = 150
training_size = 30000
batchsize = 10 # este va de 10 a 100, pero 10 es el recomendado para este caso
eta = 0.0001
nsteps = 10*30000

gas = qho.QHOGas(N=28*28, T=0.00001, w=1000000)
# Init the boltzmann machine and train it
m = bm.BoltzmannMachine(num_hidden=500)
training_set = gas.generate(amount=30000)
a,b,w = m.train(training_set, batchsize=10, eta=0.0001, nsteps=10*30000, display_after_steps=10*1500)
