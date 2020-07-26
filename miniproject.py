import BoltzmannMachine as bm
import QHO as qho
import numpy as np

# Init the quantum gas at temperature T, frequency w and
# N particles
N = 100
T = 0.00001
w = 1000000
gas = qho.QHOGas(N, T, w)

# hyper-params
hidden_units = 10
training_size = 30000
batchsize = 10
eta = 0.0001
nsteps = 10*30000

# Init the boltzmann machine and train it
m = bm.BoltzmannMachine(hidden_units)
training_set = gas.generate(training_size)
a,b,w = m.train(training_set, batchsize, eta, nsteps)

# Test an output generated after s steps
s = 1000
initial_state = gas.generate(1)
generated = m.produce(initial_state, s)

# Compute the average <n> and check with the theoretical value
print(f" <n> : theoretical={gas.calculate_expected_n()} vs generated={np.average(generated)}")