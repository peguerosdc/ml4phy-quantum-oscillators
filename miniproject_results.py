import BoltzmannMachine as bm
import QHO as qho
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

# load stored training
a = np.loadtxt('a_34.csv', delimiter=',')
b = np.loadtxt('b_34.csv', delimiter=',')
w = np.loadtxt('w_34.csv', delimiter=',')

# load en existing boltzmann machine
gas = qho.QHOGas(N=a.shape[0])
m = bm.BoltzmannMachine(a=a, b=b, w=w)

# create new samples with the machine
trial_amount = 20
trial_steps = 100
initial_state = gas.generate(trial_amount)
# initial_state = np.random.randn(trial_amount, a.shape[0])
generated = m.produce(initial_state, trial_steps)*10
# Multiply by 10 to extract data from gaussian units and roundInt
# to convert to quantum numbers (which are integers)
generated_quantum_numbers = np.rint(generated)

# Show average quantum number
print(f"<n> :\ttheoretical\t= {gas.calculate_expected_n()} \n\
    \tgenerated\t= {gas.calculate_expected_n(generated)} \n\
    generated rounded\t= {gas.calculate_expected_n(generated_quantum_numbers)}\n"
)

# Show average energy
print(f"<E> :\ttheoretical\t= {gas.calculate_expected_e()} \n\
    \tgenerated\t= {gas.calculate_expected_e(generated)} \n\
    generated rounded\t= {gas.calculate_expected_e(generated_quantum_numbers)}\n"
)


# Plot the theoretical probability distribution and the generated histogram
n = np.arange(0,10)
plt.hist( generated_quantum_numbers.flatten(), bins=np.arange(0,10), density=True, label="Sampled" )
plt.plot( n, gas.p_n(n), label="Theor." )
plt.title(f"P(n) for {trial_amount} samples generated after {trial_steps} steps")
plt.xlabel('n')
plt.ylabel('P(n)')
plt.legend()

# Plot the weights and biases for reference
def plotIt(axis, values, label):
    axis.hist(values)
    axis.set_title(f"{label}: mm = {np.mean(np.fabs(values))}", fontsize=6)
    axis.tick_params(axis='both', labelsize=4)
weights = plt.figure(1)
fig, ax = plt.subplots(1, 3)
fig.suptitle('Weights and biases')
plotIt(ax[0], a, 'a')
plotIt(ax[1], w.flatten(), 'w')
plotIt(ax[2], b, 'b')

plt.show()
