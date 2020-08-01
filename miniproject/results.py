import BoltzmannMachine as bm
import QHO as qho
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi']=300
plt.style.use('seaborn-deep')

def load_data(apath, bpath, wpath):
    a = np.loadtxt(apath, delimiter=',')
    b = np.loadtxt(bpath, delimiter=',')
    w = np.loadtxt(wpath, delimiter=',')
    return (a,b,w)

def init_gas_and_machine(a,b,w):
    gas = qho.QHOGas(N=a.shape[0])
    m = bm.BoltzmannMachine(a=a, b=b, w=w)
    return gas, m

def get_results(machine, initial_state, steps):
    # Multiply by 10 to extract data from gaussian units and roundInt
    # to convert to quantum numbers (which are integers)
    generated = machine.generate(initial_state, steps)*10
    asQuantumNumbers = np.rint(generated)
    return generated, asQuantumNumbers

# load stored training
a,b,w = load_data('./data/a47.csv', './data/b47.csv', './data/w47.csv')

# load en existing boltzmann machine
gas, m = init_gas_and_machine(a,b,w)

# create new samples with the machine
trial_amount = 20
trial_steps = 100
initial_state = gas.generate(trial_amount)
generated, generated_quantum_numbers = get_results(m, initial_state, trial_steps)

# Show average quantum number
print(f"<n> :\ttheoretical\t= {gas.calculate_expected_n()} \n\
    \tgenerated\t= {gas.calculate_expected_n(generated)} \n\
    generated rounded\t= {gas.calculate_expected_n(generated_quantum_numbers)}\n"
)

# Show average energy
print(f"<E> :\ttheoretical\t= {gas.calculate_internal_energy()} \n\
    \tgenerated\t= {gas.calculate_internal_energy(generated)} \n\
    generated rounded\t= {gas.calculate_internal_energy(generated_quantum_numbers)}\n"
)

# Show average energy per oscillator
print(f"<e> :\ttheoretical\t= {gas.calculate_expected_e()} \n\
    \tgenerated\t= {gas.calculate_expected_e(generated)} \n\
    generated rounded\t= {gas.calculate_expected_e(generated_quantum_numbers)}\n"
)

# Plot the theoretical probability distribution and the generated histogram
n = np.arange(0,10)
plt.hist( [generated, generated_quantum_numbers.flatten()], bins=np.arange(0,10), density=True, label=["Sampled", "Rounded"])
plt.plot( n, gas.p_n(n), label="Theoretical", lw=2.5 )
plt.title(r"$\rho_n$" + f" for {trial_amount} samples generated after {trial_steps} steps")
plt.xlabel('n')
plt.ylabel('P(n)')
plt.legend()

# Plot the weights and biases for reference
plt.figure(1)
fig, ax = plt.subplots(1, 3)
def plotHistogram(axis, values, label):
    axis.hist(values)
    axis.set_title(f"{label}: mm = {np.mean(np.fabs(values))}", fontsize=6)
    axis.tick_params(axis='both', labelsize=5.5)
plotHistogram(ax[0], a, 'a')
plotHistogram(ax[1], w.flatten(), 'w')
plotHistogram(ax[2], b, 'b')

plt.show()
