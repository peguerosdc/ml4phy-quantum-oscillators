import BoltzmannMachine as bm
import QHO as qho
import numpy as np
import datetime
# Visualization imports
from IPython.display import clear_output
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi']=300
def sigmoid(x):
    return .5 * (1 + np.tanh(x / 2.))

# Set the quantum gas with N particles, a limit of 10 for the
# quantum numbers and default temperature and frequency
N = 10*10
gas = qho.QHOGas(N=N)
n_max = 10

training_size = 100000
# the amount of hidden units was set by trial and error
hidden_units = 70
# the recipe suggests to set the batchsize to 10, though it can range
# from 10 to 100
batchsize = 10
# the recipe suggests a learning rate that makes the weight updates about
# 1e-3 times the weights (to within an order of magnitude)
eta = 0.0001
# the amount of steps was set by trial and error
nsteps = 20000000

# define the validation set to be used in training_visualization
validation_set = gas.generate(amount=10)

def training_visualization(machine, current_step, total_steps, eta, a, b, w, da, db, dw):
    # Every now and then (every 50k steps), let us know that the training
    # is still running
    if current_step%50000 == 0:
        print("{:08d} / {:08d}".format(current_step, total_steps), end="   \r")

    # After 'checkpoint_steps', show the suggested plots
    checkpoint_steps = 90000000
    if current_step%checkpoint_steps == 0 or current_step == total_steps-1:
        # Produce a sample starting from the validation set after 100 steps 
        v = np.copy(validation_set)
        for k in range(100):
            v,h,v_prime,h_prime = machine.boltzmann_sequence(v,a,b,w)
            v = v_prime
        # print useful plots for training
        plot_training(validation_set, v_prime, eta, a, b, w, da, db, dw)

def plot_training(v, v_prime, eta, a, b, w, da, db, dw):
    clear_output(wait=True)
    # Show how the weights light up for the state v
    hMean = sigmoid(np.dot(v, w) + b)
    image = Image.fromarray(hMean * 256).show()

    # Create the grid for all the other plots we want
    plt.rcParams.update({'font.size': 2})
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(ncols=7, nrows=3)

    # plot histogram of initial vs generated
    for i in [0,1,2]:
        ax = fig.add_subplot(gs[i,0])
        ax.hist( v[i], bins=np.arange(0, n_max)/n_max )
        ax.set_title("Initial")
        ax.xaxis.set_ticks_position('top')
        ax = fig.add_subplot(gs[i,1])
        ax.hist( v_prime[i], bins=np.arange(0, n_max)/n_max )
        ax.set_title("Generated")
        ax.xaxis.set_ticks_position('top')

    # plot histogram of visible, hidden, weights
    def plotit(axis, values, title):
        axis.hist(values)
        axis.set_title(f"{title}: mm = {np.mean(np.fabs(values))}")
    plotit(fig.add_subplot(gs[0,2]), a, 'a')
    plotit(fig.add_subplot(gs[0,3]), w.flatten(), 'w')
    plotit(fig.add_subplot(gs[0,4]), b, 'b')
    # plot histogram of d_visible, d_hidden, d_weights
    plotit(fig.add_subplot(gs[1,2]), eta*da, 'da')
    plotit(fig.add_subplot(gs[1,3]), eta*dw.flatten(), 'dw')
    plotit(fig.add_subplot(gs[1,4]), eta*db, 'db')
    # show free energies of the average of samples
    x = lambda vv : b + np.dot(vv, w)
    free_training  = -np.dot(v, a) - np.sum( np.log(1 + np.exp(x(v))), axis=1)
    free_valdation = -np.dot(v_prime, a) - np.sum( np.log(1 + np.exp(x(v_prime))), axis=1)
    print(f"\nF_training={np.average(free_training)} vs F_validation={np.average(free_valdation)}\n")
    # Show.
    # CAUTION! This will freeze the execution
    plt.show()

# Init the boltzmann machine and train it while visualizing the suggested plots
training_set = gas.generate(amount=training_size, n_max=n_max)
m = bm.BoltzmannMachine(num_hidden=hidden_units)
a,b,w = m.train(training_set, batchsize=batchsize, eta=eta, nsteps=nsteps, do_while_training=training_visualization)
# Store in a file
run_id = int(datetime.datetime.now().timestamp())
np.savetxt(f"a_{run_id}.csv", a, delimiter=',')
np.savetxt(f"b_{run_id}.csv", b, delimiter=',')
np.savetxt(f"w_{run_id}.csv", w, delimiter=',')
