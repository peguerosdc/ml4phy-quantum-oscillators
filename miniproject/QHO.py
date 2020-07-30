import scipy.constants as constants
import numpy as np
import math

class QHOGas():

    """Generates the statistics of a gas of N quantum harmonic oscillators
    at thermal equilibrium at temperature T and frequency w """
    def __init__(self, N, T=0.00001, w=1000000):
        self.N = N
        self.T = T
        self.w = w

    def p_n(self, n):
        """
        Calculates the probabilities of finding a quantum harmonic
        oscillator with quantum numbers n
        """
        bhw = (constants.hbar * self.w)/(constants.k * self.T)
        Z_1 = 1/( 2*np.sinh(bhw/2) )
        return np.exp( -bhw*(n + 1/2) )/Z_1

    def calculate_expected_n(self, ns = None):
        """
        Calculates the expected quantum number.
        Returns the theoretical value if no samples are provided
        """
        if ns is None:
            bhw = (constants.hbar * self.w)/(constants.k * self.T)
            return 1/( math.exp(bhw) - 1 )
        else:
            return np.average(ns)

    def calculate_expected_e(self, ns = None):
        """
        Calculates the expected energy (which equals to the internal energy).
        Returns the theoretical value if no samples are provided
        """
        if ns is None:
            bhw = (constants.hbar * self.w)/(constants.k * self.T)
            return self.N*constants.hbar*self.w*( 1/2 + 1/( math.exp(bhw) - 1 ) )
        else:
            # Compute the total energy of each batch and then average
            total_energy_per_gas = np.sum(constants.hbar*self.w*( ns + 1/2 ), axis=1 )
            return np.average(total_energy_per_gas)

    def generate(self, amount, normalize=True, n_max=10):
        """
        Generates 'amount' states of this gas where the quantum numbers are
        limited to a max of 'n_max'.
        The states can be normalized so that the quantum numbers returned
        range from 0 to 1 where one corresponds to n=n_max.
        """
        quantum_numbers = self.sample_quantum_numbers(amount, n_max)
        return quantum_numbers/n_max if normalize else quantum_numbers

    def sample_quantum_numbers(self, batchsize, n_max):
        """
        Generates the quantum numbers of 'amount' systems of N quantum harmonic
        oscillators each.
        The result is a 2-D tensor of shape (amount, N)
        """
        # Calculate the probability for the first n_max quantum numbers and normalize
        p = self.p_n( np.arange(0, n_max) )
        accumulated = np.cumsum(p)
        accumulated /= max(accumulated)
        # Get a uniform distribution to do the usual trick
        j = np.random.uniform(size=(batchsize, self.N))
        # Sample with this auxiliar distribution
        quantum_numbers = np.zeros( j.shape )
        prev = 0
        for i, val in enumerate(accumulated):
            # Set this quantum number i if the random number j
            # is in the range of probabilities of this quantum number
            indexes = np.logical_and(j>prev, j<val) 
            quantum_numbers[indexes] = i
            prev = val
        return quantum_numbers