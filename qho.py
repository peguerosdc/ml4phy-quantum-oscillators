from scipy.special import factorial
import scipy.constants as constants
import numpy as np
import math

def produceTrainingSet_idealGas(N, batchsize, m, T, L=1):
    """
    Produce los momentos para un sistema de N particulas
    de masa m, en una caja de L*L*L a temperatura T
    Debe regresar n_particles momentos, batchsize veces.
    O sea, shape=[batchsize, n_particles]
    """
    # Calcular el volumen de la caja
    V = L**3
    # Calcular la longitud de onda termica
    lam = (( constants.h**2 )/(2*constants.pi*m*T*constants.k))**(0.5)
    # Calcular el factorial usando una aprox hecha por
    # scipy.
    # ! ALERT: para numeros muy grandes, esto da inf
    Z = (V**N)/( factorial(N) * lam**(3*N) )
    # Usar el viejo truco para escoger muestras aleatorias de esta
    # distribucion
    j = np.random.randint(low=0,high=amount_of_samples, size=batchsize)
    
def training_set(N, batchsize, w=1000000, T=0.00001, n_max=10):
    """
    Produce los números cuánticos para batchsize sistemas de
    N particulas cada uno.
    Los números serán calculados entre 0 y 10
    """
    def p_n(n):
        """
        Regresa la probabilidad de encontrar a un oscilador
        con un numero cuantico n a frecuencia w y temperatura
        T
        """
        bhw = (constants.hbar*w)/(constants.k*T)
        Z_1 = 1/( 2*np.sinh(bhw/2) )
        return np.exp( -bhw*(n+1/2) )/Z_1
    # Take random quantum numbers according to the distribution p_n
    j = np.random.uniform(size=(batchsize, N))
    p = p_n(np.arange(0,10))
    normalized = np.cumsum(p) / max( np.cumsum(p) ) 
    def take_random_quantum_numbers(j, normalized):
        prev = 0
        cum = np.zeros( j.shape )
        for i,val in enumerate(normalized):
            indexes = np.logical_and(j>prev, j<val) 
            cum[indexes] = i
            prev = val
        return cum
    return take_random_quantum_numbers(j, normalized)

def normalized_quantum_numbers(qn, n_max=10):
    return qn / n_max