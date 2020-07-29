
import tensorflow as tf
import numpy as np
# to visualize
from IPython.display import clear_output
from time import sleep
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

from numpy import tanh, fabs, mean, ones
from PIL import Image
def sigmoid(xx):
    return .5 * (1 + tanh(xx / 2.))

class BoltzmannMachine(object):

    """docstring for BoltzmannMachine"""
    def __init__(self, num_hidden=None, a=None, b=None, w=None):
        self.a = a
        self.b = b
        self.w = w
        self.num_hidden = num_hidden if b is None else b.shape[0]
    
    def train(self, dataset, batchsize=10, eta=0.0001, nsteps=10*30000, display_after_steps=None):
        # dataset must already be provided as a plane layer shape=[total_amount_samples, num_visible]
        num_visible = np.shape(dataset)[1]
        validation_size = 10
        validation = self.__get_minibatch__(dataset, validation_size)
        # Inicializar los hyperparametros  de la red neuronal (a,b,w)
        # de acuerdo a las recetas dadas por la guia suprema:
        # get average brightness of training images:
        # Check if a training was already present:
        if self.a is None:
            p_avg=np.average(dataset, axis=0)
            a=np.log(p_avg/(1.0+1e-6-p_avg)+1e-6) # recipe for visible biases
            b=np.ones(self.num_hidden) # recipe for hidden biases
            w=0.01*np.random.randn(num_visible, self.num_hidden) # recipe for weights
        else:
            a = np.copy(self.a)
            b = np.copy(self.b)
            w = np.copy(self.w)
        # Iterar en los pasos dados
        try:
            for j in range(nsteps):
                v = self.__get_minibatch__(dataset, batchsize)
                da,db,dw = self.__train_step__(v, a, b, w)
                a += eta*da
                b += eta*db
                w += eta*dw

                if j%50000 == 0:
                    print("{:08d} / {:08d}".format(j,nsteps), end="   \r")

                if display_after_steps and (j%display_after_steps == 0 or j == nsteps-1):
                    # producir una imagen de prueba
                    v=np.copy(validation)
                    for k in range(100):
                        v,h,v_prime,h_prime=self.__sequence_of_steps__(v,a,b,w)
                        v = v_prime
                    # print useful plots for training
                    self.print_training(validation, v_prime, a,b,w , eta, da,db,dw)
        except KeyboardInterrupt as e:
            print("Stopped. Saving...")

        # Return tuned hyper-parameters
        self.a = a
        self.b = b
        self.w = w
        return (a,b,w)

    def print_training(self, v, v_prime, a,b,w, eta, da, db, dw):
        clear_output(wait=True)
        # mostrar los weights
        hMean = sigmoid(np.dot(v, w) + b)
        image = Image.fromarray(hMean * 256).show()
        # graficar aqui todos los histogramas que queremos
        plt.rcParams.update({'font.size': 2})
        fig3 = plt.figure(constrained_layout=True)
        gs = fig3.add_gridspec(ncols=7, nrows=3)

        # plot histogram of initial vs generated
        for i in [0,1,2]:
            ax = fig3.add_subplot(gs[i,0])
            ax.hist( v[i], bins=np.arange(0, 10)/10 )
            ax.set_title("Initial")
            ax.xaxis.set_ticks_position('top')
            ax = fig3.add_subplot(gs[i,1])
            ax.hist( v_prime[i], bins=np.arange(0, 10)/10 )
            ax.set_title("Produced")
            ax.xaxis.set_ticks_position('top')
        # plot histogram of visible, hidden, weights
        def plotit(axis, values, title):
            axis.hist(values)
            axis.set_title(f"{title}: mm = {np.mean(np.fabs(values))}")
        plotit(fig3.add_subplot(gs[0,2]), a, 'a')
        plotit(fig3.add_subplot(gs[0,3]), w.flatten(), 'w')
        plotit(fig3.add_subplot(gs[0,4]), b, 'b')
        # plot histogram of d_visible, d_hidden, d_weights
        plotit(fig3.add_subplot(gs[1,2]), eta*da, 'da')
        plotit(fig3.add_subplot(gs[1,3]), eta*dw.flatten(), 'dw')
        plotit(fig3.add_subplot(gs[1,4]), eta*db, 'db')
        # show free energies of the average of samples
        x = lambda vv : b + np.dot(vv, w)
        free_training  = -np.dot(v, a) - np.sum( np.log(1 + np.exp(x(v))), axis=1)
        free_valdation = -np.dot(v_prime, a) - np.sum( np.log(1 + np.exp(x(v_prime))), axis=1)
        print(f"F_training={np.average(free_training)} vs F_validation={np.average(free_valdation)}")
        # show
        plt.ioff()
        plt.draw()
        plt.show()

    def produce(self, initial, steps):
        """
        Dado una estado inicial (valido) 'starting_v', producir nuevas
        muestras.
        - a,b,w son los hyperparámetros ya entrenados que obtuvimos al correr train()
        Para proucir nuevos estados v, simplemente damos pasos markoviamos
        para llegar a un nuevo v_prime
        """
        # Produce an output
        # Extraer el numero de visible units de sus respectivos valores
        num_visible = np.shape(self.a)[0]
        # Extraer el tamaño de batchsize del arreglo initial
        batchsize = np.shape(initial)[0]

        v = np.copy(initial)
        v_prime = np.zeros([batchsize, num_visible])
        h = np.zeros([batchsize, self.num_hidden])
        h_prime = np.zeros([batchsize, self.num_hidden])

        for k in range(steps):
            # step from v via h to v_prime!
            v,h,v_prime,h_prime = self.__sequence_of_steps__(v, self.a, self.b, self.w) 
        # Return the new batch
        return v_prime

    def __step__(self, v, b, w, do_random_sampling=True):
        # Para dar un paso markoviano de un estado v a otro h,
        # primero calculamos la probabilidad de dar este paso a
        # h=1
        batchsize = np.shape(v)[0]
        hidden_dim = np.shape(w)[1]
        z = b + np.dot(v, w)
        P = 1/(np.exp(-z) + 1)
        if do_random_sampling:
            # Hacemos el truco de siempre para usar esta probabilidad para
            # decidir si pasamos a h=1 (si un número aleatorio cae
            # dentro de [0,p]) o h=0
            p = np.random.uniform(size=[batchsize, hidden_dim])
            return(np.array(p<=P, dtype='int'))
        else:
            # Para el caso NO binario, no vamos a samplear entre 0 y 1,
            # sino que más bien, vamos a regresar la probabilidad tal
            # cual que sera nuestro nuevo estado
            return(P) 
        
    def __sequence_of_steps__(self, v, a, b, w):
        """
        Da pasos markovianos para poder aproximar:
        < vi  hj >_P
        como
        < vi' hj'>_P0
        ya que, suponemos P está muy cercano al steady
        state (P ~ P0), por lo que al dar n pasos (aquí n=2)
        estamos aproximando P0 a P más precisamente.
        vi' y hj' son los nuevos estados después de dar los
        n pasos.
        Notar que la 'v' que regresa es exactamente la misma
        que la inicial.
        """
        # Empezamos en v y damos un paso para llegar a h
        # NOTA: sólo en esta sampleamos
        h = self.__step__(v, b, w, do_random_sampling=True)
        # Ahora de h damos otro paso para llegar a v'
        v_prime = self.__step__(h, a, np.transpose(w), do_random_sampling=False)
        # Dar otro paso para llegar a h'
        h_prime = self.__step__(v_prime, b, w, do_random_sampling=False)
        return (v, h, v_prime, h_prime)

    def __train_step__(self, v, a, b, w):
        """
        Calcula cómo dar el paso con Contrastive-Divergence
          ( contrastive-divergence = dar dos pasos markovianos
          para poder usar la aproximacion <x'> con P0 en vez
          de con P)
        """
        v, h, v_prime, h_prime = self.__sequence_of_steps__(v, a, b, w)            
        # delta a_i  = eta ( <v_i> - <v'_i>)
        da = np.average(v, axis=0) - np.average(v_prime, axis=0)
        # delta b_j  = eta ( <h_j> - <h'_j>)
        db = np.average(h, axis=0) - np.average(h_prime, axis=0)
        # delta w_ij = eta ( <v_i h_j> - <v'_i h'_j> )
        dw = np.average(v[:,:,None]*h[:,None,:],axis=0) - np.average(v_prime[:,:,None]*h_prime[:,None,:], axis=0) 
        return da, db, dw

    def __get_minibatch__(self, data, batchsize):
        """
        Produce los inputs (neuronas visibles) de prueba para
        entrenar. Cada neurona ya NO debe ser binaria; es decir, o 1
        o 0.
        """
        amount_of_samples = np.shape(data)[0]
        num_visible = np.shape(data)[1]
        # pick random samples
        j = np.random.randint(low=0, high=amount_of_samples, size=batchsize)
        return data[j,:]