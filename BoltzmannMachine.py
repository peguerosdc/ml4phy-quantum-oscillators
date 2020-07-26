
import tensorflow as tf
import numpy as np

class BoltzmannMachine(object):

    """docstring for BoltzmannMachine"""
    def __init__(self, num_hidden):
        self.num_hidden = num_hidden
        self.a =None
        self.b =None
        self.w =None
    
    def train(self, dataset, batchsize=10, eta=0.0001, nsteps=10*30000):
        # dataset must already be provided as a plane layer shape=[total_amount_samples, num_visible]
        num_visible = np.shape(dataset)[1]
        # Inicializar los hyperparametros  de la red neuronal (a,b,w)
        # de acuerdo a las recetas dadas por la guia suprema:
        # get average brightness of training images:
        p_avg=np.average(dataset, axis=0)
        a=np.log(p_avg/(1.0+1e-6-p_avg)+1e-6) # recipe for visible biases
        b=np.zeros(self.num_hidden) # recipe for hidden biases
        w=0.01*np.random.randn(num_visible, self.num_hidden) # recipe for weights
        # Iterar en los pasos dados
        for j in range(nsteps):
            v = self.__get_minibatch__(dataset, batchsize)
            da,db,dw = self.__train_step__(v, a, b, w)
            a += eta*da
            b += eta*db
            w += eta*dw
            print("{:06d} / {:06d}".format(j,nsteps),end="   \r")
        # Return tuned hyper-parameters
        self.a = a
        self.b = b
        self.w = w
        return (a,b,w)

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

    def __train_step__(self, v, a, b, w, do_random_sampling=True):
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