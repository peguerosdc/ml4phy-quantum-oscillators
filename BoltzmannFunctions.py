import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

def BoltzmannStep(v,b,w,do_random_sampling=True):
    # Para dar un paso markoviano de un estado v a otro h,
    # primero calculamos la probabilidad de dar este paso a
    # h=1
    batchsize=np.shape(v)[0]
    hidden_dim=np.shape(w)[1]
    z=b+np.dot(v,w)
    P=1/(np.exp(-z)+1)
    if do_random_sampling:
        # Hacemos el truco de siempre para usar esta probabilidad para
        # decidir si pasamos a h=1 (si un número aleatorio cae
        # dentro de [0,p]) o h=0
        p=np.random.uniform(size=[batchsize,hidden_dim])
        return(np.array(p<=P,dtype='int'))
    else:
        # Para el caso NO binario, no vamos a samplear entre 0 y 1,
        # sino que más bien, vamos a regresar la probabilidad tal
        # cual que sera nuestro nuevo estado
        return(P) 
    
def BoltzmannSequence(v,a,b,w):
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
    h = BoltzmannStep(v,b,w, do_random_sampling=True)
    # Ahora de h damos otro paso para llegar a v'
    v_prime=BoltzmannStep(h,a,np.transpose(w),do_random_sampling=False)
    # Dar otro paso para llegar a h'
    h_prime=BoltzmannStep(v_prime,b,w,do_random_sampling=False)
    return (v,h,v_prime,h_prime)

def trainStep(v,a,b,w,do_random_sampling=True):
    """
    Calcula cómo dar el paso con Contrastive-Divergence
      ( contrastive-divergence = dar dos pasos markovianos
      para poder usar la aproximacion <x'> con P0 en vez
      de con P)
    """
    v,h,v_prime,h_prime=BoltzmannSequence(v,a,b,w)            
    # delta a_i  = eta ( <v_i> - <v'_i>)
    da = np.average(v,axis=0)-np.average(v_prime,axis=0)
    # delta b_j  = eta ( <h_j> - <h'_j>)
    db = np.average(h,axis=0)-np.average(h_prime,axis=0)
    # delta w_ij = eta ( <v_i h_j> - <v'_i h'_j> )
    dw = np.average(v[:,:,None]*h[:,None,:],axis=0) - np.average(v_prime[:,:,None]*h_prime[:,None,:],axis=0) 
    return da,db,dw

def produce_sample_images(x_train, batchsize, threshold=0.7):
    """
    Produce los inputs (neuronas visibles) de prueba para
    entrenar. Cada neurona ya NO debe ser binaria; es decir, o 1
    o 0.
    """
    amount_of_samples = np.shape(x_train)[0]
    num_visible = np.shape(x_train)[1]
    # pick random samples
    j=np.random.randint(low=0,high=amount_of_samples, size=batchsize)
    return x_train[j,:]


# Load the MNIST data using tensorflow/keras
def load_data():
    (x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data(path="mnist.npz")
    x_train=x_train/256.
    return x_train, y_train

def displayNImages(images):
    N = np.shape(images)[0]
    fig,ax=plt.subplots(ncols=N,nrows=1,figsize=(N,1))
    for n in range(N):
        ax[n].imshow(images[n,:,:])
        ax[n].axis('off')
    plt.show()

def train(x_train, num_hidden=60, batchsize=10, eta=0.0001, nsteps=10*30000):
    # x_train must already be provided as a plane layer shape=[total_amount_samples, num_visible]
    num_visible = np.shape(x_train)[1]
    # Inicializar los hyperparametros  de la red neuronal (a,b,w)
    # de acuerdo a las recetas dadas por la guia suprema:
    # get average brightness of training images:
    p_avg=np.average(x_train, axis=0)
    a=np.log(p_avg/(1.0+1e-6-p_avg)+1e-6) # recipe for visible biases
    b=np.zeros(num_hidden) # recipe for hidden biases
    w=0.01*np.random.randn(num_visible,num_hidden) # recipe for weights
    # Iterar en los pasos dados
    for j in range(nsteps):
        v=produce_sample_images(x_train, batchsize)
        da,db,dw=trainStep(v,a,b,w)
        a+=eta*da
        b+=eta*db
        w+=eta*dw
        print("{:06d} / {:06d}".format(j,nsteps),end="   \r")
    # Return tuned hyper-parameters
    return (a,b,w)

def produce_samples(starting_point, a,b,w, nsteps=10000):
    """
    Dado una estado inicial (valido) 'starting_v', producir nuevas
    muestras.
    - a,b,w son los hyperparámetros ya entrenados que obtuvimos al correr train()
    Para proucir nuevos estados v, simplemente damos pasos markoviamos
    para llegar a un nuevo v_prime
    """
    # Extraer el numero de visible units de sus respectivos valores
    num_visible = np.shape(a)[0]
    # Extraer el numero de hidden units de sus respectivos valores
    num_hidden = np.shape(b)[0]
    # Extraer el tamaño de batchsize del arreglo starting_point
    batchsize = np.shape(starting_point)[0]

    v=np.copy(starting_point)
    v_prime=np.zeros([batchsize,num_visible])
    h=np.zeros([batchsize,num_hidden])
    h_prime=np.zeros([batchsize,num_hidden])

    for k in range(nsteps):
        # step from v via h to v_prime!
        v,h,v_prime,h_prime = BoltzmannSequence(v,a,b,w) 
    # Return the new batch
    return v_prime


# Main
if __name__ == '__main___mnist':
    # First, load the data (x_train=images, y_train = labels)
    data, labels = load_data()
    amount_of_data = np.shape(data)[0]
    Npixels = np.shape(data)[1]
    # visualize some images for fun
    # displayNImages(x_train)
    
    # Prepare the x_train as a dense single visible layer
    num_visible = np.shape(data)[1]*np.shape(data)[2]
    x_train = np.reshape(data,[amount_of_data, num_visible])
    # Train the neural network
    a,b,w = train(x_train)

    # View a result
    produced_batchsize = 10
    initial = produce_sample_images(x_train, batchsize=produced_batchsize)
    produced = produce_samples(initial, a,b,w)
    # Check
    back_to_images = np.reshape(produced, [produced_batchsize, Npixels, Npixels])
    result = np.transpose(np.reshape(np.transpose(back_to_images, axes=[0,2,1]),[produced_batchsize*Npixels,Npixels]))
    plt.imshow(result, interpolation='none')
    plt.show()

import qho
if __name__ == '__main___':
    # Numeros cuanticos para 'S' sistemas
    # de N particulas cada uno
    N = 28*28
    S = 60000
    data = qho.training_set(N, 60000)
    data = qho.normalized_quantum_numbers(data)

    # train neural network
    a,b,w = train(data, batchsize=10, eta=0.0001)

    # get a new distribution
    initial = qho.training_set(N, batchsize=1)
    produced = produce_samples(initial, a,b,w)

    # Check
    expected_n = np.average(produced)
    print(expected_n)