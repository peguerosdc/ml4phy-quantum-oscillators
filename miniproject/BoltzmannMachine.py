
import numpy as np

class BoltzmannMachine(object):
    """Encapsulated version of the Restricted Boltzmann Machine implementation provided
    by Florian Marquardt in the 'Online Course: Machine Learning for Physicists 2020' 
    (https://pad.gwdg.de/s/HJtiTE__U#Course-overview)."""

    def __init__(self, num_hidden=None, a=None, b=None, w=None, l2=None):
        """Defines a Restricted Boltzmann Machine with some possible pre-trained weights (w)
        and biases (a,b).
        The number of hidden units is only required if no b is supplied.
        The L2 coefficient typically ranges from 0.01 to 0.00001. A good starting
        point would be 0.0001"""
        self.a = a
        self.b = b
        self.w = w
        self.num_hidden = num_hidden if b is None else b.shape[0]
        self.l2 = l2
    
    def train(self, dataset, batchsize, eta, nsteps, do_while_training=None):
        """Train this boltzmann machine based on the dataset and the hyper-parameters provided.
        'dataset' must be given as a dense layer with shape=[total_amount_samples, num_visible]"""
        num_visible = np.shape(dataset)[1]
        # Init the hyperparams (a,b,w) of the neural network based on the ultimate guide by Hinton
        # or take the pre-existing training values if provided.
        if self.a is None:
            p_avg = np.average(dataset, axis=0)
            # recipe for visible biases
            a = np.log(p_avg/(1.0+1e-6 - p_avg) + 1e-6)
            # recipe for hidden biases
            b = np.zeros(self.num_hidden)
            # recipe for weights
            w = 0.01*np.random.randn(num_visible, self.num_hidden)
        else:
            a = np.copy(self.a)
            b = np.copy(self.b)
            w = np.copy(self.w)
        # Train in the given amount of steps 'nsteps'
        for j in range(nsteps):
            try:
                v = self.get_minibatch(dataset, batchsize)
                da,db,dw = self.train_step(v, a, b, w)
                a += eta*da
                b += eta*db
                # consider l2 penalty
                w += eta*dw - ( self.l2*w if self.l2 else 0 )
                
                if do_while_training:
                    do_while_training(self, j, nsteps, eta, a, b, w, da, db, dw)

            except KeyboardInterrupt as e:
                print(f"Stopped at step {j} of {nsteps}. Saving...")
                break

        # Return tuned hyper-parameters
        self.a = a
        self.b = b
        self.w = w
        return (a,b,w)

    def generate(self, initial, steps, a=None, b=None, w=None):
        """
        Given an initial state 'initial', produce new samples after 'steps' steps.
        """
        a = self.a if a is None else a
        b = self.b if b is None else b
        w = self.w if w is None else w
        # Get some variables from the trained parameters and the initial state
        num_visible = np.shape(a)[0]
        num_hidden = np.shape(b)[0]
        batchsize = np.shape(initial)[0]
        # Pre-allocate the variables we are going to use for performance
        v = np.copy(initial)
        v_prime = np.zeros([batchsize, num_visible])
        h = np.zeros([batchsize, num_hidden])
        h_prime = np.zeros([batchsize, num_hidden])

        # step from v via h to v_prime after 'steps' steps
        for k in range(steps):
            v,h,v_prime,h_prime = self.boltzmann_sequence(v, a, b, w)
            # Shouldn't we set v = v_prime?
            v = np.copy(v_prime)

        # Return the new batch
        return v_prime

    def boltzmann_step(self, v, b, w, do_random_sampling=True):
        """Gives a markovian state from one layer v to another h.
        do_random_sampling tells if the step should result in a binary
        or gaussian layer."""
        # First, we calculate the probability of giving this step to h=1
        batchsize = np.shape(v)[0]
        hidden_dim = np.shape(w)[1]
        z = b + np.dot(v, w)
        P = 1/(np.exp(-z) + 1)
        if do_random_sampling:
            # We do the trick we learnt in the lectures to decide (with the
            # help of a uniform distribution) if we are moving to h=1 (if
            # the random number falls into [0,P]) or to h=0
            p = np.random.uniform(size=[batchsize, hidden_dim])
            return(np.array(p<=P, dtype='int'))
        else:
            # For the non-binary case we are not going to sample between
            # 0 or 1, but instead we are returning the actual probability
            # as the new state.
            return(P) 
        
    def boltzmann_sequence(self, v, a, b, w):
        """
        Gives 2 markovian steps to approximate:
            < vi  hj >_P
        as
            < vi' hj'>_P0
        We are supposing P is already pretty close to the steady state
        (P ~ P0) so after giving a finite (and small) amount of steps
        (here n=2) we are approximating P0 to P. The more steps we give,
        the better the approximation we get.
        The returned vi' and hj' are the new state after giving the n steps.
        Note: the v returned is exactly the same as the one provided.
        """
        # We start from v and give a step to get to h
        # NOTE: only in this step we are sampling
        h = self.boltzmann_step(v, b, w, do_random_sampling=True)
        # Then, from h we give another step to get to v'
        v_prime = self.boltzmann_step(h, a, np.transpose(w), do_random_sampling=False)
        # DAnd we give the last step to get to h'
        h_prime = self.boltzmann_step(v_prime, b, w, do_random_sampling=False)
        return (v, h, v_prime, h_prime)

    def train_step(self, v, a, b, w):
        """
        Calculates the next training step using Contrastive-Divergence
        (which means we are giving markovian steps to be able to approximate
        <x'> with the known probability distribution P0 rather than )
        """
        v, h, v_prime, h_prime = self.boltzmann_sequence(v, a, b, w)            
        # delta a_i  = eta ( <v_i> - <v'_i>)
        da = np.average(v, axis=0) - np.average(v_prime, axis=0)
        # delta b_j  = eta ( <h_j> - <h'_j>)
        db = np.average(h, axis=0) - np.average(h_prime, axis=0)
        # delta w_ij = eta ( <v_i h_j> - <v'_i h'_j> )
        dw = np.average(v[:,:,None]*h[:,None,:],axis=0) - np.average(v_prime[:,:,None]*h_prime[:,None,:], axis=0) 
        return da, db, dw

    def get_minibatch(self, data, batchsize):
        """
        Extracts a minibatch of size 'batchsize' from the given data.
        In the Gaussian-Bernoulli implementation, these samples are not
        meant to be binary.
        """
        amount_of_samples = np.shape(data)[0]
        num_visible = np.shape(data)[1]
        # pick random samples
        j = np.random.randint(low=0, high=amount_of_samples, size=batchsize)
        return data[j,:]