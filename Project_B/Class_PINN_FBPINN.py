# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from Common import *

# Access the gpu (also apple MPS) if available
device = "mps" if getattr(torch,'has_mps',False) \
    else "cuda" if torch.cuda.is_available() else "cpu"

if device == "mps": device = "cpu"  # MPS still has some bugs
print("Running on ", device)

# Set the device
DEVICE = torch.device(device)

# Set the dtype for the tensors
DTYPE = torch.float32

# Set seed for reproducibility
np.random.seed(0)
torch.manual_seed(123)  # This seed is the same as the one used in the paper

################################################################################################
################################################################################################

class FBPINN_Cos_nD(nn.Module):
    """
    This class implements the Finite Basis PINN (FBPINN_Cos_nD) in order to solve the problem that the vanilla PINN is not able to solve.

    This class can be used for any multi-scale problem of the form:

            du
            -- = sum_{i=1}^{n_multi_scale} w_i^2 * cos(w_i*x)         with        u(0) = 0
            dx
    
    The exact solution is:

            u(x) = sum_{i=1}^{n_multi_scale} sin(w_i*x)

    The idea behind the FBPINN is to decompose the domain in n subdomains and to define a different NN for each subdomain.

    This class uses the NN defined in Common.py ->  each subdomain has its own NN which is defined in Common.py as NeuralNet

    This class also has the automatic stopping criterion implemented.
    """

    def __init__(self, domain_extrema, n_subdomains, overlap, sigma, n_hidden_layers, neurons, activation_function, n_multi_scale, w_list):
        super(FBPINN_Cos_nD, self).__init__()

        # The extrema of the domain
        self.domain_extrema = domain_extrema

        # The number of subdomains
        self.n_subdomains = n_subdomains

        # The overlap between two consecutive subdomains
        self.overlap = overlap

        # The parameter defined s.t. the window function is 0 outside the overlap
        self.sigma = sigma

        # The width of each subdomain
        self.width = (self.domain_extrema[1] - self.domain_extrema[0])/self.n_subdomains

        # The number of hidden layers
        assert n_hidden_layers > 0, "Number of hidden layers must be greater than 0"
        self.n_hidden_layers = n_hidden_layers

        # The neurons for each hidden layer
        self.neurons = neurons

        # The activation function
        self.activation_function = activation_function
        
        # The frequencies of the problem
        assert n_multi_scale == len(w_list), "Number of frequecies w do not match the number of multi-scale"
        self.n_multi_scale = n_multi_scale
        self.w_list = w_list

        # Do the domain decomposition
        self.make_subdomains()

        # Create the sub_NNs for each subdomain
        self.make_neural_networks()
    
    ################################################################################################

    def make_subdomains(self):
        """
        This method splits the domain in n_subdomains.
        For each subdomain it is created a list with the midpoints of the overlap
        """

        # Create the subdomains with the overlap & the midpoints of the overlap
        self.midpoints_overlap = []         # List of a&b midpoints of each overlap
        self.subdomains = []                # List of subdomains

        for i in range(self.n_subdomains):

            self.midpoints_overlap.append([self.domain_extrema[0] + i*self.width, self.domain_extrema[0] + (i+1)*self.width])

            if i != 0 and i != self.n_subdomains - 1:
                self.subdomains.append([self.domain_extrema[0] + i*self.width - self.overlap/2, self.domain_extrema[0] + (i+1)*self.width + self.overlap/2])
            elif i == 0:
                self.subdomains.append([self.domain_extrema[0] + i*self.width, self.domain_extrema[0] + (i+1)*self.width + self.overlap/2])
            else:
                self.subdomains.append([self.domain_extrema[0] + i*self.width - self.overlap/2, self.domain_extrema[0] + (i+1)*self.width])
    
    ################################################################################################

    def window_function(self, x, a, b):
        """
        This method computes the window function given x, a, b
        Where:
            x is the input of the NN
            a is the left midpoint of the overlap
            b is the right midpoint of the overlap
        """
        # If x is a numpy array, convert it to a torch tensor
        if type(x) == np.ndarray: 
            x = torch.tensor(x, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
        
        # Compute the window function
        # If a or b are the extrema of the domain, then the window function must not be zero on that side
        if a == self.domain_extrema[0]:
            return torch.sigmoid((b - x)/self.sigma)
        elif b == self.domain_extrema[1]:
            return torch.sigmoid((x - a)/self.sigma)
        else:
            return torch.sigmoid((x - a)/self.sigma) * torch.sigmoid((b - x)/self.sigma)

    ################################################################################################

    def make_neural_networks(self):
        """
        This method creates the neural network for each subdomain
        and moves them to the DEVICE
        """

        # List of the NNs
        self.neural_networks = []

        for i in range(self.n_subdomains):
            NN_i = NeuralNet(input_dimension = 1, output_dimension = 1,
                                                    n_hidden_layers = self.n_hidden_layers,
                                                    neurons = self.neurons,
                                                    regularization_param = 0.,
                                                    regularization_exp = 2.,
                                                    retrain_seed = 0
                                                    )
            NN_i.to(DEVICE)                     # Move the NN to the DEVICE
            self.neural_networks.append(NN_i)
    
    ################################################################################################

    def normalize_input(self, x):
        """
        This method normalizes the input x in the range [-1, 1]
        """
        return 2*(x - torch.min(x))/(torch.max(x) - torch.min(x)) - 1
    
    ################################################################################################

    def unnormalize_output(self, u):
        """
        This method unnormalizes the output of the NN as explained in the paper
        multipling the output of the sub_NN, namely u(x), by n_multi_scale

                unnormalize( u(x) ) = u(x) * n_multi_scale
        """
        return u*self.n_multi_scale
    
    ################################################################################################
    
    def exact_solution(self, x):
        """
        This method computes the exact solution at x for any given number of multi-scales
        The solution is a sum of sines with different frequencies:

                u(x) = sum_{i=1}^{n_multi_scale} sin(w_i*x)
        """
        u_exact = 0

        for i in range(self.n_multi_scale):
            u_exact += torch.sin(self.w_list[i]*x)

        return u_exact
    
    ################################################################################################

    def forward(self, x):
        """
        This method computes the output of the FBPINN for the given x.
        The output is computed following equation (13) in the paper:

                NN(x, theta) = sum_{i=1}^{n_subdomains} window_function(x, a_i, b_i) * unnormalization * NN_i * normalization_i(x)

        Where: * stands for the function composition

        Here the given x is the unnormalized input and this method does the normalization for each subdomain
        """

        # If x is a numpy array, convert it to a torch tensor
        if type(x) == np.ndarray: 
            x = torch.tensor(x, dtype=DTYPE, device=DEVICE).reshape(-1, 1)

        output = torch.zeros_like(x)

        for i in range(self.n_subdomains):
            window_function = self.window_function(x, self.midpoints_overlap[i][0], self.midpoints_overlap[i][1])
            output += window_function * self.unnormalize_output(self.neural_networks[i](self.normalize_input(x)))

        return output

    ################################################################################################

    def loss_function(self, x, verbose=False):
        """
        This method computes the loss function for the FBPINN

        The loss function is computed using the TFC method as explained in the paper.
        To do so, we first build an ansatz that satisfies the boundary conditions and the PDE 
        and with this ansatz we compute the loss function as the mean squared error of the PDE residual.


        The ansatz is built as follows:

                u(x) = tanh(w_n_multi_scale*x) * NN(x)
        """

        # Compute the ansatz
        u = torch.tanh(self.w_list[-1] * x) * self(x)

        # Compute the gradient of the ansatz
        grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        # Compute the RHS of the PDE
        equation = 0
        for i in range(self.n_multi_scale):
            equation += self.w_list[i] * torch.cos(self.w_list[i] * x)

        # Compute the loss as the mean squared error of the PDE residual
        loss = (grad_u - equation).square().mean()

        return loss
    
    ################################################################################################

    def fit(self, num_points, num_epochs=1, patience=5,  min_delta=2 ,verbose=False):
        """
        This method trains the FBPINN using Adam as the optimizer.

        To train the FBPINN, we train each NN separately in its subdomain.
        """
        
        # Start timer for training
        start_time = time.time()

        # Devide the domain in num_points on which to train the NN
        x = torch.linspace(self.domain_extrema[0], self.domain_extrema[1], num_points, dtype=DTYPE, device=DEVICE, requires_grad=True).reshape(-1, 1)

        # Make a list with the parameters of each sub_NN
        parameters = []
        for i in range(self.n_subdomains):
            parameters += self.neural_networks[i].parameters()

        # Define the optimizer for all the parameters
        optimizer = optim.Adam(parameters, lr=float(0.001))

        # List to save the loss
        history = []

        # List to save the L1 loss
        test_L1_loss = []
    
        # Define the print_every parameter
        print_every = 100

        # Define the early stopper
        early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    

        for epoch in range(num_epochs):
            # Start timer for epoch
            start_epoch_time = time.time()

            self.train()

            def closure():
                optimizer.zero_grad()
                loss = self.loss_function(x, verbose=verbose)
                loss.backward()

                history.append(loss.item())

                return loss
            
            optimizer.step(closure)

            # Compute the L1 loss
            self.eval()

            u_pred = torch.tanh(self.w_list[-1]*x) * self(x)
            u_exact = self.exact_solution(x)
            l1_loss = L1_loss(u_pred, u_exact)              # Compute the L1 loss using the function defined in Common.py
            test_L1_loss.append( l1_loss.detach().cpu().numpy() )

            # End timer for epoch
            end_epoch_time = time.time()

            if verbose and epoch % print_every == 0: print("Epoch : ", epoch, "\t Loss: ", history[-1], "\t Epoch_time: ", round(end_epoch_time - start_epoch_time), ' s')

            # Check if early stopping is needed
            if early_stopper.early_stop(history[-1]):             
                break
        
        # End timer for training
        end_time = time.time()

        print("Final loss: ", history[-1], "\t Training_time: ", round(end_time - start_time)//60, ' min ', round(end_time - start_time)%60, ' s')

        return history, test_L1_loss

################################################################################################
################################################################################################

class FBPPINN_Cos_1D(nn.Module):
    """
    This class implements the Finite Basis PINN (FBPINN_Cos_nD) in order to solve the problem that the vanilla PINN is not able to solve.

    This class can be used for single-scale problem of the form:

            du
            -- = cos(w*x)         with        u(0) = 0
            dx
    
    The exact solution is:

            u(x) = 1/w * sin(w*x)

    The idea behind the FBPINN is to decompose the domain in n subdomains and to define a different NN for each subdomain.

    This class uses the NN defined in Common.py ->  each subdomain has its own NN which is defined in Common.py as NeuralNet
    """

    def __init__(self, domain_extrema, n_subdomains, overlap, sigma, n_hidden_layers, neurons, activation_function, w):
        super(FBPPINN_Cos_1D, self).__init__()

        # The extrema of the domain
        self.domain_extrema = domain_extrema

        # The number of subdomains
        self.n_subdomains = n_subdomains

        # The overlap between two consecutive subdomains
        self.overlap = overlap

        # The parameter defined s.t. the window function is 0 outside the overlap
        self.sigma = sigma

        # The width of each subdomain
        self.width = (self.domain_extrema[1] - self.domain_extrema[0])/self.n_subdomains

        # The number of hidden layers
        assert n_hidden_layers > 0, "Number of hidden layers must be greater than 0"
        self.n_hidden_layers = n_hidden_layers

        # The neurons for each hidden layer
        self.neurons = neurons

        # The activation function
        self.activation_function = activation_function

        # The frequency of the problem
        self.w = w

        # Do the domain decomposition
        self.make_subdomains()

        # Create the sub_NNs for each subdomain
        self.make_neural_networks()
    
    ################################################################################################

    def make_subdomains(self):
        """
        This method splits the domain in n_subdomains.
        For each subdomain it is created a list with the midpoints of the overlap
        """

        # Create the subdomains with the overlap & the midpoints of the overlap
        self.midpoints_overlap = []         # List of a&b midpoints of each overlap
        self.subdomains = []                # List of subdomains

        for i in range(self.n_subdomains):

            self.midpoints_overlap.append([self.domain_extrema[0] + i*self.width, self.domain_extrema[0] + (i+1)*self.width])

            if i != 0 and i != self.n_subdomains - 1:
                self.subdomains.append([self.domain_extrema[0] + i*self.width - self.overlap/2, self.domain_extrema[0] + (i+1)*self.width + self.overlap/2])
            elif i == 0:
                self.subdomains.append([self.domain_extrema[0] + i*self.width, self.domain_extrema[0] + (i+1)*self.width + self.overlap/2])
            else:
                self.subdomains.append([self.domain_extrema[0] + i*self.width - self.overlap/2, self.domain_extrema[0] + (i+1)*self.width])
    
    ################################################################################################

    def window_function(self, x, a, b):
        """
        This method computes the window function given x, a, b
        Where:
            x is the input of the NN
            a is the left midpoint of the overlap
            b is the right midpoint of the overlap
        """
        # If x is a numpy array, convert it to a torch tensor
        if type(x) == np.ndarray: 
            x = torch.tensor(x, dtype=DTYPE, device=DEVICE).reshape(-1, 1)
        
        # Compute the window function
        # If a or b are the extrema of the domain, then the window function must not be zero on that side
        if a == self.domain_extrema[0]:
            return torch.sigmoid((b - x)/self.sigma)
        elif b == self.domain_extrema[1]:
            return torch.sigmoid((x - a)/self.sigma)
        else:
            return torch.sigmoid((x - a)/self.sigma) * torch.sigmoid((b - x)/self.sigma)

    ################################################################################################

    def make_neural_networks(self):
        """
        This method creates the neural network for each subdomain
        """

        # List of the NNs
        self.neural_networks = []

        for i in range(self.n_subdomains):
            self.neural_networks.append( NeuralNet(input_dimension = 1, output_dimension = 1,
                                                    n_hidden_layers = self.n_hidden_layers,
                                                    neurons = self.neurons,
                                                    regularization_param = 0.,
                                                    regularization_exp = 2.,
                                                    retrain_seed = 0
                                                    )
                                        )
    
    ################################################################################################

    def normalize_input(self, x):
        """
        This method normalizes the input x in the range [-1, 1]
        """
        return 2*(x - torch.min(x))/(torch.max(x) - torch.min(x)) - 1
    
    ################################################################################################

    def unnormalize_output(self, u):
        """
        This method unnormalizes the output of the NN as explained in the paper
        multipling the output of the sub_NN, namely u(x), by 1/w

                unnormalize( u(x) ) = u(x) * 1/w
        """
        return u/self.w
    
    ################################################################################################
    
    def exact_solution(self, x):
        """
        This method computes the exact solution at x.
        The solution is given by:

                u(x) = 1/w * sin(w*x)
        """
        return torch.sin(self.w*x)/self.w
    
    ################################################################################################

    def forward(self, x):
        """
        This method computes the output of the FBPINN for the given x.
        The output is computed following equation (13) in the paper:

                NN(x, theta) = sum_{i=1}^{n_subdomains} window_function(x, a_i, b_i) * unnormalization * NN_i * normalization_i(x)

        Where: * stands for the function composition

        Here the given x is the unnormalized input and this method does the normalization for each subdomain
        """

        # If x is a numpy array, convert it to a torch tensor
        if type(x) == np.ndarray: 
            x = torch.tensor(x, dtype=DTYPE, device=DEVICE).reshape(-1, 1)

        output = torch.zeros_like(x)

        for i in range(self.n_subdomains):
            window_function = self.window_function(x, self.midpoints_overlap[i][0], self.midpoints_overlap[i][1])
            output += window_function * self.unnormalize_output(self.neural_networks[i](self.normalize_input(x)))

        return output

    ################################################################################################

    def loss_function(self, x, verbose=False):
        """
        This method computes the loss function for the FBPINN

        The loss function is computed using the TFC method as explained in the paper.
        To do so, we first build an ansatz that satisfies the boundary conditions and the PDE 
        and with this ansatz we compute the loss function as the mean squared error of the PDE residual.


        The ansatz is built as follows:

                u(x) = tanh(w_1*x) * NN(x)
        """

        # Compute the ansatz
        u = torch.tanh(self.w * x) * self(x)

        # Compute the gradient of the ansatz
        grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        loss = (grad_u - torch.cos(self.w * x)).square().mean()

        return loss
    
    ################################################################################################

    def fit(self, num_points, num_epochs=1, verbose=False):
        """
        This method trains the FBPINN using Adam as the optimizer.

        To train the FBPINN, we train each NN separately in its subdomain.
        """
        
        # Start timer for training
        start_time = time.time()

        # Devide the domain in num_points on which to train the NN
        x = torch.linspace(self.domain_extrema[0], self.domain_extrema[1], num_points, dtype=DTYPE, device=DEVICE, requires_grad=True).reshape(-1, 1)

        # Make a list with the parameters of each sub_NN
        parameters = []
        for i in range(self.n_subdomains):
            parameters += self.neural_networks[i].parameters()

        # Define the optimizer for all the parameters
        optimizer = optim.Adam(parameters, lr=float(0.001))

        # List to save the loss
        history = []

        # List to save the L1 loss
        test_L1_loss = []
    
        print_every = 100

        for epoch in range(num_epochs):

            # Start timer for epoch
            start_epoch_time = time.time()

            self.train()

            def closure():
                optimizer.zero_grad()
                loss = self.loss_function(x, verbose=verbose)
                loss.backward()

                history.append(loss.item())

                return loss
            
            optimizer.step(closure)

            # End timer for epoch
            end_epoch_time = time.time()

            # Compute the L1 loss
            self.eval()

            u_pred = torch.tanh(self.w*x) * self(x)
            u_exact = self.exact_solution(x)
            l1_loss = L1_loss(u_pred, u_exact)              # Compute the L1 loss using the function defined in Common.py
            test_L1_loss.append( l1_loss.detach().numpy() )

            if verbose and epoch % print_every == 0: print("Epoch : ", epoch, "\t Loss: ", history[-1], "\t Epoch_time: ", round(end_epoch_time - start_epoch_time), ' s')
        
        # End timer for training
        end_time = time.time()

        print("Final loss: ", history[-1], "\t Training_time: ", round(end_time - start_time)//60, ' min ', round(end_time - start_time)%60, ' s')

        return history, test_L1_loss

################################################################################################
################################################################################################

class PINN_Cos_1D(nn.Module):

    """
    This class implements the Physics Informed Neural Network (PINN).

    In this class we define the 1D problem to reproduce the plots of the section 5.2.1 and 5.2.2 of the paper.

    The goal will be to solve the following problem:

            du
            -- = cos(w*x)         with        u(0) = 0
            dx

    The exact solution is:

            u(x) = 1/w * sin(w * x)
    """
    
    def __init__(self, domain_extrema, n_hidden_layers, neurons, activation_function, w):
        super(PINN_Cos_1D, self).__init__()

        # Define the domain extrema
        self.domain_extrema = domain_extrema

        # Define the frequencie
        self.w = w

        # Define the neurons for each hidden layer
        if type(neurons) == list:
            assert len(neurons) == n_hidden_layers, "Number of hidden layers do not match the number of neurons"
            self.neurons = neurons                                          # if neurons_ is a list, then it is the number of neurons per hidden layer
        else:
            self.neurons = [neurons for _ in range(n_hidden_layers)]        # if neurons_ is an integer, then it is the number of neurons per hidden layer

        # Define the number of hidden layers
        assert n_hidden_layers > 0, "Number of hidden layers must be greater than 0"
        self.n_hidden_layers = n_hidden_layers

        # Define the activation function
        self.activation_function = activation_function

        # Define the NN architecture as a fully connected NN
        self.input_layer = nn.Linear(1, self.neurons[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons[i], self.neurons[i+1]) for i in range(self.n_hidden_layers-1)])
        self.output_layer = nn.Linear(self.neurons[-1], 1)

        # Initialize the weights
        self.xavier()

        # Number of total parameters
        self.size = np.sum([np.prod([i for i in p.shape]) for p in self.parameters()]) 
    
    ################################################################################################

    def forward(self, x):
        """
        This method computes the output of the NN for the given x
        """
        x = self.activation_function(self.input_layer(x))

        for layer in self.hidden_layers:
            x = self.activation_function(layer(x))

        x = self.output_layer(x)

        return x
    
    ################################################################################################
    
    def xavier(self):
        """
        This method initializes the weights of the NN using the Xavier initialization
        """
        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                # torch.nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0)
        
        self.apply(init_weights)
    
    ################################################################################################
    
    def normalize_input(self, x):
        """
        This method normalizes the input x to the range [-1, 1]
        """
        return 2*(x - torch.min(x))/(torch.max(x) - torch.min(x)) - 1
    
    ################################################################################################

    def unnormalize_output(self, u):
        """
        This method unnormalizes the output of the NN as explained in the paper:
        multipling the output of the sub_NN, namely u(x), by 1/w

                unnormalize( u(x) ) = u(x) * 1/w
        """
        return u*1/self.w
    
    ################################################################################################
    
    def restore_output(self, u):
        """
        This method restore the output of the NN to what it should be
        if it was given the non normalized input
        """
        max_value = self.domain_extrema[1]
        min_value = self.domain_extrema[0]

        return u * (max_value - min_value)/2 + (max_value+min_value)/2

    ################################################################################################

    def exact_solution(self, x):
        """
        This method computes the exact solution at x for any given number of multi-scales
        The solution is a sum of sines with different frequencies:

                u(x) = 1/w * sin(w * x)
        """
        return 1/self.w * torch.sin(self.w * x)
    
    ################################################################################################

    def loss_function(self, x, verbose=False):
        """ 
        This method computes the loss function for the PINN
        
        The loss in calculated using an ansatz s.t. the problems becomes unconstrained.
        This is done following the Theory of Functional Connections (TFC) approach.
        The ansatz we will use is the following:

            u(x) = tanh(w*x) * NN(x)
        """

        # Normalize the input
        x_norm = self.normalize_input(x)
        x_norm.requires_grad = True

        # Compute the ansatz
        u = torch.tanh(self.w * x_norm) *  self.unnormalize_output( self.forward( x_norm ) )

        # compute the gradient of the ansatz
        grad_u = torch.autograd.grad(u, x_norm, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        loss =  (grad_u - torch.cos(self.w * x)).square().mean()

        return loss
    
    ################################################################################################
    
    def fit(self, num_points, optimizer, num_epochs=1, verbose=False):
        """
        This methods trains the PINN using Adam as the optimizer.
        """

        # Start timer for training
        start_time = time.time()

        # Devide the domain in num_points on which to train the NN
        x = torch.linspace(self.domain_extrema[0], self.domain_extrema[1], num_points, dtype=DTYPE, device=DEVICE, requires_grad=False).reshape(-1, 1)   # the input has to be of shape (n, 1)

        # List to save the loss
        history = []

        # List to save the L1 loss
        test_L1_loss = []
    
        print_every = 100

        for epoch in range(num_epochs):
            # Start timer for epoch
            start_epoch_time = time.time()

            self.train()

            def closure():
                optimizer.zero_grad()
                loss = self.loss_function(x, verbose=verbose)
                loss.backward()

                history.append(loss.item())

                return loss
            
            optimizer.step(closure)

            # End timer for epoch
            end_epoch_time = time.time()

            # Compute the L1 loss
            self.eval()

            u_pred = self.restore_output( torch.tanh(self.w * self.normalize_input(x)) * self.unnormalize_output( self( self.normalize_input(x)) ) )
            u_exact = self.exact_solution(x)
            l1_loss = L1_loss(u_pred, u_exact)              # Compute the L1 loss using the function defined in Common.py
            test_L1_loss.append( l1_loss.detach().numpy() )

            if verbose and epoch % print_every == 0: print("Epoch : ", epoch, "\t Loss: ", history[-1], "\t Epoch_time: ", round(end_epoch_time - start_epoch_time), ' s')
        
        # End timer for training
        end_time = time.time()

        print("Final loss: ", history[-1], "\t Training_time: ", round(end_time - start_time)//60, ' min ', round(end_time - start_time)%60, ' s')

        return history, test_L1_loss

################################################################################################
################################################################################################

class PINN_Cos_nD(nn.Module):
    """
    This class implements the Physics Informed Neural Network (PINN).

    This class can be used for any multi-scale problem of the form:

            du
            -- = sum_{i=1}^{n_multi_scale} w_i^2 * cos(w_i*x)         with        u(0) = 0
            dx
    
    The exact solution is:

            u(x) = sum_{i=1}^{n_multi_scale} sin(w_i*x)
    """
    
    def __init__(self, domain_extrema, n_hidden_layers, neurons, activation_function, n_multi_scale, w_list):
        super(PINN_Cos_nD, self).__init__()

        # Define the domain extrema
        self.domain_extrema = domain_extrema

        # Define the frequencies
        assert n_multi_scale == len(w_list), "Number of frequecies w do not match the number of multi-scale"
        self.n_multi_scale = n_multi_scale
        self.w_list = w_list

        # Define the neurons for each hidden layer
        if type(neurons) == list:
            assert len(neurons) == n_hidden_layers, "Number of hidden layers do not match the number of neurons"
            self.neurons = neurons                                          # if neurons_ is a list, then it is the number of neurons per hidden layer
        else:
            self.neurons = [neurons for _ in range(n_hidden_layers)]        # if neurons_ is an integer, then it is the number of neurons per hidden layer

        # Define the number of hidden layers
        assert n_hidden_layers > 0, "Number of hidden layers must be greater than 0"
        self.n_hidden_layers = n_hidden_layers

        # Define the activation function
        self.activation_function = activation_function

        # Define the NN architecture as a fully connected NN
        self.input_layer = nn.Linear(1, self.neurons[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons[i], self.neurons[i+1]) for i in range(self.n_hidden_layers-1)])
        self.output_layer = nn.Linear(self.neurons[-1], 1)

        # Initialize the weights
        self.xavier()

        # Number of total parameters
        self.size = np.sum([np.prod([i for i in p.shape]) for p in self.parameters()]) 
    
    ################################################################################################

    def forward(self, x):
        """
        This method computes the output of the NN for the given x
        """
        x = self.activation_function(self.input_layer(x))

        for layer in self.hidden_layers:
            x = self.activation_function(layer(x))

        x = self.output_layer(x)

        return x
    
    ################################################################################################
    
    def xavier(self):
        """
        This method initializes the weights of the NN using the Xavier initialization
        """
        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                # torch.nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0)
        
        self.apply(init_weights)
    
    ################################################################################################
    
    def normalize_input(self, x):
        """
        This method normalizes the input x to the range [-1, 1]
        """
        return 2*(x - torch.min(x))/(torch.max(x) - torch.min(x)) - 1
    
    ################################################################################################

    def unnormalize_output(self, u):
        """
        This method unnormalizes the output of the NN as explained in the paper:
        multipling the output of the sub_NN, namely u(x), by 2

                unnormalize( u(x) ) = u(x) * n_multi_scale
        """
        return u*self.n_multi_scale
    
    ################################################################################################
    
    def restore_output(self, u):
        """
        This method restore the output of the NN to what it should be
        if it was given the non normalized input
        """
        max_value = self.domain_extrema[1]
        min_value = self.domain_extrema[0]

        return u * (max_value - min_value)/2 + (max_value+min_value)/2

    ################################################################################################

    def exact_solution(self, x):
        """
        This method computes the exact solution at x for any given number of multi-scales
        The solution is a sum of sines with different frequencies:

                u(x) = sum_{i=1}^{n_multi_scale} sin(w_i*x)
        """
        u_exact = 0

        for i in range(self.n_multi_scale):
            u_exact += torch.sin(self.w_list[i]*x)

        return u_exact
    
    ################################################################################################

    def loss_function(self, x, verbose=False):
        """ 
        This method computes the loss function for the PINN
        
        The loss in calculated using an ansatz s.t. the problems becomes unconstrained.
        This is done following the Theory of Functional Connections (TFC) approach.
        The ansatz we will use is the following:

            u(x) = tanh(w*x) * NN(x)
        """

        # Normalize the input
        x_norm = self.normalize_input(x)
        x_norm.requires_grad = True

        # Compute the ansatz
        u = torch.tanh(self.w_list[-1] * x_norm) *  self.unnormalize_output( self.forward( x_norm ) )

        # compute the gradient of the ansatz
        grad_u = torch.autograd.grad(u, x_norm, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        # Compute the RHS of the PDE
        equation = 0
        for i in range(self.n_multi_scale):
            equation += self.w_list[i] * torch.cos(self.w_list[i] * x)

        # Compute the loss as the mean squared error of the PDE residual
        loss = (grad_u - equation).square().mean()

        return loss
    
    ################################################################################################
    
    def fit(self, num_points, optimizer, num_epochs=1, verbose=False):
        """
        This methods trains the PINN using Adam as the optimizer.
        """

        # Start timer for training
        start_time = time.time()

        # Devide the domain in num_points on which to train the NN
        x = torch.linspace(self.domain_extrema[0], self.domain_extrema[1], num_points, dtype=DTYPE, device=DEVICE, requires_grad=False).reshape(-1, 1)   # the input has to be of shape (n, 1)

        # List to save the loss
        history = []

        # List to save the L1 loss
        test_L1_loss = []
    
        print_every = 100

        for epoch in range(num_epochs):
            # Start timer for epoch
            start_epoch_time = time.time()

            self.train()

            def closure():
                optimizer.zero_grad()
                loss = self.loss_function(x, verbose=verbose)
                loss.backward()

                history.append(loss.item())

                return loss
            
            optimizer.step(closure)

            # End timer for epoch
            end_epoch_time = time.time()

            # Compute the L1 loss
            self.eval()

            u_pred = self.restore_output( torch.tanh(self.w_list[-1] * self.normalize_input(x)) * self.unnormalize_output( self( self.normalize_input(x)) ) )
            u_exact = self.exact_solution(x)
            l1_loss = L1_loss(u_pred, u_exact)              # Compute the L1 loss using the function defined in Common.py
            test_L1_loss.append( l1_loss.detach().numpy() )

            if verbose and epoch % print_every == 0: print("Epoch : ", epoch, "\t Loss: ", history[-1], "\t Epoch_time: ", round(end_epoch_time - start_epoch_time), ' s')
        
        # End timer for training
        end_time = time.time()

        print("Final loss: ", history[-1], "\t Training_time: ", round(end_time - start_time)//60, ' min ', round(end_time - start_time)%60, ' s')

        return history, test_L1_loss

################################################################################################
################################################################################################