from Class_PINN_FBPINN import *
import matplotlib.pyplot as plt

#################################################################################################

# Common parameters
domain_extrema = [-2*torch.pi, 2*torch.pi]
activation_function = nn.Tanh()

#################################################################################################

#################################################################################################
#                                                                                               #
#                   PINN --> w = 1, n_hidden_layers = 2, neurons = 16                           #
#                                                                                               #
#################################################################################################
w = 1
n_hidden_layers = 2
neurons = 16

num_points = 200
n_epochs = 50000

# Create the model and train it
model = PINN_Cos_1D(domain_extrema, n_hidden_layers, neurons, activation_function, w)
optimizer_ADAM = optim.Adam(model.parameters(),
                            lr=float(0.001))
history, l1_loss = model.fit(num_points, optimizer_ADAM, n_epochs, verbose=False)

# Save the model, the history and the l1_loss
torch.save(model.state_dict(), f'Models/PINN_w_{w}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}.pt')
np.savetxt(f'Models/PINN_w_{w}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_history.txt', np.array(history))
np.savetxt(f'Models/PINN_w_{w}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_l1_loss.txt', np.array(l1_loss))

#################################################################################################

#################################################################################################
#                                                                                               #
#                   PINN --> w = 15, n_hidden_layers = 2, neurons = 16                          #
#                                                                                               #
#################################################################################################
w = 15
n_hidden_layers = 2
neurons = 16

num_points = 200*15
n_epochs = 50000

# Create the model and train it
model = PINN_Cos_1D(domain_extrema, n_hidden_layers, neurons, activation_function, w)
optimizer_ADAM = optim.Adam(model.parameters(),
                            lr=float(0.001))
history, l1_loss = model.fit(num_points, optimizer_ADAM, n_epochs, verbose=False)

# Save the model, the history and the l1_loss
torch.save(model.state_dict(), f'Models/PINN_w_{w}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}.pt')
np.savetxt(f'Models/PINN_w_{w}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_history.txt', np.array(history))
np.savetxt(f'Models/PINN_w_{w}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_l1_loss.txt', np.array(l1_loss))

#################################################################################################

#################################################################################################
#                                                                                               #
#                   PINN --> w = 15, n_hidden_layers = 4, neurons = 64                          #
#                                                                                               #
#################################################################################################
w = 15
n_hidden_layers = 4
neurons = 64

num_points = 200*15
n_epochs = 50000

# Create the model and train it
model = PINN_Cos_1D(domain_extrema, n_hidden_layers, neurons, activation_function, w)
optimizer_ADAM = optim.Adam(model.parameters(),
                            lr=float(0.001))
history, l1_loss = model.fit(num_points, optimizer_ADAM, n_epochs, verbose=False)

# Save the model, the history and the l1_loss
torch.save(model.state_dict(), f'Models/PINN_w_{w}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}.pt')
np.savetxt(f'Models/PINN_w_{w}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_history.txt', np.array(history))
np.savetxt(f'Models/PINN_w_{w}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_l1_loss.txt', np.array(l1_loss))

#################################################################################################

#################################################################################################
#                                                                                               #
#                   PINN --> w = 15, n_hidden_layers = 5, neurons = 128                         #
#                                                                                               #
#################################################################################################
w = 15
n_hidden_layers = 5
neurons = 128

num_points = 200*15
n_epochs = 50000

# Create the model and train it
model = PINN_Cos_1D(domain_extrema, n_hidden_layers, neurons, activation_function, w)
optimizer_ADAM = optim.Adam(model.parameters(),
                            lr=float(0.001))
history, l1_loss = model.fit(num_points, optimizer_ADAM, n_epochs, verbose=False)

# Save the model, the history and the l1_loss
torch.save(model.state_dict(), f'Models/PINN_w_{w}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}.pt')
np.savetxt(f'Models/PINN_w_{w}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_history.txt', np.array(history))
np.savetxt(f'Models/PINN_w_{w}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_l1_loss.txt', np.array(l1_loss))

#################################################################################################

#################################################################################################
#                                                                                               #
#                   PINN --> w = [1, 15], n_hidden_layers = 2, neurons = 16                     #
#                                                                                               #
#################################################################################################

w_list = [1, 15]
n_hidden_layers = 2
neurons = 16

num_points = 200*15
n_epochs = 50000

# Create the model and train it
model = PINN_Cos_nD(domain_extrema, n_hidden_layers, neurons, activation_function, len(w_list), w_list)
optimizer_ADAM = optim.Adam(model.parameters(),
                            lr=float(0.001))

history, l1_loss = model.fit(num_points, optimizer_ADAM, n_epochs, verbose=False)
# Save the model, the history and the l1_loss
torch.save(model.state_dict(), f'Models/PINN_w_{w_list[0]}_{w_list[1]}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}.pt')
np.savetxt(f'Models/PINN_w_{w_list[0]}_{w_list[1]}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_history.txt', np.array(history))
np.savetxt(f'Models/PINN_w_{w_list[0]}_{w_list[1]}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_l1_loss.txt', np.array(l1_loss))


#################################################################################################

#################################################################################################
#                                                                                               #
#                   PINN --> w = [1, 15], n_hidden_layers = 4, neurons = 64                     #
#                                                                                               #
#################################################################################################

w_list = [1, 15]
n_hidden_layers = 4
neurons = 64

num_points = 200*15
n_epochs = 50000

# Create the model and train it
model = PINN_Cos_nD(domain_extrema, n_hidden_layers, neurons, activation_function, len(w_list), w_list)
optimizer_ADAM = optim.Adam(model.parameters(),
                            lr=float(0.001))

history, l1_loss = model.fit(num_points, optimizer_ADAM, n_epochs, verbose=False)
# Save the model, the history and the l1_loss
torch.save(model.state_dict(), f'Models/PINN_w_{w_list[0]}_{w_list[1]}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}.pt')
np.savetxt(f'Models/PINN_w_{w_list[0]}_{w_list[1]}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_history.txt', np.array(history))
np.savetxt(f'Models/PINN_w_{w_list[0]}_{w_list[1]}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_l1_loss.txt', np.array(l1_loss))


#################################################################################################

#################################################################################################
#                                                                                               #
#                   PINN --> w = [1, 15], n_hidden_layers = 5, neurons = 128                    #
#                                                                                               #
#################################################################################################

w_list = [1, 15]
n_hidden_layers = 5
neurons = 128

num_points = 200*15
n_epochs = 50000

# Create the model and train it
model = PINN_Cos_nD(domain_extrema, n_hidden_layers, neurons, activation_function, len(w_list), w_list)
optimizer_ADAM = optim.Adam(model.parameters(),
                            lr=float(0.001))

history, l1_loss = model.fit(num_points, optimizer_ADAM, n_epochs, verbose=False)
# Save the model, the history and the l1_loss
torch.save(model.state_dict(), f'Models/PINN_w_{w_list[0]}_{w_list[1]}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}.pt')
np.savetxt(f'Models/PINN_w_{w_list[0]}_{w_list[1]}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_history.txt', np.array(history))
np.savetxt(f'Models/PINN_w_{w_list[0]}_{w_list[1]}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_l1_loss.txt', np.array(l1_loss))


#################################################################################################

#################################################################################################
#                                                                                               #
#                   FBPINN --> w = 1, n_hidden_layers = 2, neurons = 16                         #
#                              n_subdomains = 5, overlap = 1.3, sigma = 0.1                     #
#                                                                                               #
#################################################################################################
w = 1
n_hidden_layers = 2
neurons = 16

n_subdomains = 5
overlap = 1.3
sigma = 0.1

num_points = 200
n_epochs = 50000

# Create the model and train it
model = FBPPINN_Cos_1D(domain_extrema=domain_extrema, n_subdomains=n_subdomains, overlap=overlap, sigma=sigma, n_hidden_layers=n_hidden_layers, neurons=neurons, activation_function=nn.Tanh(), w=w)
history, l1_loss = model.fit(num_points, n_epochs, verbose=False)

# Save the model, the history and the l1_loss
for i in range(n_subdomains):
    # Save all the subnets
    torch.save(model.neural_networks[i].state_dict(), f'Models/FBPINN_w_{w}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_n_subdomains_{n_subdomains}_overlap_{overlap}_sigma_{sigma}_subnet_{i}.pt')
np.savetxt(f'Models/FBPINN_w_{w}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_n_subdomains_{n_subdomains}_overlap_{overlap}_sigma_{sigma}_history.txt', np.array(history))
np.savetxt(f'Models/FBPINN_w_{w}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_n_subdomains_{n_subdomains}_overlap_{overlap}_sigma_{sigma}_l1_loss.txt', np.array(l1_loss))

#################################################################################################

#################################################################################################
#                                                                                               #
#                   FBPINN --> w = 1, n_hidden_layers = 2, neurons = 16                         #
#                              n_subdomains = 5, overlap = 0.2, sigma = 0.1                     #
#                                                                                               #
#################################################################################################
w = 1
n_hidden_layers = 2
neurons = 16

n_subdomains = 5
overlap = 0.2
sigma = 0.1

num_points = 200
n_epochs = 50000

# Create the model and train it
model = FBPPINN_Cos_1D(domain_extrema=domain_extrema, n_subdomains=n_subdomains, overlap=overlap, sigma=sigma, n_hidden_layers=n_hidden_layers, neurons=neurons, activation_function=nn.Tanh(), w=w)
history, l1_loss = model.fit(num_points, n_epochs, verbose=False)

# Save the model, the history and the l1_loss
for i in range(n_subdomains):
    # Save all the subnets
    torch.save(model.neural_networks[i].state_dict(), f'Models/FBPINN_w_{w}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_n_subdomains_{n_subdomains}_overlap_{overlap}_sigma_{sigma}_subnet_{i}.pt')
np.savetxt(f'Models/FBPINN_w_{w}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_n_subdomains_{n_subdomains}_overlap_{overlap}_sigma_{sigma}_history.txt', np.array(history))
np.savetxt(f'Models/FBPINN_w_{w}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_n_subdomains_{n_subdomains}_overlap_{overlap}_sigma_{sigma}_l1_loss.txt', np.array(l1_loss))

#################################################################################################

#################################################################################################
#                                                                                               #
#                   FBPINN --> w = 15, n_hidden_layers = 2, neurons = 16                        #
#                              n_subdomains = 30, overlap = 0.3, sigma = 0.05                   #
#                                                                                               #
#################################################################################################
w = 15
n_hidden_layers = 2
neurons = 16

n_subdomains = 30
overlap = 0.3
sigma = 0.1

num_points = 200*15
n_epochs = 50000

# Create the model and train it
model = FBPPINN_Cos_1D(domain_extrema=domain_extrema, n_subdomains=n_subdomains, overlap=overlap, sigma=sigma, n_hidden_layers=n_hidden_layers, neurons=neurons, activation_function=nn.Tanh(), w=w)
history, l1_loss = model.fit(num_points, n_epochs, verbose=False)

# Save the model, the history and the l1_loss
for i in range(n_subdomains):
    # Save all the subnets
    torch.save(model.neural_networks[i].state_dict(), f'Models/FBPINN_w_{w}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_n_subdomains_{n_subdomains}_overlap_{overlap}_sigma_{sigma}_subnet_{i}.pt')
np.savetxt(f'Models/FBPINN_w_{w}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_n_subdomains_{n_subdomains}_overlap_{overlap}_sigma_{sigma}_history.txt', np.array(history))
np.savetxt(f'Models/FBPINN_w_{w}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_n_subdomains_{n_subdomains}_overlap_{overlap}_sigma_{sigma}_l1_loss.txt', np.array(l1_loss))

#################################################################################################

#################################################################################################
#                                                                                               #
#                   FBPINN --> w = [1, 15], n_hidden_layers = 2, neurons = 16                   #
#                              n_subdomains = 30, overlap = 0.3, sigma = 0.1                    #
#                                                                                               #
#################################################################################################

w_list = [1, 15]
n_hidden_layers = 2
neurons = 16

n_subdomains = 30
overlap = 0.3
sigma = 0.1

num_points = 200*15
n_epochs = 50000

# Create the model and train it
model = FBPINN_Cos_nD(domain_extrema=domain_extrema, n_subdomains=n_subdomains, overlap=overlap, sigma=sigma, n_hidden_layers=n_hidden_layers, neurons=neurons, activation_function=nn.Tanh(), n_multi_scale=len(w_list), w_list=w_list)
history, l1_loss = model.fit(num_points, n_epochs, verbose=False)

# Save the model, the history and the l1_loss
for i in range(n_subdomains):
    # Save all the subnets
    torch.save(model.neural_networks[i].state_dict(), f'Models/FBPINN_w_{w_list[0]}_{w_list[1]}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_n_subdomains_{n_subdomains}_overlap_{overlap}_sigma_{sigma}_subnet_{i}.pt')
np.savetxt(f'Models/FBPINN_w_{w_list[0]}_{w_list[1]}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_n_subdomains_{n_subdomains}_overlap_{overlap}_sigma_{sigma}_history.txt', np.array(history))
np.savetxt(f'Models/FBPINN_w_{w_list[0]}_{w_list[1]}_n_hidden_layers_{n_hidden_layers}_neurons_{neurons}_n_subdomains_{n_subdomains}_overlap_{overlap}_sigma_{sigma}_l1_loss.txt', np.array(l1_loss))