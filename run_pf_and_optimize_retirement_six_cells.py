import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
import scipy
import scipy.stats as stats


## RUN FOR ALL SIX CELLS? ##
# If true, runs the particle filter for all six cells and plots the first (takes 30 minutes)
# If false, runs only the first of the six cells and plots it
run_six = False

# Particle Filter Toggles
plot_each_cycle = False # Plot the capacity trajectory prediction each cycle?
plot_final_RUL = True # Plot the predicted RUL over all cycles once complete?




###############################################################################

# Plotting preferences
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.default"] = "rm"
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'Times New Roman'



###############################################################################
# STEP 1 - Load in the data, split into train/test, and extract relevant features

# load in the capacity dataset
dataset = pickle.load(open(r'capacity_dataset_124_lfp.pkl', 'rb'))


# Extract all data
all_cycles = dataset['x_train'] + dataset['x_test1'] + dataset['x_test2']
all_capacities = dataset['y_train'] + dataset['y_test1'] + dataset['y_test2']
all_c1 = list(dataset['train_conds']['c1'].to_numpy()) + list(dataset['test1_conds']['c1'].to_numpy()) + list(dataset['test2_conds']['c1'].to_numpy())
all_percent = list(dataset['train_conds']['percent'].to_numpy()) + list(dataset['test1_conds']['percent'].to_numpy()) + list(dataset['test2_conds']['percent'].to_numpy())
all_c2 = list(dataset['train_conds']['c2'].to_numpy()) + list(dataset['test1_conds']['c2'].to_numpy()) + list(dataset['test2_conds']['c2'].to_numpy())


# Fix seed for reproduceability
np.random.seed(2)

# Randomly split the cells into training and test
# ids = np.random.permutation(np.arange(0,123,1))
ids = np.arange(0,123,1)
train_ids = ids[0:41] # use only cells in the small training dataset for hyperparameter optimization
test_ids = ids[41:]

# Split the data according to the random indices
# Extract the required matricies for training the model.

# Q capacity values
# K cycle numbers
Q_train = []
K_train = []
Q_test = []
K_test = []

Q_eol_train = []
Q_eol_test = []
K_eol_train = []
K_eol_test = []

K_fpt_train = []
K_fpt_test = []
idx_fpt_train = []
idx_fpt_test = []
Q_0_train = []
Q_0_test = []

conditions_train = []
conditions_test = []

# Define the capacity threshold at which RUL prediction begins.
cap_threshold = 0.985 # corresponds to a 98% remaining normalized capacity FPT threshold.

# Define the threshold at which the cell fails. 
eol_threshold = 0.50 # The percentage of normalized capacity remaining

# If we choose to, we could subsample the data and make the algorithm run more quickly. 
# However, the subsample feature was not implemented for the multi-attribute utility 
# portion of this work, and therefore, subsample should always be set to 1. 
subsample = 1
for i in range(0,len(all_cycles)):
    if i in train_ids:
        q_vec = all_capacities[i][0::subsample] / all_capacities[i][0]
        # Q_train.append(np.where(q_vec >= 1.0, 0.999, q_vec))
        Q_train.append(q_vec)
        Q_0_train.append(all_capacities[i][0])
        K_train.append(all_cycles[i][0::subsample])        
        # K_fpt_train.append(all_cycles[i][0::subsample][np.where(np.where(q_vec >= 1.0, 0.999, q_vec) <= cap_threshold)[0][0]])
        # idx_fpt_train.append([np.where(np.where(q_vec >= 1.0, 0.999, q_vec) <= cap_threshold)[0][0]])
        K_fpt_train.append(all_cycles[i][0::subsample][np.where(q_vec <= cap_threshold)[0][0]])
        idx_fpt_train.append([np.where(q_vec <= cap_threshold)[0][0]])
        conditions_train.append(np.hstack((all_c1[i], all_percent[i]/100, all_c2[i])).reshape(1,-1))
        if len(np.where(q_vec <= eol_threshold)[0]) != 0:
            Q_eol_train.append(all_capacities[i][np.where(q_vec <= eol_threshold)[0][0]])
            K_eol_train.append(all_cycles[i][0::subsample][np.where(q_vec <= eol_threshold)[0][0]])
        else:
            Q_eol_train.append(all_capacities[i][-1])
            K_eol_train.append(all_cycles[i][0::subsample][-1])
    if i in test_ids:
        q_vec = all_capacities[i][0::subsample] / all_capacities[i][0]
        # Q_test.append(np.where(q_vec >= 1.0, 0.999, q_vec))
        Q_test.append(q_vec)
        Q_0_test.append(all_capacities[i][0])
        K_test.append(all_cycles[i][0::subsample])
        # K_fpt_test.append(all_cycles[i][0::subsample][np.where(np.where(q_vec >= 1.0, 0.999, q_vec) <= cap_threshold)[0][0]])
        # idx_fpt_test.append([np.where(np.where(q_vec >= 1.0, 0.999, q_vec) <= cap_threshold)[0][0]])
        K_fpt_test.append(all_cycles[i][0::subsample][np.where(q_vec <= cap_threshold)[0][0]])
        idx_fpt_test.append([np.where(q_vec <= cap_threshold)[0][0]])
        conditions_test.append(np.hstack((all_c1[i], all_percent[i]/100, all_c2[i])).reshape(1,-1))
        if len(np.where(q_vec <= eol_threshold)[0]) != 0:
            Q_eol_test.append(all_capacities[i][np.where(q_vec <= eol_threshold)[0][0]])
            K_eol_test.append(all_cycles[i][0::subsample][np.where(q_vec <= eol_threshold)[0][0]])
        else:
            Q_eol_test.append(all_capacities[i][-1])
            K_eol_test.append(all_cycles[i][0::subsample][-1])

# Stack the features in an array instead of a list.
Q_eol_train = np.vstack(Q_eol_train)
Q_eol_test = np.vstack(Q_eol_test)
K_eol_train = np.vstack(K_eol_train)
K_eol_test = np.vstack(K_eol_test)





###############################################################################





# This is the function which describes the capacity of a cell as a function of cycle number. 
def power_law(x, log10_a, b):
    return 1 - ((10**log10_a)*x**b)


# Set the number of particles used to estimate the states of the capacity fade model parameters
n_particles = 200

# Set the number of random samples to draw from the particle distribution for probabilistically
# estimating the EOL and RUL of the cell.
n_param_samples = 500


def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, np.size(mean,0)))
    for i in range(0,len(mean)):
        particles[:, i] = mean[i] + (np.random.randn(N) * std[i])
    return particles

def predict(particles, std):
    particles_propogated = np.empty((np.size(particles,0), np.size(particles,1)))
    for i in range(0,np.size(particles,1)):
        particles_propogated[:, i] = particles[:,i] + (np.random.randn(len(particles[:,i])) * std[i])
    return particles_propogated

def update(particles, weights, measurement_noise_std, capacity, cycle):
    # The particles approximate the capacity using the power law model
    capacity_pred = power_law(cycle, particles[:,0], particles[:,1])
    # Determine which values are closest to the capacity measurement
    pdf_vals = (scipy.stats.norm(capacity_pred, measurement_noise_std).pdf(capacity))
    
    # Update the weights of the particles based on their closeness to the true measurement
    weights_updated = np.zeros_like(weights)
    for i in range(0,np.size(weights,1)):
        # Scale the weights by the PDF values and normalize
        weights_updated[:, i] = (weights[:, i] * pdf_vals) / sum((weights[:, i] * pdf_vals))
        # Normalize weights
    
    return weights_updated

def estimate(particles, weights):
    means_ = np.zeros((np.size(weights,1)))
    std_ = np.zeros((np.size(weights,1)))
    for i in range(0,np.size(weights,1)):
        means_[i] = np.average(particles[:,i], weights=weights[:,i], axis=0)
        std_[i] = np.sqrt(np.average((particles[:,i] - means_[i])**2, weights=weights[:,i], axis=0))
    return means_, std_

def resample(particles, weights):
    resampled_particles = np.zeros_like(particles)
    for i in range(0,np.size(particles,1)):
        cdf = np.cumsum(weights[:,i])
        cdf[-1] = 1. # avoid round-off error
        randomNums = np.random.random(len(particles))
        indexes = np.searchsorted(cdf, randomNums)
        resampled_particles[:,i] = particles[indexes,i]
    # Reset the weights
    weights = 1/n_particles*np.ones_like((particles))
    return resampled_particles, weights



def PF(K_data, Q_data, K_fpt, K_eol, idx_fpt, eol_threshold,
       log10_a_initial, b_initial, measurement_noise_std, process_noise_std,
       n_particles, plot_each_cycle=False, plot_final_RUL=False):
    
    # Initialize the cell data
    cycles = np.arange(K_fpt, K_eol + subsample, subsample, dtype=int).reshape(-1)
    capacities = Q_data[idx_fpt:]
    eol_cycle = K_eol
    
    # Generate particles
    particles = create_gaussian_particles((log10_a_initial, b_initial),
                                          (process_noise_std[0],
                                           process_noise_std[1]),
                                          n_particles)

    # Initialize particle weights
    weights = 1/n_particles*np.ones_like((particles))
    
    log10_a_pred = []
    b_pred = []
    log10_a_pred_std = []
    b_pred_std = []
    eol_pred = []
    rul_pred = []
    rul_pred_lb = []
    rul_pred_ub = []
    
    # For simulating multiple trajectories
    cap_trajectory_simulation_list = []
    eol_samples_list = []
    rul_samples_list = []
    
    log10_a_samples_list = []
    b_samples_list = []
    
    true_rul = np.arange(eol_cycle,0,-subsample)[idx_fpt:]
    for i in range(0,len(cycles)):
        cycle = cycles[i]
        capacity = capacities[i]
        
        ## Particle Filter ##
        
        ## STEP 1 ##
        # Propogate the particles through time using "random walk" model for each parameter. 
        # Predict the distribution of log10(a) and b one step in the future
        particles = predict(particles, (process_noise_std[0], process_noise_std[1]))
        
        ## Step 2 ##
        # Update the weights using the latest capacity measurement        
        weights = update(particles, weights, measurement_noise_std, capacity, cycle)
        
        
        ## Step 3 ##
        # Estimate the values of log10(a) and b using the particles and their updated weights.        
        estimated_parameter_means, estimated_parameter_stds = estimate(particles, weights)
        
        # Generate many random log10(a) and b values from the particle distributions.
        # The samples will be used to probabilistically determine the EOL and RUL of the cell
        log10_a_samples = np.random.choice(particles[:,0], n_param_samples, p=weights[:,0])
        b_samples = np.random.choice(particles[:,1], n_param_samples, p=weights[:,1])
        log10_a_samples_list.append(log10_a_samples)
        b_samples_list.append(b_samples)
        
        ## Step 4 ##
        # Resample the particles
        particles, weights = resample(particles, weights)
        
        ## End Particle Filter ##
        
        
        #######################################################################
        
        
        ## Predict mean EOL and RUL using Particle Filter Outputs ##
        # Below is code used to generate end-of-life (EOL) and remaining useful life (RUL) predictions 
        # using the mean and std of the weighted particle distribution.
        
        # Append the predictions to the lists to keep track of them 
        log10_a_pred.append(estimated_parameter_means[0])
        b_pred.append(estimated_parameter_means[1])
        
        # Calculate three capacity trajectories starting from cycle 0 for plotting.
        # Mean trajectory
        cap_long = power_law(np.arange(0,10000,1), estimated_parameter_means[0], estimated_parameter_means[1])
        cycles_long = np.arange(0,10000,1)
        capacity_plot_mean = cap_long[np.where(cap_long >= eol_threshold)[0]]
        cycles_plot_mean = np.arange(0,10000,1)[np.where(cap_long >= eol_threshold)[0]]
        
        # Use the randomly drawn samples of log10(a) and b to simulate many capacity trajectories
        # and determine the EOL, RUL, and their non-parametric distribution.
        # Calculate the capacity over 10,000 cycles so the prediction crosses the EOL threshold
        cycles_long = np.arange(cycle,10000,subsample)
        cap_trajectory_simulation = np.zeros((len(cycles_long), len(log10_a_samples)))
        for j in range(0, len(log10_a_samples)):
            cap_trajectory_simulation[:,j] = power_law(cycles_long, log10_a_samples[j], b_samples[j])
        
        eol_samples = []
        rul_samples = []
        for j in range(0,len(log10_a_samples)):
            # Calculate EOL and RUL
            if len(np.where(cap_trajectory_simulation[:,j] <= eol_threshold)[0]) >= 1:
                eol_idx = np.where(cap_trajectory_simulation[:,j] <= eol_threshold)[0][0]
                eol = cycles_long[eol_idx]
                eol_samples.append(eol)
                rul_samples.append(eol - cycle)
            else:
                eol = 100000
                eol_samples.append(eol)
                rul_samples.append(eol - cycle)
        eol_samples = np.vstack(eol_samples).reshape(-1)
        rul_samples = np.vstack(rul_samples).reshape(-1)
        # eol_samples_list.append(eol_samples)
        # rul_samples_list.append(rul_samples)
        
        # Calculate the EOL and RUL as the median of the distribution.
        eol_pred.append(np.median(eol_samples))
        rul_pred.append(np.median(rul_samples))
        
        # Calculate the 5th and 95th percentile values to be used as a non-parametric confidence interval
        percentiles = np.percentile(rul_samples, (5, 95))
        rul_pred_lb.append(percentiles[0])
        rul_pred_ub.append(percentiles[1])
        
        
        
        if plot_each_cycle == True:
            
            fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(4,3.5), dpi=400)
            fig.tight_layout(pad=2.5)
            title_pad = 10
            fontsize = 12
            
            # Plot the capacity data, EOL threshold, and the PF trajectory
            ax.plot(K_data, Q_data, color='k', label='Measured\nCapacity')
            ax.hlines(eol_threshold, 0, 2500, color='#CB4335', linestyle='--')
            ax.vlines(cycle, 0, capacity, color='#707B7C', label='Current Cycle', linestyle=':', linewidth=1.25)
            ax.plot(cycles_plot_mean, capacity_plot_mean, color='#2E86C1', label='PF', linestyle='-.')
            ax.text(np.max(cycles)*1.50, eol_threshold - 0.018, 'EOL Threshold', fontsize=11, color='#CB4335')
            ax.set_xlim([0, np.max(cycles)*2.25])
            ax.set_ylim([eol_threshold - 0.1,1.02])
            ax.set_xlabel('Cycle Number', fontsize=fontsize)
            ax.set_ylabel('Normalized Discharge Capacity', fontsize=fontsize)
            ax.tick_params(axis='both', labelsize=fontsize-1)
            ax.arrow(cycle, eol_threshold - 0.08, 50, 0, head_width=0.005, head_length=17, fc='#707B7C', ec='#707B7C')
            test_str = 'Primary Test Cell'
            if cell_idx >= 42:
                test_str = 'Secondary Test Cell'
            ax.set_title(test_str + ' ' + str(cell_idx + 1), fontsize=fontsize)
            
            # Plot the probabilty density for the PF EOL estimate
            density = stats.gaussian_kde(eol_samples)
            n, xx, _ = ax.hist(eol_samples, color='#3498DB', alpha=0.2, density=True, bins=50, bottom=0)
            density = density(xx)
            scaled_density = ((density - np.min(density)) / (np.max(density) - np.min(density))) * (0.05) + eol_threshold
            ax.plot(xx, scaled_density, color='#3498DB', linewidth=1.5, alpha=0.7)
            ax.fill_between(xx, eol_threshold, scaled_density, alpha=.5, fc='#3498DB', ec='None', label='EOL PDF')
            
            leg = ax.legend(fontsize=fontsize - 2, loc="upper right", ncol=1, frameon=True, framealpha=1, edgecolor='k', fancybox=False)
            for line in leg.get_lines():
                line.set_linewidth(3.0)
            # plt.savefig(r'Figures\pf_eol_prediction.pdf', bbox_inches = "tight")
            plt.show()
            # programPause = input("Press the <ENTER> key to continue...")
    
    # Gather all data and convert lists to arrays for plotting
    eol_pred = np.vstack(eol_pred)
    rul_pred = np.vstack(rul_pred)
    rul_pred_lb = np.vstack(rul_pred_lb)
    rul_pred_ub = np.vstack(rul_pred_ub)
    log10_a_pred = np.vstack(log10_a_pred)
    b_pred = np.vstack(b_pred)
    
    results_dic = {'eol_pred':eol_pred,
                   'rul_pred':rul_pred,
                   'true_rul':true_rul,
                   'rul_pred_lb':rul_pred_lb,
                   'rul_pred_ub':rul_pred_ub,
                    'log10_a_pred':log10_a_pred,
                    'b_pred':b_pred,
                   # 'log10_a_samples_list':log10_a_samples_list,
                   # 'b_samples_list':b_samples_list,
                   'cycles':cycles,
                   'capacities':capacities,
                   # 'cap_trajectory_simulation_list':cap_trajectory_simulation_list,
                   # 'eol_samples_list':eol_samples_list,
                   # 'rul_samples_list':rul_samples_list
                   }
    
    # Plot final RUL
    if plot_final_RUL == True:
        plt.figure(2, dpi=400, figsize=(4,3.5))
        fontsize = 12
        plt.plot(np.arange(2,eol_cycle,subsample), np.arange(eol_cycle,2,-subsample), color='k', label='True RUL')
        plt.plot(cycles, rul_pred, color='#2E86C1', label='PF')
        plt.fill(np.concatenate([cycles, cycles[::-1]]), np.concatenate([rul_pred_lb, rul_pred_ub[::-1]]), alpha=.3, fc='#3498DB', ec='None', label='95% Confidence\nInterval')
        plt.xlabel('Cycle Number', fontsize=fontsize)
        plt.ylabel('RUL', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(fontsize=fontsize - 2, loc="upper right", ncol=1, frameon=True, framealpha=1, edgecolor='k', fancybox=False)
        plt.savefig(r'Figures\pf_rul_prediction.pdf', bbox_inches = "tight")
        test_str = 'Primary Test Cell'
        if cell_idx >= 42:
            test_str = 'Secondary Test Cell'
        plt.title('{} {}'.format(test_str, str(cell_idx + 1)), fontsize=fontsize)
        plt.show()
    return results_dic





##############################################################################



## Generate PF predictions for a group of 6 cells  ##
rand_cells = np.random.permutation(len(test_ids))

if run_six == True:
    rand_cells = rand_cells[0:6]
else:
    rand_cells = rand_cells[0:1]

results_lists = []


for i in range(0,len(rand_cells)):
    
    # Select the data for the cell
    cell_idx = int(rand_cells[i]) # 34 is a really accurate prediction #21 
    K_data = K_test[cell_idx]
    Q_data = Q_test[cell_idx]
    K_fpt = K_fpt_test[cell_idx]
    K_eol = K_eol_test[cell_idx]
    idx_fpt = idx_fpt_test[cell_idx][0]
    
    log10_a_initial = -15.77
    b_initial = 5.45
    measurement_noise_std = 0.01
    process_noise_std = (0.05, 0.05)
    
    # Set the number of particles used to estimate the states of the capacity fade model parameters
    n_particles = 200
    
    # Set the number of random samples to draw from the particle distribution for probabilistically
    # estimating the EOL and RUL of the cell.
    n_param_samples = 500
    
    # Test the model on a single cell.
    results_dic = PF(K_data, Q_data, K_fpt, K_eol, idx_fpt, eol_threshold, log10_a_initial,
                     b_initial, measurement_noise_std, process_noise_std, n_particles,
                     plot_each_cycle=plot_each_cycle, plot_final_RUL=plot_final_RUL)
    
    results_lists.append(results_dic)






###############################################################################




## Optimize replacement time for one of the six randomly slected cells ##

# Select which of the six test cells to optimize the retirement on.

cell = 0 # 0-5

# Select the data for the cell
cell_idx = rand_cells[cell]
K_data = K_test[cell_idx]
Q_data = Q_test[cell_idx]
K_fpt = K_fpt_test[cell_idx]
K_eol = K_eol_test[cell_idx]
idx_fpt = idx_fpt_test[cell_idx][0]

results_dic = results_lists[cell]





# When the cell gets to 95% capacity, we need to determine when we want to remove
# it from service. 95% tends to be around the knee-point for many of the cells.
# This is a good time to determine when it should be removed from service. 

# Determine the PF's capacity trajectory prediction when the cell was at 95% capacity. 
# The threshold can be changed if desired. 

opt_capacity_threshold = 0.95

# Grab the PF estimated model parameters for that cycle
opt_log10_a = results_dic['log10_a_pred'][np.where(results_dic['capacities'] <= opt_capacity_threshold)[0][0]]
opt_b = results_dic['b_pred'][np.where(results_dic['capacities'] <= opt_capacity_threshold)[0][0]]

# Grab the original raw capacity to show what the "true" optimal replacement would have been
complete_original_capacity = (Q_data * Q_0_test[cell_idx])[0:int(K_eol_test[cell_idx])]
complete_original_cycles = K_test[cell_idx][0:int(K_eol_test[cell_idx])]

# Now, we need to mimic what it would be like if the cell were online. We would not 
# have all the true capacity data after the current cycle. We will replace the future capacity
# data with the projected capacity from the Particle Filter. 

# Determine the index where we no longer have the true capacity measurements
online_idx = np.where(Q_data <= opt_capacity_threshold)[0][0]
combined_pf_capacity = np.hstack((Q_data[0:online_idx],
                                  power_law(complete_original_cycles[online_idx:], opt_log10_a, opt_b)
                                  )) * Q_0_test[cell_idx]


# Plot the two capacity trajectories for comparison
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(4,3.5), dpi=400)
fig.tight_layout(pad=2.5)
fontsize = 12

ax.plot(complete_original_cycles, complete_original_capacity, color='k', label='Measured\nCapacity')
ax.hlines(opt_capacity_threshold * Q_0_test[cell_idx], 0, 2500, color='#707B7C', linewidth=1.25, linestyle=':')
ax.vlines(complete_original_cycles[online_idx], 0, 2, color='#707B7C', label='Current Cycle', linewidth=1.25, linestyle=':')
ax.plot(complete_original_cycles, combined_pf_capacity, color='#2E86C1', label='Measured + PF', linestyle='-.')
# ax.text(np.max(copmlete_original_cycles)*1.50, opt_capacity_threshold * Q_0_test[cell_idx] - 0.018, 'EOL Threshold', fontsize=11, color='#CB4335')
ax.set_xlim([0, np.max(complete_original_cycles)*2.25])
ax.set_ylim([.50,1.20])
ax.set_xlabel('Cycle Number', fontsize=fontsize)
ax.set_ylabel('Discharge Capacity (Ah)', fontsize=fontsize)
ax.tick_params(axis='both', labelsize=fontsize-1)
ax.arrow(complete_original_cycles[online_idx], 0.72, 50, 0, head_width=0.005, head_length=17, fc='#707B7C', ec='#707B7C')
test_str = 'Primary Test Cell'
if cell_idx >= 42:
    test_str = 'Secondary Test Cell'
ax.set_title(test_str + ' ' + str(cell_idx + 1), fontsize=fontsize)

leg = ax.legend(fontsize=fontsize - 2, loc="upper right", ncol=1, frameon=True, framealpha=1, edgecolor='k', fancybox=False)
for line in leg.get_lines():
    line.set_linewidth(1.5)
# plt.savefig(r'Figures\pf_eol_prediction.pdf', bbox_inches = "tight")
plt.show()



# Loop through all the cells and determine the cycle at which we will determine the 
# optimal replacement. 
K_opt_train = []
K_opt_test = []
for i in range(0,len(K_train)):
    K_opt_train.append(K_train[i][np.where(Q_train[i] <= opt_capacity_threshold)[0][0]])
for i in range(0,len(K_test)):
    K_opt_test.append(K_test[i][np.where(Q_test[i] <= opt_capacity_threshold)[0][0]])




# Define the function used to calculate the mean recharge interval over a
# given interval
def calculate_mean_charge_time(cycle_1, cycle_2, K_data, Q_data, conditions):
    # Ensure the cycles are integers
    cycle_1 = int(cycle_1)
    cycle_2 = int(cycle_2)
    
    cycle_idxs = np.arange(cycle_1 - 2, cycle_2 - 2, 1, dtype=int)
    
    # Assuming the discharge capacity is the same as the charge capacity, 
    # we calculate the time of first charging current up to the % cutoff, and 
    # then calculate the second charge time up to the 80% cutoff. After 80% SOC,
    # all cells use a 1C CC CV step, which we are not accounting for here.
    amperage_1 = conditions[:,0] * 1.1 # Convert C-rate to Amps using the nominal capacity
    amperage_2 = conditions[:,2] * 1.1
    
    # Now we calculate the time in hours for each of the two charging steps. 
    # This is dataset specific because the authors used two-step fast charging 
    # protocols. The third step is 1C CC-CV charge from 80%-100% SOC. 
    # We will add in the 1C CC step, but ignore the CV step because it is cell
    # specific and the data is not easily accessible.
    chg_time_1 = (Q_data[cycle_idxs] * conditions[:,1]) / amperage_1 # ((Ah * %) / A) = hours
    chg_time_2 = (Q_data[cycle_idxs] * (0.8 - conditions[:,1])) / amperage_2
    chg_time_3 = (Q_data[cycle_idxs] * 0.2) / 1.1 # 1.1A is 1C becuase 1.1Ah nominal Capacity
    # All cells discharge at 4C. Nominal Capacity is 1.1 Ah so 4C is 4.4 A.
    mean_chg_time = np.mean(chg_time_1 + chg_time_2 + chg_time_3) # Units of hours
    return mean_chg_time # Units of hours


def calculate_mean_discharge_time(cycle_1, cycle_2, K_data, Q_data):
    # Ensure the cycles are integers
    cycle_1 = int(cycle_1)
    cycle_2 = int(cycle_2)
    
    cycle_idxs = np.arange(cycle_1 - 2, cycle_2 - 2, 1, dtype=int)
    # All cells discharge at 4C. Nominal Capacity is 1.1 Ah so 4C is 4.4 A.
    mean_recharge_interval = np.mean(Q_data[cycle_idxs] / 4.4) # Units of hours
    return mean_recharge_interval # Units of hours


# Define the function used to calculate the total Ah output from the battery over
# a given interval.
def calculate_total_discharge_Ah(cycle_1, cycle_2, K_data, Q_data):
    # Ensure the cycles are integers
    cycle_1 = int(cycle_1)
    cycle_2 = int(cycle_2)
    
    cycle_idxs = np.arange(cycle_1 - 2, cycle_2 - 2, 1, dtype=int)
    cumulative_Ah = np.sum(Q_data[cycle_idxs])
    return cumulative_Ah # Units of Ah



# Go through each of the training cells and calculate the maximum attribute values so that 
# effective utility functins can be defined
total_ah_all_cells = []
for i in range(0,len(Q_train)):
    total_ah_all_cells.append(calculate_total_discharge_Ah(2, K_eol_train[i], K_train[i], Q_train[i]))

# # Plot histogram to see typical values
# plt.figure(dpi = 400)
# plt.hist(total_ah_all_cells, bins=25)
# plt.show()

ah_lb = 300
ah_ub = 1000
ah_1_bounds = [ah_lb, ah_ub]


# Do the same for the next attribute
mtbc_all_cells = []
for i in range(0,len(Q_train)):
    mtbc_all_cells.append(calculate_mean_discharge_time(2, K_eol_train[i]/2, K_train[i], Q_train[i]))

# # Plot histogram to see typical values
# plt.figure(dpi = 400)
# plt.hist(mtbc_all_cells, bins=25)
# plt.show()

mtbc_lb = 0.210 #0.215
mtbc_ub = 0.250
dchg_time_1_bounds = [mtbc_lb, mtbc_ub]


# Define the utility function which is used to map total Ah to the range 0-1.
# This function encodes our preference for greater first life use.
def total_discharge_ah_utility(Ah, lower_bound, upper_bound):
    Ah = np.array(Ah).reshape(-1)
    utility = []
    for i in range(0,np.size(Ah,0)):
        if Ah[i] < lower_bound:
            utility.append(0)
        if Ah[i] > upper_bound:
            utility.append(1)
        if (Ah[i] >= lower_bound) and (Ah[i] <= upper_bound):
            # utility.append((1 / (upper_bound - lower_bound)) * Ah[i] + (-lower_bound / (upper_bound - lower_bound)))
            # utility.append( 1 - np.exp((-Ah[i] + lower_bound) / 0.03))
            R = 200
            # lower_bound = lower_bound * 0.75
            # upper_bound = upper_bound * 1.25
            a = (np.exp(-lower_bound / R)) / (np.exp(-lower_bound / R) - np.exp(-upper_bound / R))
            b = 1 / (np.exp(-lower_bound / R) - np.exp(-upper_bound / R))
            utility.append(a - b*np.exp(-Ah[i] / R))
    return np.vstack(utility)


# Define the utility function which is used to map the dchg time to the
# range 0-1. This function encodes our preference for longer lifetime cells
def mean_dchg_time_utility(mean_hours, lower_bound, upper_bound):
    mean_hours = np.array(mean_hours).reshape(-1)
    utility = []
    for i in range(0,np.size(mean_hours,0)):
        if mean_hours[i] < lower_bound:
            utility.append(0)
        if mean_hours[i] > upper_bound:
            utility.append(1)
        if (mean_hours[i] >= lower_bound) and (mean_hours[i] <= upper_bound):
            # utility.append((1 / (upper_bound - lower_bound)) * mean_hours[i] + (-lower_bound / (upper_bound - lower_bound)))
            # utility.append( 1 - np.exp(((-mean_hours[i] + (lower_bound)) / 0.02)))
            # utility.append( np.exp((-mean_hours[i] + upper_bound) / -0.01))
            R = .015 # 0.005
            a = (np.exp(-lower_bound / R)) / (np.exp(-lower_bound / R) - np.exp(-upper_bound / R))
            b = 1 / (np.exp(-lower_bound / R) - np.exp(-upper_bound / R))
            utility.append(a - b*np.exp(-mean_hours[i] / R))
    return np.vstack(utility)


# Define the utility function which is used to map the mean chg time to the
# range 0-1. This function encodes our preference for faster charging times
def mean_chg_time_utility(mean_hours, lower_bound, upper_bound):
    mean_hours = np.array(mean_hours).reshape(-1)
    utility = []
    for i in range(0,np.size(mean_hours,0)):
        if mean_hours[i] < lower_bound:
            utility.append(1)
        if mean_hours[i] > upper_bound:
            utility.append(0)
        if (mean_hours[i] >= lower_bound) and (mean_hours[i] <= upper_bound):
            utility.append((-1 / (upper_bound - lower_bound)) * mean_hours[i] + (upper_bound / (upper_bound - lower_bound)))
    return np.vstack(utility)





# Plot the two utility functions
# ah_utility_values = []
# mtbc_utility_values = []
ah_utility_values = total_discharge_ah_utility(np.linspace(3, 2000, 1000), ah_1_bounds[0], ah_1_bounds[1])
mtbc_utility_values = mean_dchg_time_utility(np.linspace(0.15, 0.3, 1000), dchg_time_1_bounds[0], dchg_time_1_bounds[1])

linestyles = ['-.','--','-']
colors = ['#8E44AD','#2ECC71','#3498DB']
fontsize=14

# Total Ah utility
plt.figure(dpi=400, figsize=(4,3.5))
plt.plot(np.linspace(3, 2000, 1000), ah_utility_values, color=colors[0], linestyle=linestyles[0], linewidth=2)
plt.xlabel('Total Ah Throughput', fontsize=fontsize)
plt.ylabel('Utility', fontsize=fontsize)
plt.savefig(r'Figures\Ah_utility.pdf', bbox_inches = "tight")
plt.show()

# MTBC utility
plt.figure(dpi=400, figsize=(4,3.5))
plt.plot(np.linspace(0.15, 0.3, 1000), mtbc_utility_values, color=colors[1], linestyle=linestyles[1], linewidth=2)
plt.xlabel('Mean Time Between Charges', fontsize=fontsize)
plt.ylabel('Utility', fontsize=fontsize)
plt.savefig(r'Figures\MTBC_utility.pdf', bbox_inches = "tight")
plt.show()





###############################################################################



## Perform the optimization ##

# consider only total Ah and the MTBC
weights = np.array([0.5, 0.5])
def objective_function(x, K_data, Q_data, weights):
    x = int(x)
    # Total Ah throughput
    Ah_utility_first_life = total_discharge_ah_utility(calculate_total_discharge_Ah(2, x, K_data, Q_data), ah_1_bounds[0], ah_1_bounds[1])
    
    # Mean Discharge Time
    DChg_utility_first_life = mean_dchg_time_utility(calculate_mean_discharge_time(2, x, K_data, Q_data), dchg_time_1_bounds[0], dchg_time_1_bounds[1])

    first_life_utility = (weights[0] * Ah_utility_first_life) + (weights[1] * DChg_utility_first_life)
    
    return np.hstack((first_life_utility, Ah_utility_first_life, DChg_utility_first_life)).reshape(1,-1)

function_values = []
utility_cycles = np.arange(K_opt_test[cell_idx], int(K_eol-1), 1)
for x in utility_cycles:
    function_values.append(objective_function(x, complete_original_cycles, combined_pf_capacity, weights))
function_values = np.vstack(function_values)
idx_max = np.argmax(function_values[:,0])
cycle_max = utility_cycles[idx_max]
print('\n\nOptimal Cycle: {}\nEnd of Life: {}\n'.format(cycle_max, K_eol))




# Plot the single cell optimization results. 
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize=(4,7), dpi=400)
fig.tight_layout(pad=2.75)
fontsize = 14
aa = 0
start = K_opt_test[cell_idx]
ax[aa].plot(K_data, Q_data, color='k', label='Measured Capacity', linewidth=1.75)
ax[aa].plot(complete_original_cycles[start:], combined_pf_capacity[start:] / combined_pf_capacity[0], color='#2E86C1', label='PF Projection', linestyle='--', linewidth=1.75)
ax[aa].vlines(cycle_max, -1, 2, color='#E74C3C', label='Optimal Retirement', linestyle='-.', linewidth=1.75)
ax[aa].vlines(complete_original_cycles[online_idx], -10, 12, color='#707B7C', label='Current Cycle', linewidth=1.75, linestyle=':')
ax[aa].set_xlabel('Cycle Number', fontsize=fontsize)
ax[aa].set_ylabel('Normalized Discharge Capacity', fontsize=fontsize)
ax[aa].tick_params(axis='both', labelsize=fontsize-1)
ax[aa].set_ylim([0, 1.05])
ax[aa].set_xlim([np.min(complete_original_cycles), K_eol])
ax[aa].grid()
test_str = 'Primary Test Cell'
if cell_idx >= 42:
    test_str = 'Secondary Test Cell'
    ax[aa].set_title(test_str + ' ' + str(cell_idx + 1 - 42), fontsize=fontsize)
else:
    ax[aa].set_title(test_str + ' ' + str(cell_idx + 1), fontsize=fontsize)
##
leg = ax[aa].legend(fontsize=fontsize - 2, loc="upper center", ncol=2, frameon=True, framealpha=1, edgecolor='k', fancybox=False, 
                bbox_to_anchor=(0.5,1.35))
for line in leg.get_lines():
    line.set_linewidth(1.5)
aa = 1
linestyles = ['-.','--','-']
colors = ['#8E44AD','#2ECC71','#3498DB']
ax[aa].plot(utility_cycles, 0.5*function_values[:,1], color=colors[0], label='Total Ah', linestyle=linestyles[0])
ax[aa].plot(utility_cycles, 0.5*function_values[:,2], color=colors[1], label='MTBC', linestyle=linestyles[1])
ax[aa].plot(utility_cycles, function_values[:,0], color=colors[2], label='Overall', linestyle=linestyles[2])
ax[aa].vlines(complete_original_cycles[online_idx], -10, 12, color='#707B7C', label='Current Cycle', linewidth=1.25, linestyle=':')
ax[aa].vlines(cycle_max, -1, 2, color='#E74C3C', label='Optimal Retirement', linestyle='-.', linewidth=1.75)
ax[aa].plot(cycle_max, function_values[idx_max,0], color='#E74C3C', marker='o', markersize=6)
ax[aa].set_xlabel('Cycle Number', fontsize=fontsize)
ax[aa].set_ylabel('Utiilty', fontsize=fontsize)
ax[aa].tick_params(axis='both', labelsize=fontsize-1)
ax[aa].set_ylim([0, 1.05])
ax[aa].grid()
ax[aa].set_xlim([np.min(complete_original_cycles), K_eol])
leg = ax[aa].legend(fontsize=fontsize - 2, loc="upper center", ncol=2, frameon=True, framealpha=1, edgecolor='k', fancybox=False,
                    bbox_to_anchor=(0.5,-0.18))
for line in leg.get_lines():
    line.set_linewidth(1.5)
plt.savefig(r'Figures\utility_panel_cell{}.pdf'.format(cell_idx), bbox_inches = "tight")
plt.show()




