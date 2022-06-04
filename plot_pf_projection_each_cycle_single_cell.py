import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
import scipy
import scipy.stats as stats



# Particle Filter Toggles
plot_each_cycle = True # Plot the capacity trajectory prediction each cycle?
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
        # cap_trajectory_simulation_list.append(cap_trajectory_simulation)
        
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
            ax.vlines(cycle, 0, capacity, color='#707B7C', label='Current Cycle', linestyle=':', linewidth=1.5)
            ax.plot(cycles_plot_mean[cycle:], capacity_plot_mean[cycle:], color='#2E86C1', label='PF', linestyle='-.')
            ax.text(np.max(cycles)*1.50, eol_threshold - 0.032, 'EOL Threshold', fontsize=11, color='#CB4335')
            ax.set_xlim([0, np.max(cycles)*2.25])
            ax.set_ylim([eol_threshold - 0.1,1.02])
            ax.set_xlabel('Cycle Number', fontsize=fontsize)
            ax.set_ylabel('Normalized Discharge Capacity', fontsize=fontsize)
            ax.tick_params(axis='both', labelsize=fontsize-1)
            ax.arrow(cycle, eol_threshold - 0.08, 50, 0, head_width=0.01, head_length=20, fc='#707B7C', ec='#707B7C')
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
                line.set_linewidth(2.0)
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
                   'log10_a_pred':log10_a_pred,
                   'b_pred':b_pred,
                   'log10_a_samples_list':log10_a_samples_list,
                   'b_samples_list':b_samples_list,
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



## Generate PF predictions for a single cell ##



# Testing
cell_idx = 42
K_data = K_test[cell_idx]
Q_data = Q_test[cell_idx]
K_fpt = K_fpt_test[cell_idx]
K_eol = K_eol_test[cell_idx]
idx_fpt = idx_fpt_test[cell_idx][0]

log10_a_initial = -15.77
b_initial = 5.45
measurement_noise_std = 0.005
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





# Plot the pf predictions at different points in time to see why the PF has trouble converging
# to the true EOL at later cycles
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(4,3.5), dpi=400)
fig.tight_layout(pad=2.5)
title_pad = 10
fontsize = 12

ax.plot(K_data, Q_data, color='k', label='Measured\nCapacity', linewidth=1.75)
ax.set_xlim([0, np.max(results_dic['cycles'])*1.75])
ax.set_ylim([eol_threshold - 0.1,1.02])
ax.set_xlabel('Cycle Number', fontsize=fontsize)
ax.set_ylabel('Normalized Discharge Capacity', fontsize=fontsize)
ax.tick_params(axis='both', labelsize=fontsize-1)
test_str = 'Primary Test Cell'
if cell_idx >= 42:
    test_str = 'Secondary Test Cell'
    ax.set_title(test_str + ' ' + str(cell_idx + 1 - 42), fontsize=fontsize)
else:
    ax.set_title(test_str + ' ' + str(cell_idx + 1), fontsize=fontsize)
ax.hlines(eol_threshold, 0, 5000, color='#CB4335', linestyle='--')
ax.text(np.max(results_dic['cycles'])*1.15, eol_threshold - 0.033, 'EOL Threshold', fontsize=11, color='#CB4335')

linestyles = ['-.',':','--']
colors = ['#8E44AD','#2ECC71','#3498DB']

cycs = np.linspace(0,len(results_dic['cycles']),4, dtype=int)[:-1]
for i in range(0, len(cycs)):
    xx = cycs[i]
    xxx = np.arange(results_dic['cycles'][xx],5000, 1)
    plt.plot(xxx, power_law(xxx, results_dic['log10_a_pred'][xx], results_dic['b_pred'][xx]),
             linestyle=linestyles[i], linewidth=1.75, color=colors[i], label='Cycle {}'.format(results_dic['cycles'][xx]))
    # ax.vlines(results_dic['cycles'][xx], 0, 1.5, color='#707B7C', linestyle=':', linewidth=1.5)
leg = ax.legend(fontsize=fontsize - 2, loc="upper right", ncol=1, frameon=True, framealpha=1, edgecolor='k', fancybox=False)
for line in leg.get_lines():
    line.set_linewidth(2)
plt.savefig(r'Figures\many_pf_trajectory_predictions.pdf', bbox_inches = "tight")
plt.show()


