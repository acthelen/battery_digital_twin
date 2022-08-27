import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

# load in the conditions of each cell and seperate them
conditions = pd.read_excel('124 LFP Cell Conditions.xlsx')
train_conds = conditions[ conditions['Dataset'] == 'Train']
test1_conds = conditions[ conditions['Dataset'] == 'Prim. Test']
test2_conds = conditions[ conditions['Dataset'] == 'Sec. test']

# The original authors of the dataset noticed that one of the cells in the primary
# test dataset had very short lifetime. The authors removed the cell from the 
# dataset before performing any analysis. We have done the same. This is why
# the dataset only has 123 cells in total when it is labeled "124." 
test1_conds = test1_conds.drop(index=42) # Was previously 44 which was incorrect. 

# Load in the data
x_train = []
y_train = []
for i in range(0,41):
    data = np.loadtxt(r'124 LFP Capacity Data\train\cell{}.csv'.format(i+1), 
                      delimiter=',')
    x_train.append(data[:,0])
    y_train.append(data[:,1])

x_test1 = []
y_test1 = []
for i in range(0,42):
    data = np.loadtxt(r'124 LFP Capacity Data\test1\cell{}.csv'.format(i+1), 
                      delimiter=',')
    x_test1.append(data[:,0])
    y_test1.append(data[:,1])


x_test2 = []
y_test2 = []
for i in range(0,40):
    data = np.loadtxt(r'124 LFP Capacity Data\test2\cell{}.csv'.format(i+1), 
                      delimiter=',')
    x_test2.append(data[:,0])
    y_test2.append(data[:,1])


sz = []
for i in range(0,len(x_train)):
    sz.append(len(x_train[i]))
sz = np.vstack(sz)
sz_df = pd.DataFrame(sz)


# Reset the index before saving
test1_conds = test1_conds.reset_index().drop(columns='index')
test2_conds = test2_conds.reset_index().drop(columns='index')
train_conds = train_conds.reset_index().drop(columns='index')


# Combine data for plotting
all_x = x_train + x_test1 + x_test2
all_y = y_train + y_test1 + y_test2





###############################################################################


## Extrapolate the capacity fade curves ##

from scipy.optimize import curve_fit

def linear_model(x, a, b):
    return a*x + b

extrapolated_cap_data_all = []
extrapolated_cycle_data_all = []
# Perform extrapolation on the capacity data so the cells last longer. Typically,
# a much lower EOL threshold is used for grid storage.
for i in range(0, len(all_x)):
    data = all_y[i]
    x = np.arange(1,len(data)+1,1)
    x = x[-30:]
    y = data[-30:]
    
    extrap_len = 3000
    
    parameters, cov = curve_fit(linear_model, x, y)
    a, b, = parameters
    next_x = np.linspace(x[-1:] + 1, x[-1:]+extrap_len, extrap_len)
    next_y = linear_model(next_x, a, b)
    # idxs = np.where(next_y > 0.790)[0]
    # next_y = next_y[idxs]
    
    ext_data = np.concatenate((data.reshape((-1,1)), next_y), axis=0).reshape(-1)
    extrapolated_cap_data_all.append(ext_data)
    extrapolated_cycle_data_all.append(np.arange(2,len(ext_data)+2,1))

# Regroup the data
x_train_ext = extrapolated_cycle_data_all[0:0+41]
x_test1_ext = extrapolated_cycle_data_all[41:41+42]
x_test2_ext = extrapolated_cycle_data_all[41+42:41+42+40]

y_train_ext = extrapolated_cap_data_all[0:0+41]
y_test1_ext = extrapolated_cap_data_all[41:41+42]
y_test2_ext = extrapolated_cap_data_all[41+42:41+42+40]


# Create the save dictionary
my_dic = {'x_train':x_train_ext,
          'x_test1':x_test1_ext,
          'x_test2':x_test2_ext,
          
          'y_train':y_train_ext,
          'y_test1':y_test1_ext,
          'y_test2':y_test2_ext,
          
          'train_conds':train_conds,
          'test1_conds':test1_conds,
          'test2_conds':test2_conds,
    }

output = open(r'capacity_dataset_124_lfp.pkl', 'wb')
pickle.dump(my_dic, output)
output.close()



# Combine data for plotting
all_x = x_train_ext + x_test1_ext + x_test2_ext
all_y = y_train_ext + y_test1_ext + y_test2_ext




###############################################################################




# Plotting preferences
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.default"] = "rm"
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'Times New Roman'


# Determine lifetime of each cell
lifetimes = []
for i in range(0, len(all_x)):
    try:
        lifetimes.append(all_x[i][np.where(all_y[i] < 0.88)[0][0]])
    except:
        lifetimes.append(all_x[i][-1])
lifetimes = np.vstack(lifetimes)

# Sort the cells by cycle life
sort_idxs = np.argsort(lifetimes, axis=0).reshape(-1)
# sort_idxs = np.flip(sort_idxs)

# Create the colors for the lines
xs = np.linspace(0,1,len(lifetimes)+30)
xs = xs[30:]
color_list = matplotlib.cm.Blues(xs)


# Plot the data
plt.figure(num=1, dpi=400, figsize=(4,3.5))
fontsize=12
c = 0
for idx in sort_idxs:
    plt.plot(all_x[idx], all_y[idx], color=color_list[c], linewidth=1)
    c += 1
plt.hlines(0.88,0,2500, colors='k', linestyles='-.')
plt.xlim([0,2500])
plt.ylim([0.4, 1.15])
# plt.title('124 LFP Cells', fontsize=fontsize)
plt.xlabel('Cycle Number', fontsize=fontsize)
plt.ylabel('Discharge Capacity (Ah)', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.savefig(r'Figures\all_124_capacity_fade_curves.pdf', bbox_inches = "tight")
plt.show()









