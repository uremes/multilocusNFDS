import numpy as np
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import elfi
import GPy


def frequency(data, SC, timepoint):
    # Empty array, same size as the number of sequence clusters
    z = np.asarray((np.arange(SC+1), np.zeros(SC+1)), dtype='int32').T
    freq_list = []
    for d_item in data:
        df = np.copy(d_item)
        vt = np.copy(z)
        nvt = np.copy(z)
        # Counts of each vaccine type isolates (VT and NVT)
        unique_vt, counts_vt = np.unique(df[np.where((df['Time'] == timepoint) & (df['VT'] == 1))]['SC'], return_counts=True)
        count_vt = np.asarray((unique_vt, counts_vt)).T
        # Takes zero counts also into account
        for item in count_vt:
            if (item[0] == vt[item[0], 0]):
                vt[item[0], 1] += item[1]
        # Same for NVT types
        unique_nvt, counts_nvt = np.unique(df[np.where((df['Time'] == timepoint) & (df['VT'] == 0))]['SC'], return_counts=True)
        count_nvt = np.asarray((unique_nvt, counts_nvt)).T
        for item in count_nvt:
            if (item[0] == nvt[item[0], 0]):
                nvt[item[0], 1] += item[1]
        # Frequencies of the different vaccinetype isolates
        vt_freq = vt[:, 1] / len(df[np.where(df['Time'] == timepoint)])
        nvt_freq = nvt[:, 1] / len(df[np.where(df['Time'] == timepoint)])

        freq_list.append(np.append(vt_freq, nvt_freq))
    return freq_list


def get_timepoints(data, sum_node=1):
    timepoints = np.unique(data[0]['Time'])
    timepoints.sort()
    if sum_node == 1:
        timepoints = np.delete(timepoints, 0)
    return timepoints


def median_frequencies(data):
    # Will be as long as the number of sequence clusters
    med_freqs = []
    # Goes through each sequence cluster
    for sc in np.arange(len(data[0])):
        # Empty list to get the frequency of single sc from all simulations
        freqs = []
        for simulation in data:
            freqs.append(simulation[sc])
        # Appends the median frequency list with a sc frequency median of the simulations
        med_freqs.append(np.median(freqs))
    return med_freqs


def frequency_plot(data, SC, simulation=1):
    # To increase the figure size
    plt.rcParams['figure.figsize'] = [20, 5]
    # Creating the plot
    timepoints = len(data)
    SC_index = np.arange(SC + 1)*timepoints
    width = 0.75  # the width of the bars (#2.5/len(data))
    l = int((len(data[0])/2))  # used to divide the data into VT and NVT types
    fig, ax = plt.subplots()  # Creates the plot
    label = 0
    pos = SC_index - width*(timepoints/2) + (width/2)  # starting position
    for array in data:
        if (label == 0):
            ax.bar(pos, array[0:l], width, bottom=0, linewidth=1, edgecolor='black', color='tomato', label='Vaccinetype isolates')
            ax.legend()
            ax.bar(pos, array[l:], width, bottom=array[0:l], linewidth=1, edgecolor='black', color='teal', label='Non-vaccinetype isolates')
            ax.legend()
            label += 1
        elif (label != 0):
            ax.bar(pos, array[0:l], width, bottom=0, linewidth=1, edgecolor='black', color='tomato')
            ax.bar(pos, array[l:], width, bottom=array[0:l], linewidth=1, edgecolor='black', color='teal')
        pos += width
    # Adding rest of the labels and titles
    if (simulation == 1):
        ax.set_ylabel('Frequency in simulation')
    elif (simulation == 0):
        ax.set_ylabel('Frequency in sample')
    ax.set_xlabel('Sequence cluster')
    ax.set_title('Frequency of each sequence cluster at three different timepoints, with the vaccination types')
    ax.set_xticks(SC_index)
    ax.set_xticklabels(np.arange(SC+1))
    fig


def create_target_model(m, bounds):
    input_dim = len(m.parameter_names)
    # Create SE-kernel
    kernel_seard = GPy.kern.RBF(input_dim=input_dim, ARD=True)
    kernel_seard.lengthscale.set_prior(GPy.priors.Gamma(2, 2), warning=False)
    target_model = elfi.GPyRegression(m.parameter_names, bounds=bounds, kernel=kernel_seard)
    return target_model


def find_min(model, bounds):
    # this is the best param combination based on what we observed:
    x_opt = model.X[np.argmin(model.Y)]
    # this is the best param combination based on what we observed and the model:
    predict_mean = lambda x: model.predict(np.atleast_2d(x))[0]
    optim = scipy.optimize.minimize(predict_mean, x_opt, bounds=bounds)
    return optim.x, optim.fun


def print_find_min(param_names, xmin, fmin):
    minimum_jsd = np.square(np.exp(fmin))
    print('parameters:', create_string(param_names, '{}'), sep='\t')
    print('min values:', create_string(xmin, '{:06.5f}'), sep='\t')
    print('\nfmin:', fmin, '\nminimum_jsd:', minimum_jsd, sep='\t')


def create_string(string_list, format_type):
    string_var = ''
    for i in np.arange(len(string_list)):
        if i != len(string_list)-1:
            string_var += format_type.format(string_list[i]) + '\t'
        elif i == len(string_list)-1:
            string_var += format_type.format(string_list[i])
    return string_var
