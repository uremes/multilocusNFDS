import operator
import os
import uuid

import GPy
import numpy as np
import scipy
import scipy.stats as ss

import elfi

SIMULATOR_PATH = './code/simulator/'
SAMPLE_FILE = './data/mass.input'
COGORDERING_FILE = './data/mass.cogOrdering'


def calculate_frequencies(observed, num_clusters, timepoint):
    fs = []
    for obs in observed:
        obs_attime = obs[np.where(obs['Time'] == timepoint)]
        obscount_1 = np.bincount(obs_attime[np.where(obs_attime['VT'] == 1)]['SC'],
                                 minlength=num_clusters)
        obscount_0 = np.bincount(obs_attime[np.where(obs_attime['VT'] == 0)]['SC'],
                                 minlength=num_clusters)
        fs.append(np.concatenate((obscount_1, obscount_0))/len(obs_attime))
    return np.array(fs)


def collate(frequencies):
    frequencies = np.atleast_2d(frequencies)
    num_sequence_clusters = int(len(frequencies[0])/2)
    collated = []
    for obs in frequencies:
        new_obs = np.zeros(4)
        # vaccine type
        inds = np.array([5, 13, 14, 15, 18, 22, 24, 29, 31, 39, 40])
        new_obs[0] = np.sum(obs[inds]) + np.sum(obs[num_sequence_clusters + inds])
        # non-vaccine type
        inds = np.array([2, 4, 8, 10, 11, 12, 16, 17, 19, 21, 28, 30, 33, 38])
        new_obs[1] = np.sum(obs[inds]) + np.sum(obs[num_sequence_clusters + inds])
        # mixed
        inds = np.array([0, 1, 9, 23])
        new_obs[2] = np.sum(obs[inds]) + np.sum(obs[num_sequence_clusters + inds])
        # empty
        inds = np.array([3,  6,  7, 20, 25, 26, 27, 32, 34, 35, 36, 37])
        new_obs[3] = np.sum(obs[inds]) + np.sum(obs[num_sequence_clusters + inds])
        collated.append(new_obs)
    return np.array(collated)


def JSD(sims, observed=None, pi=0.5):
    p = np.squeeze(observed)/np.sum(np.squeeze(observed))
    d = []
    for sim in sims:
        q = np.squeeze(sim)/np.sum(np.squeeze(sim))
        m = pi * p + (1-pi) * q
        d.append(max(pi * ss.entropy(p, m) + (1-pi) * ss.entropy(q, m), np.exp(-20)))
    return np.array(d)


def prepare_inputs_neutral(*inputs, **kwinputs):
    input_list = list(inputs)
    kwinputs['output_filename'] = 'MA.popsim_2_params_' + str(uuid.uuid4())
    kwinputs['input_filename'] = input_list[-1]
    kwinputs['simulator_path'] = input_list[-2]

    for i in range(2):
        input_list[i] = np.exp(input_list[i])

    inputs = tuple(input_list)
    return inputs, kwinputs


def prepare_inputs_ho(*inputs, **kwinputs):
    input_list = list(inputs)
    kwinputs['output_filename'] = 'MA.popsim_3_{}'.format(str(uuid.uuid4()))
    kwinputs['input_filename'] = input_list[-1]
    kwinputs['simulator_path'] = input_list[-2]

    for i in range(3):
        input_list[i] = np.exp(input_list[i])

    inputs = tuple(input_list)
    return inputs, kwinputs


def prepare_inputs_he(*inputs, **kwinputs):
    input_list = list(inputs)
    kwinputs['output_filename'] = 'MA.popsim_5_params_' + str(uuid.uuid4())
    kwinputs['COG_ordering_file'] = input_list[-1]
    kwinputs['input_filename'] = input_list[-2]
    kwinputs['simulator_path'] = input_list[-3]

    for i in range(4):
        input_list[i] = np.exp(input_list[i])

    inputs = tuple(input_list)
    return inputs, kwinputs


def process_result(completed_process, *inputs, **kwinputs):
    # Reads the simulations from the file.
    output_filename = kwinputs['output_filename']
    output_filename += '.sample.out'
    dt = np.dtype([('Time', np.int32), ('VT', np.int32), ('SC', np.int32)])
    simulation = np.genfromtxt(output_filename, dtype=dt, names=True, usecols=(1, 3, 4))

    # Cleans up the output file after reading the data in.
    os.remove(output_filename)

    # This will be passed to ELFI as the result of the command.
    return simulation


def get_sim_vec(sim_params, prepare_inputs, M=0, N=0):
    # compile simulator
    os.system('{}/compile.sh {}'.format(SIMULATOR_PATH, SIMULATOR_PATH))

    # make external command to call the simulator
    sim_call = ('{simulator_path}/freqDepSelect -c CLS02514 -p f -t 1 '
                '-n 100000 -g 72 -l 0.05 -u 0.95 -o {output_filename} -f {input_filename} ')
    sim_call = sim_call + sim_params

    if M > 0 and N > 0:
        sim_call = '{} -M {} -N {}'.format(sim_call, M, N)

    # make vectorised simulator call
    nfds_sim = elfi.tools.external_operation(sim_call, stdout=True, prepare_inputs=prepare_inputs,
                                             process_result=process_result)
    nfds_sim_vector = elfi.tools.vectorize(nfds_sim)
    return nfds_sim_vector


def add_summaries_distance(sim, use_collate=False):
    if use_collate:
        f_36 = elfi.Summary(calculate_frequencies, sim, 41, 36)
        f_72 = elfi.Summary(calculate_frequencies, sim, 41, 72)
        s_36 = elfi.Summary(collate, f_36)
        s_72 = elfi.Summary(collate, f_72)
    else:
        s_36 = elfi.Summary(calculate_frequencies, sim, 41, 36)
        s_72 = elfi.Summary(calculate_frequencies, sim, 41, 72)

    # JSD
    d_36 = elfi.Discrepancy(JSD, s_36)
    d_72 = elfi.Discrepancy(JSD, s_72)
    d_sum = elfi.Operation(operator.add, d_36, d_72)
    # convert to distance
    d_sqrt = elfi.Operation(np.sqrt, d_sum)
    # normalise range
    d_norm = elfi.Operation(operator.add, d_sqrt, -1)


def get_model_neutral(bounds, observed, M=0, N=0, use_collate=False):
    m = elfi.ElfiModel(name='neutral')
    elfi.Prior('uniform', bounds['v'][0], bounds['v'][1]-bounds['v'][0], model=m, name='v')
    elfi.Prior('uniform', bounds['i'][0], bounds['i'][1]-bounds['i'][0], model=m, name='i')

    sim_params = '-v {0} -i {1} -s 0'
    nfds_sim_vector = get_sim_vec(sim_params, prepare_inputs_neutral, M=M, N=N)
    nfds = elfi.Simulator(nfds_sim_vector, m['v'], m['i'],
                          SIMULATOR_PATH, SAMPLE_FILE, observed=observed)
    add_summaries_distance(nfds, use_collate=use_collate)
    return m


def get_model_ho(bounds, observed, M=0, N=0, use_collate=False, use_scale=False):
    m = elfi.ElfiModel(name='NFDS_ho')
    elfi.Prior('uniform', bounds['v'][0], bounds['v'][1]-bounds['v'][0], model=m, name='v')
    elfi.Prior('uniform', bounds['s'][0], bounds['s'][1]-bounds['s'][0], model=m, name='s')
    elfi.Prior('uniform', bounds['i'][0], bounds['i'][1]-bounds['i'][0], model=m, name='i')

    sim_params = '-v {0} -s {1} -i {2}'
    nfds_sim_vector = get_sim_vec(sim_params, prepare_inputs_ho, M=M, N=N)
    nfds = elfi.Simulator(nfds_sim_vector, m['v'], m['s'], m['i'],
                          SIMULATOR_PATH, SAMPLE_FILE, observed=observed)
    add_summaries_distance(nfds, use_collate=use_collate)
    return m


def get_model_he(bounds, observed, M=0, N=0, use_collate=False, use_scale=False):
    m = elfi.ElfiModel(name='NFDS_he')
    elfi.Prior('uniform', bounds['v'][0], bounds['v'][1]-bounds['v'][0], model=m, name='v')
    elfi.Prior('uniform', bounds['s'][0], bounds['s'][1]-bounds['s'][0], model=m, name='s')
    elfi.Prior('uniform', bounds['i'][0], bounds['i'][1]-bounds['i'][0], model=m, name='i')
    scale = elfi.Operation(operator.sub, m['s'], bounds['j'][0], name='scale')
    elfi.Prior('uniform', bounds['j'][0], scale, model=m, name='j')
    elfi.Prior('uniform', bounds['y'][0], bounds['y'][1]-bounds['y'][0], model=m, name='y')

    sim_params = '-v {0} -s {1} -i {2} -j {3} -y {4} -r {COG_ordering_file}'
    nfds_sim_vector = get_sim_vec(sim_params, prepare_inputs_he, M=M, N=N)
    nfds = elfi.Simulator(nfds_sim_vector, m['v'], m['s'], m['i'], m['j'], m['y'],
                          SIMULATOR_PATH, SAMPLE_FILE, COGORDERING_FILE, observed=observed)
    add_summaries_distance(nfds, use_collate=use_collate)
    return m


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
