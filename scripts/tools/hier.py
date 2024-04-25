import scipy.stats, pickle, time, os, sys, cProfile, io, pstats, matplotlib

sys.path.insert(0, os.getcwd()+'\\tools')
from tools import inputHandler, diseaseTree, AI, simulator, trialGenerator, plotter
from run_sim import *

matplotlib.use ('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import warnings
warnings.filterwarnings("ignore")

import calculator
import numpy as np
import pandas as pd

ymin = 1e-5
day_to_second = 60 * 60 * 24
hour_to_second = 60 * 60 
minute_to_second = 60  

queuetypes = ['fifo', 'preresume', 'hierarchical']

################################
## Define lambdas
################################
convert_time = lambda time, time0: (time - time0).total_seconds() / hour_to_second



def get_95_ci(a):
    '''Compute 95% confidence interval.
    scipy.stats.norm.interval can also be used instead of scipy.stats.t.interval if counts>30.
    In that case, both give the same result.
    '''
    return scipy.stats.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=scipy.stats.sem(a))

def get_hi_lo_AIs(dis_withAI_hierarchy, ven_hierarchy, disease_in):
    
    '''Given the hierarchy of diseases with AI and the chosen disease,
    return two lists of AI: hi and lo -- these lists will be used to compute equivalent AIs.
    e.g, for chosen disease B and hierarchy A>B>C>D, hi : [A] and lo: [A, B]'''
    
    idx = dis_withAI_hierarchy.index(disease_in)
    hi_AIs = ven_hierarchy[:idx]
    lo_AIs = ven_hierarchy[:idx+1]
    return hi_AIs, lo_AIs

def get_eqvt_se_sp(se_array, sp_array, all_pop_d_prevalence):
    '''Compute equivalent sensitivity and specificity from 
    the lists of Se and Sp corresponding to the AIs to be combined.
    Note: This function is only valid if disease groups for all AIs are independent.'''
    all_pop_nd_prevalence = 1 - all_pop_d_prevalence # array of non-diseased fractions

    eqvt_se = np.sum(se_array * all_pop_d_prevalence) / np.sum(all_pop_d_prevalence)
    eqvt_sp = np.sum(sp_array * all_pop_nd_prevalence) / np.sum(all_pop_nd_prevalence)

    return eqvt_se, eqvt_sp

def get_new_params(new_params): # do only if equivalent AIs and disease probs are needed for >1 AI/disease
    
    '''Compute and return params after creating equivalent AIs from multiple AIs. 
    Equivalent Se, Sp and disease group probabilities are computed.
    Other parameters may be updated just for consistency in access to params[variables].'''
    
    new_diseaseGroups = {}
    new_meanServiceTimes = {}
    
    # Get groups with and without AI. 
    groups_wAI = [new_params['AIinfo'][ii]['groupName'] for ii in new_params['AIinfo'].keys()] # Groups with AI
    groups_noAI = list(set(new_params['diseaseGroups'].keys()) - set(groups_wAI)) # Groups with no AI
    
    # Compute equivalent group prob after combining all unique groups with AI.
    unique_gp_probs = [new_params['diseaseGroups'][ii]['groupProb'] for ii in list(set(groups_wAI))]
    gpEQ_prob = np.sum(np.array(unique_gp_probs))

    disEQ_prob, combined_dis_probs = 0, 0
    
    # 1. Create an equivalent AI. Assumption: Diseases seen by AIs are uncorrelated!!!
    # 2. Create an equivalent group & the corresponding diseased probability. 
#     Assumption: All diseases in a gp are uncorrelated.
    all_se, all_sp, all_pop_prevalence = [], [], []

    for vendor in new_params['AIinfo'].keys():

        all_se.append(new_params['AIinfo'][vendor]['TPFThresh'])
        all_sp.append(1 - new_params['AIinfo'][vendor]['FPFThresh'])

        dis = new_params['AIinfo'][vendor]['targetDisease']
        gp = new_params['AIinfo'][vendor]['groupName']
        # Find disease index in the list of all diseases in a gp.
        dis_idx = new_params['diseaseGroups'][gp]['diseaseNames'].index(dis) 
        # Compute disease prevalence in population
        pop_prev = new_params['diseaseGroups'][gp]['diseaseProbs'][dis_idx] * new_params['diseaseGroups'][gp]['groupProb']
        all_pop_prevalence.append(pop_prev)

    all_se = np.array(all_se)
    all_sp = np.array(all_sp)
    all_pop_prevalence = np.array(all_pop_prevalence)

    EQSe, EQSp = get_eqvt_se_sp(all_se, all_sp, all_pop_prevalence)
    new_params['AIinfo'] = {'Vendor0': {'groupName': 'GroupEQ', 'targetDisease': 'Z', 'TPFThresh': EQSe, 'FPFThresh': 1-EQSp, 'rocFile': None}}
    
    disEQ_prob = np.sum(all_pop_prevalence) / gpEQ_prob # prob of diseased patients in the EQ group. Diseases are assumed to be uncorrelated, hence probs are summed.
    new_diseaseGroups.update({'GroupEQ': {'diseaseNames': ['Z'], 'diseaseProbs': [disEQ_prob], 'groupProb': gpEQ_prob}})

    if groups_noAI:
        for gp in groups_noAI:
            new_diseaseGroups.update({key: new_params['diseaseGroups'][key] for key in new_params['diseaseGroups'] if key == gp})

    new_params['diseaseGroups'] = new_diseaseGroups
    
    # 3. Update the corresponding mean service times.
    # Set the default service time for diseased & non-diseased = the non-diseased reading time in the first group with AI
    # This will change if mu_d != mu_nd.
    for ii in new_params['diseaseGroups'].keys():
        new_meanServiceTimes.update({key: new_params['meanServiceTimes'][key] for key in new_params['meanServiceTimes'] if key == ii})
    
    defaultServiceTime = new_params['meanServiceTimes'][groups_wAI[0]]['non-diseased'] 

    new_meanServiceTimes.update({'interrupting':new_params['meanServiceTimes']['interrupting']})
    new_meanServiceTimes.update({'GroupEQ':{'Z': defaultServiceTime, 'non-diseased': defaultServiceTime}})
    new_params['meanServiceTimes'] = new_meanServiceTimes
    
    return new_params

def update_disease_names(new_params):
    '''This function only updates disease and group names for consistency in names in the theoretical calculations. 
    Note only one AI is expected. This function replaces the group name by 'GroupEQ' and disease name by 'Z'.
    This is called when only one disease group is present in the list of hi_AIs.
    '''

    vendor = list(new_params['AIinfo'].keys())[0] 
    gp = new_params['AIinfo'][vendor]['groupName']
    dis = new_params['AIinfo'][vendor]['targetDisease']

    # 1. Update AIinfo
    new_AIinfo = {'Vendor0': {'groupName': 'GroupEQ', 'targetDisease': 'Z', 'TPFThresh': new_params['AIinfo'][vendor]['TPFThresh'], 'FPFThresh': new_params['AIinfo'][vendor]['FPFThresh'], 'rocFile': new_params['AIinfo'][vendor]['rocFile']}}
    new_params['AIinfo'] = new_AIinfo

    # 2. Update disease group names
    new_diseaseGroup = {'GroupEQ': {'diseaseNames': ['Z'], 'diseaseProbs': [new_params['diseaseGroups'][gp]['diseaseProbs'][0]], 
                                    'groupProb': new_params['diseaseGroups'][gp]['groupProb']}}
    del new_params['diseaseGroups'][gp]
    new_params['diseaseGroups'].update(new_diseaseGroup)

    # 3. Update meanServiceTime
    defaultServiceTime = new_params['meanServiceTimes'][gp]['non-diseased'] # This will change if mu_d != mu_nd.
    new_meanServiceTimes = {'GroupEQ': {'Z': defaultServiceTime, 'non-diseased': defaultServiceTime}}
    del new_params['meanServiceTimes'][gp]
    new_params['meanServiceTimes'].update(new_meanServiceTimes)
    
    return new_params

def get_all_params(config_file_in, keep_ai='all', create_EQgrps = False):
    '''1. Get params from the config file OR
    2. get params after combining multiple AIs (keep_ai is the list of AIs to be combined) OR
    3. get params for a single AI/ disease for use in theoretical calculations.'''
    params_out = inputHandler.read_args(config_file_in, False)
    
    if keep_ai is not 'all':
        num_ais = len(keep_ai)
        new_AIinfo = {} 
        for this_ai in keep_ai:
            new_AIinfo.update({key: params_out['AIinfo'][key] for key in params_out['AIinfo'] if key == this_ai}) 
        params_out['AIinfo'] = new_AIinfo
    
    # For theory, this function may be called to create equivalent groups, either by combining AIs
    # or as a dummy case with a single AI
    if create_EQgrps and num_ais > 1: 
        params_out = get_new_params(params_out)
    elif create_EQgrps and num_ais == 1:
        params_out = update_disease_names(params_out)
        
    # Create an AI object
    AIs_out = {AIname:create_AI (AIname, AIinfo, doPlots=params_out['doPlots'], plotPath=params_out['plotPath'])
           for AIname, AIinfo in params_out['AIinfo'].items()}
    ## Create a disease tree
    aDiseaseTree_out = create_disease_tree (params_out['diseaseGroups'],
                                        params_out['meanServiceTimes'], AIs_out)

    # ## Add additional params
    params_out = inputHandler.add_params (AIs_out, params_out, aDiseaseTree_out)
    return params_out, aDiseaseTree_out, AIs_out

def get_pos_prev(params_in, gp_name=None, dis_idx = None, AI_name = None, single_disease=False):
    
    '''Compute all positive prevalence as a sum of TP and FP.
    This is used to weight the positive patient groups for hi and lo subsets in the theoretical calculations.'''
    # In case of a single disease, use its probabilities directly.
    # Else, use the probablilities from the equivalent diseases/ AIs.
    
    if single_disease:
        tp_prev = params_in['diseaseGroups'][gp_name]['diseaseProbs'][dis_idx] * params_in['diseaseGroups'][gp_name]['groupProb'] * params_in['SeThreshs'][AI_name]
        fp_prev = (1 - params_in['diseaseGroups'][gp_name]['diseaseProbs'][dis_idx]) * params_in['diseaseGroups'][gp_name]['groupProb'] * (1 - params_in['SpThreshs'][AI_name])
    else:
        tp_prev = params_in['diseaseGroups']['GroupEQ']['diseaseProbs'][0] * params_in['diseaseGroups']['GroupEQ']['groupProb'] * params_in['SeThresh']
        fp_prev = (1 - params_in['diseaseGroups']['GroupEQ']['diseaseProbs'][0]) * params_in['diseaseGroups']['GroupEQ']['groupProb'] * (1 - params_in['SpThresh'])
    
    return tp_prev + fp_prev

def wait_time_duration (open_times, close_times, trigger_time):
    '''Compute wait-time durations.'''
    open_times = open_times
    close_times = close_times
    
    waiting = 0
    
    for index, time in enumerate (open_times):
        begin = trigger_time if index == 0 else close_times[index-1]
        waiting += (time - begin).total_seconds() 
    return waiting / 60.

def service_time_duration(open_times, close_times):
    '''Compute service time durations.'''
    service = 0
    
    for index, (open_time, close_time) in enumerate(zip(open_times, close_times)):
        service += (close_time - open_time).total_seconds()
    return service / 60.
        
###########################################################################################################################
parser = argparse.ArgumentParser(description="Description of your program")
parser.add_argument("configFile", help="Path to the configuration file")
args = parser.parse_args()
configFile = args.configFile
params, aDiseaseTree, AIs = get_all_params(configFile)
# Assuming params['diseaseGroups'][jj]['diseaseNames'] is a list
disease_hierarchy = [item for sublist in [params['diseaseGroups'][jj]['diseaseNames'] for jj in list(params['diseaseGroups'].keys())] for item in sublist]
vendor_hierarchy =  list(params['AIinfo'].keys())
print('vendor_hierarchy', vendor_hierarchy)
num_trials = inputHandler.num_trials
write_timelogs = inputHandler.write_timelogs

# Compute other matched inputs. 
diseases_with_AI = [params['AIinfo'][ii]['targetDisease'] for ii in vendor_hierarchy]
AI_group_hierarchy = [params['AIinfo'][ii]['groupName'] for ii in vendor_hierarchy]

matched_group_hierarchy = []

for disease in disease_hierarchy:
    for jj in list(params['diseaseGroups'].keys()):
        if disease in params['diseaseGroups'][jj]['diseaseNames']:
            matched_group_hierarchy.append(jj) 

# The following dictionary is an input to the patient class, which has been updated .
hier_classes_dict = {}
for ii, vendor_name in enumerate(vendor_hierarchy):
    # For hierarchical queuing, disease number starts at 3 (most time-sensitive class).
    hier_classes_dict.update({vendor_name : {'groupName': AI_group_hierarchy[ii], 'disease_num':ii+3}})

# Run trials with all original parameters: simulation of hierarchical queues with no equivalent AIs, 
# pre-resume (equal priority) and fifo queueing.

all_trial_wt = []
all_dfs = []


params['verbose'] = False 
# For now, all logs are off, including hard-coded probability logs.

for ii in np.arange(0, num_trials): # Run trials

    oneSim = simulator.simulator ()
    oneSim.set_params (params)
    oneSim.track_log = False
    oneSim.simulate_queue (AIs, aDiseaseTree, hier_classes_dict)

    thisdf = oneSim._waiting_time_dataframe
    thisdf['trial_num'] = ii
    
    thisdf.reset_index(drop=True, inplace=True)
    all_dfs.append(thisdf)
    
df = pd.concat(all_dfs, axis=0) 

# Compute theoretical wait-times for hierarchical queuing and compare against simulation.

# Get the mean wait-time for the AI negative patients. 
# Combine all AIs, create an equivalent group, get the AI- mean wait-times from preresume for this equivalent group.

params_all, aDiseaseTree_all, AIs_all = get_all_params(configFile, keep_ai = vendor_hierarchy, create_EQgrps = True)
params_all['SeThresh'] = list(params_all['SeThreshs'].values())[0] # To match the expected format.
params_all['SpThresh'] = list(params_all['SpThreshs'].values())[0]
theory_neg = calculator.get_theory_waitTime ('negative', 'preresume', params_all) # mean wait-time all AI-
    

# For each disease in a group (diseases and groups have one-to-one correspondence here),
# get the mean-wait times from simulation and theory. This can include diseases without AI.
print('HIERARCHICAL:')

for chosen_dis, chosen_gp in zip(disease_hierarchy, matched_group_hierarchy):

    # 1. Get the index of the chosen disease within its group. Used later to extract matching probs.
    chosen_dis_idx = params['diseaseGroups'][chosen_gp]['diseaseNames'].index(chosen_dis)
    
    # 2. Extract wait time from simulations, mean and 95% confidence interval.
    all_trial_mean, all_trial_95lo, all_trial_95hi = [], [], []
    
    for trial_idx in np.arange(0, num_trials):
        filtered_df = df.loc[(df['is_diseased'] == True) & (df['disease_name'] == chosen_dis) & (df['trial_num'] == trial_idx)]['hierarchical'].to_numpy()

        trial_mean = np.mean(filtered_df); ci_95 = get_95_ci(filtered_df)
        all_trial_mean.append(trial_mean); all_trial_95lo.append(ci_95[0]); all_trial_95hi.append(ci_95[1])
        
    sim_mean = np.mean(all_trial_mean); sim_95lo = np.mean(all_trial_95lo); sim_95hi = np.mean(all_trial_95hi)
    
   
    if chosen_dis in diseases_with_AI:
        chosen_AI = vendor_hierarchy[diseases_with_AI.index(chosen_dis)]
        
        # 3. Generate AI lists and corresponding parameters for hi and lo cases. 
        keep_HiAIs, keep_LoAIs = get_hi_lo_AIs(diseases_with_AI, vendor_hierarchy, chosen_dis)

        if keep_HiAIs:
            params_hi, aDiseaseTree_hi, AIs_hi = get_all_params(configFile, keep_ai = keep_HiAIs, create_EQgrps = True)
            params_hi['SeThresh'] = list(params_hi['SeThreshs'].values())[0]
            params_hi['SpThresh'] = list(params_hi['SpThreshs'].values())[0]
        else:
            params_hi = None

        params_lo, aDiseaseTree_lo, AIs_lo = get_all_params(configFile, keep_ai = keep_LoAIs, create_EQgrps = True)
        params_lo['SeThresh'] = list(params_lo['SeThreshs'].values())[0]
        params_lo['SpThresh'] = list(params_lo['SpThreshs'].values())[0]

        # 4. Compute wait times from theory for each of the two cases: hi and lo

        if params_hi is not None: # chosen class is not the highest priority class
            # Get wait-times for positive patients in the two cases: hi and lo 
            theory_lo = calculator.get_theory_waitTime ('positive', 'preresume', params_lo)
            theory_hi = calculator.get_theory_waitTime ('positive', 'preresume', params_hi)
            
            # Get the prevalence of positive patients in each case: hi, lo, and chosen disease.
            lo_pos_prev = get_pos_prev(params_lo)
            hi_pos_prev = get_pos_prev(params_hi)
            chosen_dis_pos_prev = get_pos_prev(params, gp_name=chosen_gp, dis_idx = chosen_dis_idx, AI_name = chosen_AI, single_disease=True)
    
            # Compute the mean wait-time for AI+ for chosen disease by taking the weighted difference of means.
            # Mean wait-time for chosen disease = (total wait-time from lo - total wait-time from hi) / num of chosen disease AI+ patients 
            theory_pos = (theory_lo*lo_pos_prev - theory_hi*hi_pos_prev) / chosen_dis_pos_prev

        else: # chosen class is the highest priority class
            theory_pos = calculator.get_theory_waitTime ('positive', 'preresume', params_lo)
        
        # Combine the postive and negative wait times to obtain the diseased wait-time.
        theory_chosen_dis = theory_pos*params['SeThreshs'][chosen_AI] + theory_neg*(1 - params['SeThreshs'][chosen_AI])
        print('diseased', chosen_dis, 'theory, sim: %.2f, %.2f , [%.2f, %.2f]'%(theory_chosen_dis, sim_mean, sim_95lo, sim_95hi))
            
    elif chosen_dis not in diseases_with_AI: 

        # If chosen disease does not have an AI, return the AI negative wait-times.
        print(chosen_dis, 'theory, sim: %.2f, %.2f , [%.2f, %.2f]'%(theory_neg, sim_mean, sim_95lo, sim_95hi))

# Report wait-times for AI negative subgroup. (Optional.)
all_trial_mean, all_trial_95lo, all_trial_95hi = [], [], []

for trial_idx in np.arange(0, num_trials):
    filtered_df = df.loc[(df['is_positive'] == False) & (df['trial_num'] == trial_idx)]['hierarchical'].to_numpy()
    trial_mean = np.mean(filtered_df); ci_95 = get_95_ci(filtered_df)
    all_trial_mean.append(trial_mean); all_trial_95lo.append(ci_95[0]); all_trial_95hi.append(ci_95[1])

sim_mean = np.mean(all_trial_mean); sim_95lo = np.mean(all_trial_95lo); sim_95hi = np.mean(all_trial_95hi)

print('AI negative groups:', 'theory, sim: %.2f, %.2f , [%.2f, %.2f]'%(theory_neg, sim_mean, sim_95lo, sim_95hi))


# Get FIFO results for each diseased subgroup from simulation and theory. 
# (This function is somewhat redundant. The two "for loops" below could be inluded in the code block above.) 
print('FIFO:')

theory_fifo = calculator.get_theory_waitTime ('positive', 'fifo', params) # mean wait-time all AI-
print(theory_fifo)

for chosen_dis in disease_hierarchy:
    all_trial_mean, all_trial_95lo, all_trial_95hi = [], [], []
    
    for trial_idx in np.arange(0, num_trials):
        filtered_df = df.loc[(df['is_diseased'] == True) & (df['disease_name'] == chosen_dis) & (df['trial_num'] == trial_idx)]['fifo'].to_numpy()
        trial_mean = np.mean(filtered_df); ci_95 = get_95_ci(filtered_df)
        all_trial_mean.append(trial_mean); all_trial_95lo.append(ci_95[0]); all_trial_95hi.append(ci_95[1])
        
    sim_mean = np.mean(all_trial_mean); sim_95lo = np.mean(all_trial_95lo); sim_95hi = np.mean(all_trial_95hi)
    print(chosen_dis, 'theory, sim: %.2f, %.2f , [%.2f, %.2f]'%(theory_fifo, sim_mean, sim_95lo, sim_95hi))
    

# Get FIFO results for the non-diseased subgroup from simulation and theory.
all_trial_mean, all_trial_95lo, all_trial_95hi = [], [], []

for trial_idx in np.arange(0, num_trials):
    filtered_df = df.loc[(df['is_positive'] == False) & (df['trial_num'] == trial_idx)]['fifo'].to_numpy()
    trial_mean = np.mean(filtered_df); ci_95 = get_95_ci(filtered_df)
    all_trial_mean.append(trial_mean); all_trial_95lo.append(ci_95[0]); all_trial_95hi.append(ci_95[1])

sim_mean = np.mean(all_trial_mean); sim_95lo = np.mean(all_trial_95lo); sim_95hi = np.mean(all_trial_95hi)
print('AI negatives', 'theory, sim: %.2f, %.2f , [%.2f, %.2f]'%(theory_fifo, sim_mean, sim_95lo, sim_95hi))


# Save time logs for each quueing discipline.
# This can be updated to write out for all trials.
# Currently, only writes the last trial log.

if write_timelogs:
    for ii in queuetypes:
        timelog = oneSim._patients_seen[ii]
        timelog_df = pd.DataFrame(timelog.values())
        hlog.to_excel('..outputs/timelog.xlsx')


# Test that the mean wait-time is the same for all queuing disciplines in M/M/1 queue with equal mean service times.
# Useful when implementing a new queuing discipline.

df_hr = df['hierarchical'].to_numpy()
df_pr = df['preresume'].to_numpy()
df_fifo = df['fifo'].to_numpy()

df_hr_mean = np.mean(df_hr)
df_95_ci_hr = get_95_ci(df_hr)
df_pr_mean = np.mean(df_pr)
df_95_ci_pr = get_95_ci(df_pr)
df_fifo_mean = np.mean(df_fifo)
df_95_ci_fifo = get_95_ci(df_fifo)

df_theory = calculator.get_theory_waitTime('positive','fifo',params)

# print(np.mean(df_hr), get_95_ci(df_hr))
# print(np.mean(df_pr), get_95_ci(df_pr))
# print(np.mean(df_fifo), get_95_ci(df_fifo), calculator.get_theory_waitTime ('positive', 'fifo', params))

# All values should be equal.