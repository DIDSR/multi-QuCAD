
##
## By Elim Thompson (11/27/2020)
##
## This script contains all theoretical calculations to predict the state
## probability of the number of patients in system (observed by every
## in-coming patient). For more information, please visit
## 
## 1. Osogami et al (2005) Closed form solutions for mapping general
##    distributions to quasi-minimal PH distributions 
##    https://www.cs.cmu.edu/~harchol/Papers/quasi-minimal-PH.pdf
## 2. Harchol-Balter et al Multi-server queueing systems with multiple
##    priority classes
##    https://www.cs.cmu.edu/~harchol/Papers/questa.pdf
## 3. Osogami's thesis (2005) Analysis of Multi-server Systems via
##    Dimensionality Reduction of Markov Chains
##    http://reports-archive.adm.cs.cmu.edu/anon/2005/CMU-CS-05-136.pdf
##
##
## 05/23/2024
## ----------
##  * added hierarchical calculation from hier.py to here
##  * What doesn't work?
##      Hierarchical:
##       - More than 1 AI in a group
##       - Presence of interrupting patients
##       - mud != mund ????
##      Preresume:
##       - mud != mund ????
###########################################################################

################################
## Import packages
################################ 
import numpy, scipy, math
from numpy.linalg import matrix_power
from scipy.linalg import lu, eig, inv
from copy import deepcopy

#######################################################
## Common calculator to be used to get mean wait-time
#######################################################
def get_theory_waitTime (params, aHierarchy):

    ''' Function to calculate waiting time for fifo, priority, hierarchical-priority.
        Mean wait-time for all diseases will be calculated, and no mean-wait time for
        non-diseased cases.
    
        For fifo, there are 2 priority classes, interrupting and non-interrupting (number
        of AIs don't matter because it is the without-AI arm).
        
        For preresume, the mean wait-time of a disease of interests depends on 3 scenarios:
          I. the disease of interests has an AI to identify it
             (may have other AIs in the same group)
         II. the disease of interests does not have an AI to identify it, but there are other
             AIs in the same group i.e. any flagged cases are FP cases of other AIs.
        III. the disease of interests belong to a group that does not have any AIs.
        See get_all_waitTime_preresume ().

        For hierarchical-preresume queue, the calculation is done in hiararchy class. Note that
        the hierarchical-preresume approach wouldn't work if
            * multiple diseased in a group (e.g. A and B) but there are more than 1 AI.

        inputs
        ------
        params (dict): dictionary with user inputs
        aHierarchy (hierarchy): hierarchy of ranked priority classes

        outputs
        -------
        theories (dict): theorectical predictions for all priority and disease classes.
    '''

    theories = {}

    ## For fifo, only interrupting and non-interrupting
    # theories['fifo'] = {}
    # for aclass in ['interrupting', 'non-interrupting']:
    #     theories['fifo'][aclass] = get_theory_waitTime_fifo_preresume (aclass, 'fifo', params)
    
    ## For priority, there are interrupting, positive, and negative
    #theories['priority'] = get_all_waitTime_preresume (params, aHierarchy) if params['isPreemptive'] else \
                           #get_all_waitTime_priority_nonpreemptive (params, aHierarchy)
    
    ## For hierarchy, no interrupting class at all
    theories['hierarchical'] = aHierarchy.predict_mean_wait_time (params) if params['isPreemptive'] else \
                               get_all_waitTime_hierarchical_nonpreemptive (params, aHierarchy)

    return theories

def get_all_waitTime_preresume (params, aHierarchy):

    ''' Function to calculate waiting time for preresume priority. Mean wait-time for all
        diseases will be calculated, and no mean-wait time for non-diseased cases.

        inputs
        ------
        params (dict): dictionary with user inputs
        aHierarchy (hierarchy): hierarchy of ranked priority classes

        outputs
        -------
        theories (dict): theorectical predictions for all priority and disease classes.
    '''

    priority = {}
    for aclass in ['interrupting', 'positive', 'negative']:
        priority[aclass] = get_theory_waitTime_fifo_preresume (aclass, 'preresume', params)
                            
    ## Now, calculate the wait-time for each diseased cases
    for disease, group in zip (aHierarchy.diseaseNames, aHierarchy.groupNames):
        posWeight, negWeight = aHierarchy.get_posNegWeights (params, disease, group)
        priority[disease] = priority['positive']*posWeight + priority['negative']*negWeight

    return priority

def get_theory_waitTime_fifo_preresume (aclass, variable, params):

    ''' Function to obtain theoretical waiting time and wait-time difference.
        If input number of radiologist is greater than 2, average waiting time
        for AI negative subgroup cannot be calculated (hence, waiting time
        and wait-time difference for radiologist diagnosis diseased and
        non-diseased subgroups). The theoretical values will not be shown,
        but the simulation results will be available.

        inputs
        ------
        aclass (str): patient subgroup. Either interrupting, non-interrupting,
                      diseased, non-diseased, positive, or negative.
        variable (str): what is being outputted? Either "fifo" for patient
                        waiting time without CADt scenario, "preresume" for
                        patient waiting time with CADt scenario, or "delta"
                        for wait-time difference between the two scenarios.
        params (dict): dictionary capsulating all simulation parameters

        output
        ------
        wait_time (float): theoretical wait time 
    '''

    if aclass == 'interrupting' and variable == 'delta':
        return 0
    if aclass == 'interrupting':
        _, state_prob = get_theory_qlength_fifo_preresume (aclass, variable, params)
        interrupting_predL = numpy.sum ([i*p for i, p in enumerate (state_prob)]).real
        interrupting_predW = interrupting_predL / params['lambdas']['interrupting']
        interrupting_predW = interrupting_predW - 1/params['mus']['interrupting']
        return interrupting_predW

    ## For without-CADt, lower class = non-interrupting
    _, state_prob = get_theory_qlength_fifo_preresume (aclass, variable, params)
    nonInterrupting_predL = numpy.sum ([i*p for i, p in enumerate (state_prob)]).real
    nonInterrupting_predW = nonInterrupting_predL / params['lambdas']['non-interrupting']
    nonInterrupting_predW = nonInterrupting_predW - 1/params['mus']['non-interrupting']
    if aclass == 'non-interrupting': return nonInterrupting_predW
    if aclass in ['diseased', 'non-diseased'] and variable == 'fifo': return nonInterrupting_predW
    if aclass in ['positive', 'negative'] and variable == 'fifo': return nonInterrupting_predW

    ## For with-CADt, non-interrupting class becomes positive and negative
    _, state_prob = get_theory_qlength_fifo_preresume (aclass, variable, params)
    positive_predL = numpy.sum ([i*p for i, p in enumerate (state_prob)]).real
    positive_predW = positive_predL / params['lambdas']['positive']
    positive_predW = positive_predW - 1/params['mus']['positive']
    if aclass == 'positive' and variable == 'preresume': return positive_predW
    if aclass == 'positive' and variable == 'delta': return positive_predW - nonInterrupting_predW

    ## If input number of radiologists is more than 2 *and* fractionED is non-zero, cannot
    ## calculate theoretical values for negative subgroup, hence, diseased and non-diseased subgroups.
    if params['nRadiologists'] > 2 and params['fractionED'] >  0:
        print ('WARN: Cannot calculate theoretical values for AI negative subgroup when more than 2 radiologists with presence of interrupting patient cases.')
        return None

    ## Negative
    _, state_prob = get_theory_qlength_fifo_preresume (aclass, variable, params)
    negative_predL = numpy.sum ([i*p for i, p in enumerate (state_prob)]).real
    negative_predW = negative_predL / params['lambdas']['negative']
    negative_predW = negative_predW - 1/params['mus']['negative']
    if aclass == 'negative' and variable == 'preresume': return negative_predW
    if aclass == 'negative' and variable == 'delta': return negative_predW - nonInterrupting_predW

    ## Diagnosis diseased
    diseased_predW = positive_predW*params['SeThresh'] + negative_predW*(1-params['SeThresh'])
    if aclass == 'diseased' and variable == 'preresume': return diseased_predW
    if aclass == 'diseased' and variable == 'delta': return diseased_predW - nonInterrupting_predW    

    ## Diagnosis non-diseased
    nondiseased_predW = positive_predW*(1-params['SpThresh']) + negative_predW*params['SpThresh']
    if aclass == 'non-diseased' and variable == 'preresume': return nondiseased_predW
    if aclass == 'non-diseased' and variable == 'delta': return nondiseased_predW - nonInterrupting_predW   

    print ('Should not land here!')

def get_theory_qlength_fifo_preresume (aclass, qtype, params):

    ''' Function to obtain theoretical queue length i.e. state probability.
        If input number of radiologist is greater than 2, average waiting time
        for AI negative subgroup cannot be calculated (hence, waiting time
        and wait-time difference for radiologist diagnosis diseased and
        non-diseased subgroups). The theoretical values will not be shown,
        but the simulation results will be available.
        
        For preemptive, state is defined by number of patients in the system
        i.e. `qlength` is the number of patients in the system. For non-
        preemptive, state may be defined by number of patients in the queue
        and so, `qlength` would be the number of patients in the queue.

        inputs
        ------
        aclass (str): patient subgroup. Either interrupting, non-interrupting,
                      positive, or negative.
        qtype (str): what is being outputted? Either "fifo" for patient
                     waiting time without CADt scenario, "preresume" for
                     patient waiting time with CADt scenario.
        params (dict): dictionary capsulating all simulation parameters

        outputs
        -------
        qlength (array): number of patient in the queueing system
        state_prob (array): state probabilities of the queueing system
    '''

    qlength = numpy.linspace (0, 1000, 1001)

    if aclass == 'interrupting':
        state_prob = get_state_pdf_MMn (qlength, params['nRadiologists'],
                                        params['lambdas']['interrupting'],
                                        params['mus']['interrupting'])
        return qlength, state_prob

    ## For without-CADt, lower class = non-interrupting
    nonInterrupting_cal = def_cal_MMs (params, lowPriority='non-interrupting',
                                       doDynamic=params['nRadiologists']>3)
    state_prob = nonInterrupting_cal.solve_prob_distributions (len (qlength))
    state_prob = numpy.array ([numpy.sum (p) for p in state_prob])
    if aclass == 'non-interrupting': return qlength, state_prob
    if aclass in ['positive', 'negative'] and qtype == 'fifo': return qlength, state_prob

    ## For with-CADt, non-interrupting class becomes positive and negative
    positive_cal = def_cal_MMs (params, lowPriority='positive',
                                doDynamic=params['nRadiologists']>3)
    state_prob = positive_cal.solve_prob_distributions (len (qlength))
    state_prob = numpy.array ([numpy.sum (p) for p in state_prob])
    if aclass == 'positive' and qtype == 'preresume': return qlength, state_prob

    ## If input number of radiologists is more than 2 *and* fractionED is non-zero, cannot
    ## calculate theoretical values for negative subgroup, hence, diseased and non-diseased subgroups.
    if params['nRadiologists'] > 2 and params['fractionED'] >  0:
        print ('WARN: Cannot calculate theoretical values for AI negative subgroup when more than 2 radiologists with presence of interrupting patient cases.')
        return None, None

    ## Negative
    negative_cal = get_cal_lowest (params) if params['fractionED'] > 0.0 else \
                   def_cal_MMs (params, lowPriority='negative', highPriority='positive',
                                doDynamic=params['nRadiologists']>3)
    state_prob = negative_cal.solve_prob_distributions (len (qlength))
    state_prob = numpy.array ([numpy.sum (p).real for p in state_prob])
    if aclass == 'negative' and qtype == 'preresume': return qlength, state_prob

    return None, None

# Commenting out this version in case we need to return to it. It works as long as there are not multiple diseases per group.

def get_all_waitTime_hierarchical_nonpreemptive (params, aHierarchy):

    ''' Function to calculate wait-time for hierarchical-NP queue introduced
        by Michelle based on Eq 21 in "Effects of the Queue Discipline on
        System Performance" by Raicu et al 2023:
            https://doi.org/10.3390/appliedmath3010003
        
                                              sum_i lambda'_i x c_i
            mean wait-time = -------------------------------------------------------------
              for class i     2 x (1 - sum_j lamda'_j x b_j) x (1 - sum_j lamda'_j x b_j)
        
        The numerator is based on Eq 20 of the papr. i goes from 1 to k
        priority classes. c_i is the 2nd moment of service time distribution.
        In the denominator, j in the first summation goes from 1 to i - 1.
        In the second summarion, j goes from 1 to i. i = 1 has the highest
        priority. b_j is the is the 1st moment (i.e. mean) service rate of
        class j. Lambda'_j is the individual arrival rate for the j-th class.

        In our case, AI+ patients in every group with AI form a priority class
        ranked by the disease rank. For each group with AI, we need to collect
        the first and second moments of service time distribution for the
        positive patients by that AI. Because it includes both diseased and
        non-diseased, the service time distribution is a hyperexponential
        distribution with probabilities of PPV (or 1-NPV) and 1-PPV (or NPV)
        to take into account the FP and FN as well. 

        This method can calculate the mean wait-time of each positive subgroup
        within each group with AI. It is currently limited to scenarios where
        each group has only 1 disease and at most 1 AI.

        inputs
        ------
        params (dict): dictionary with user inputs
        aHierarchy (hierarchy): hierarchy of ranked priority classes

        outputs
        -------
        theories (dict): theorectical predictions for all priority and disease classes.
    '''
    ## Initialize holders: these values are ordered by disease rank
    #  To store the first and second moments per priority class.
    neg_means, neg_2nd_moment = [], [] 
    pos_means, pos_2nd_moment = [], []
    #  To store the positive rate per priority class w.r.t. whole population
    prob_pos_groups = []
    #  To store individual arrival rate of positive cases by the 1 AI in the group
    arrival_rates = [] 
    #  To store the fraction of target disease (in this priority class)
    #  within the overall AI negative class
    diseased_wait_times = []

    ## Loop through all groups, get info for each AI-positive subgroup
    arr = aHierarchy.groupNames
    _, idx = numpy.unique(arr, return_index=True)
    unique_groupNames_array = arr[numpy.sort(idx)]
    for i in range(len(unique_groupNames_array)):
        groupname = unique_groupNames_array[i]
        # get read times for all diseases in group (excluding non-diseased)
        data_dict_readtimes = params['meanServiceTimes'][groupname]
        keys = list(data_dict_readtimes.keys())[:-1]
        # Extract the corresponding values
        dis_readtimes = [data_dict_readtimes[key] for key in keys]
        dis_readtimes = numpy.array(dis_readtimes)
        # get non-diseased read time
        nondis_readtime = params['meanServiceTimes'][groupname]['non-diseased']
        ## Get the second moments
        ##   E[X^2] = sum_i (pi * 2 / ratei**2) = sum_i (pi * 2 * timei**2), with i = diseased vs non-diseased
        dis_2nd_moment_factors = [2 * dis_readtime**2 for dis_readtime in dis_readtimes]
        nondis_2nd_moment_factor = 2 * nondis_readtime**2
        groupProb = params['diseaseGroups'][groupname]['groupProb']

        for diseasename in params['diseaseGroups'][groupname]['diseaseNames']: #fix AI a_i (if none corresponding to disease, Sp = 1, Se = 0)
            ## probdis_givenpos_array below accesses the array of probabilities P(diseased b_j | a_i + subgroup), looping through j, for fixed AI a_i.
            probdis_givenpos_dict = params['prob_thisdis_given_AI_pos'][groupname][diseasename]
            keys = list(probdis_givenpos_dict.keys())[:-1]
            probdis_givenpos_array = [probdis_givenpos_dict[key] for key in keys]
            probdis_givenpos_array = numpy.array(probdis_givenpos_array)
            # probnondis_givenpos = 1 - sum_{diseases j in group i} P(diseased b_j | a_i + subgroup) = P(nondis | a_i + subgroup)
            probnondis_givenpos = 1-sum(probdis_givenpos_array)
            # append to effective mean for AI-i + group: P(diseased b_j | a_i + subgroup) * (readtime b_j) + P(non-dis | a_i + subgroup) * (nondis readtime)
            pos_means.append(sum(dis_readtimes * probdis_givenpos_array) + nondis_readtime * probnondis_givenpos)
            # get affective second moment
            pos_2nd_moment.append(sum(dis_2nd_moment_factors * probdis_givenpos_array) + nondis_2nd_moment_factor * probnondis_givenpos)
            ## Get probability of AI a_i positive across whole population
            prob_pos_group = params['prob_pos_i_neg_higher_AIs'][groupname][diseasename]
            prob_pos_groups.append(prob_pos_group)
            print('diseaseName, prob_pos_group', diseasename, prob_pos_group)
            ## Get arrival rate for patients with this specific disease 
            arrival_rate = prob_pos_group * params['arrivalRates']['non-interrupting']
            arrival_rates.append(arrival_rate)

        ## probdis_givenneg accesses the array of probabilities P(diseased b_j | a- subgroup), looping through j
        probdis_givenneg_dict = params['prob_thisdis_given_AI_neg'][groupname] 
        keys = list(probdis_givenneg_dict.keys())[:]
        probdis_givenneg_array = [probdis_givenneg_dict[key] for key in keys]
        probdis_givenneg_array = numpy.array(probdis_givenneg_array)
        ## prob nondis_thisgroup_givenneg = P(non-dis, current group | a- subgroup)
        probnondis_thisgroup_givenneg = groupProb*(1-sum(probdis_givenneg_array)/groupProb)
        # below we append P(dis b_j | a- subgroup) * (readtime b_j) + P(non-dis, current group | a- subgroup) * nondis_readtime to the AI-negative means.
        # after looping through all the groups, we should be able to sum the entries of neg_means to get the effective read time of the entire AI-negative subgroup.
        neg_means.append(sum(dis_readtimes * probdis_givenneg_array) + nondis_readtime * probnondis_thisgroup_givenneg)
        neg_2nd_moment.append(sum(dis_2nd_moment_factors * probdis_givenneg_array) + nondis_2nd_moment_factor * probnondis_thisgroup_givenneg)

    ## AI_neg_mean = effective read time [min] for AI-neg group that may have different disease types
    AI_neg_mean = sum(neg_means)
    AI_neg_2nd_moment  = sum(neg_2nd_moment)
    AI_neg_arrival_rate = params['arrivalRates']['negative']

    ## Calculation of wait-time
    bs = numpy.array (pos_means + [AI_neg_mean])
    cs = numpy.array (pos_2nd_moment + [AI_neg_2nd_moment])
    lambdas = numpy.array (arrival_rates + [AI_neg_arrival_rate])
    #  Get c (Eq 20) 
    c = sum (lambdas * cs)
    ## Apply Eq 21. For the highest priority, only the second summation in the denominator counts.
    ## These are the mean wait-time of each positive subgroup by each AI in each group. For group
    ## that don't have an AI, it spit out a mean wait-time all cases are at the end of a sub-system
    ## with only higher rank AI-positive diseases in front (as if other lower groups do not exist).
    wait_times = []
    for j in range(len(lambdas)):
        wait_time = c / (2 * (1 - sum(lambdas[:j+1] * bs[:j+1]))) ## second summation
        if j > 0: wait_time /= 1 - sum(lambdas[:j] * bs[:j])
        wait_times.append (wait_time)
    wait_times = numpy.array (wait_times)

    ## Converts the AI+/- wait time into per-disease wait-time. 
    #probs_for_conversion[j] accesses an array of probabilities P(a_i + subgroup | diseased b_j) for all AIs a_i in the 
    for j in range(len(wait_times)-1):
        diseaseName = aHierarchy.diseaseNames[j]
        groupname = aHierarchy.groupNames[j]
        disease_in_group_idx = []
        for disease_in_group in params['diseaseGroups'][groupname]['diseaseNames']:
            index = numpy.where(aHierarchy.diseaseNames == disease_in_group)[0][0]
            disease_in_group_idx.append(index)
        probs_for_conversion = params['probs_for_waittime_conversion']
        probs_for_conversion_thisdis = probs_for_conversion[diseaseName]
        diseased_wait_times.append(sum(wait_times[disease_in_group_idx] * probs_for_conversion_thisdis) + wait_times[-1] * (1-sum(probs_for_conversion_thisdis)))
    
    ## Populate results into a dictionary
    priority = {}
    for index, (disease, vendor, wait_time) in enumerate (zip (aHierarchy.diseaseNames, aHierarchy.AINames, diseased_wait_times)):
        priority[disease] = {'diseased':wait_time}
        priority[disease]['negative'] = wait_times[-1]
        if vendor is None: continue
        priority[disease]['positive'] = wait_times[index]
        
    return priority


# Commenting out this version in case we need to return to it. It works as long as there are not multiple diseases per group.

# def get_all_waitTime_hierarchical_nonpreemptive (params, aHierarchy):

#     ''' Function to calculate wait-time for hierarchical-NP queue introduced
#         by Michelle based on Eq 21 in "Effects of the Queue Discipline on
#         System Performance" by Raicu et al 2023:
#             https://doi.org/10.3390/appliedmath3010003
        
#                                               sum_i lambda'_i x c_i
#             mean wait-time = -------------------------------------------------------------
#               for class i     2 x (1 - sum_j lamda'_j x b_j) x (1 - sum_j lamda'_j x b_j)
        
#         The numerator is based on Eq 20 of the papr. i goes from 1 to k
#         priority classes. c_i is the 2nd moment of service time distribution.
#         In the denominator, j in the first summation goes from 1 to i - 1.
#         In the second summarion, j goes from 1 to i. i = 1 has the highest
#         priority. b_j is the is the 1st moment (i.e. mean) service rate of
#         class j. Lambda'_j is the individual arrival rate for the j-th class.

#         In our case, AI+ patients in every group with AI form a priority class
#         ranked by the disease rank. For each group with AI, we need to collect
#         the first and second moments of service time distribution for the
#         positive patients by that AI. Because it includes both diseased and
#         non-diseased, the service time distribution is a hyperexponential
#         distribution with probabilities of PPV (or 1-NPV) and 1-PPV (or NPV)
#         to take into account the FP and FN as well. 

#         This method can calculate the mean wait-time of each positive subgroup
#         within each group with AI. It is currently limited to scenarios where
#         each group has only 1 disease and at most 1 AI.

#         inputs
#         ------
#         params (dict): dictionary with user inputs
#         aHierarchy (hierarchy): hierarchy of ranked priority classes

#         outputs
#         -------
#         theories (dict): theorectical predictions for all priority and disease classes.
#     '''
    
#     ## Initialize holders: these values are ordered by disease rank
#     Ses, Sps, gp_probs = [], [], []    
#     #  To store the first and second moments per priority class.
#     neg_means, neg_2nd_moment = [], [] 
#     pos_means, pos_2nd_moment = [], []
#     #  To store the positive rate per priority class w.r.t. whole population
#     prob_pos_groups = []
#     #  To store individual arrival rate of positive cases by the 1 AI in the group
#     arrival_rates = [] 
#     #  To store the fraction of target disease (in this priority class)
#     #  within the overall AI negative class
#     prob_thisdis_given_negs = []

#     ## Loop through all groups, get info for each AI-positive subgroup
#     for i in range(len(aHierarchy.groupNames)):
#         groupname = aHierarchy.groupNames[i]
#         diseasename = aHierarchy.diseaseNames[i]
#         vendorname = aHierarchy.AINames[i]
#         ## Get disease prob within the group and group probability
#         dis_idx = params['diseaseGroups'][groupname]['diseaseNames'].index (diseasename)
#         dis_prob = params['diseaseGroups'][groupname]['diseaseProbs'][dis_idx]
#         gp_prob = params['diseaseGroups'][groupname]['groupProb']
#         gp_probs.append(gp_prob)
#         ## Get AI vendor performance. If no vendor in this group,
#         ## Se/Sp = 0, 1 - all cases in this group is counted as AI-.
#         Se, Sp = 0, 1
#         probdis_givenpos, probdis_givenneg = 0, dis_prob
#         probnondis_givenpos, probnondis_givenneg = 0, 1-dis_prob
#         if vendorname is not None:
#             Se = params['SeThreshs'][vendorname] 
#             Sp = params['SpThreshs'][vendorname] 
#             probdis_givenpos = params['probs_ppv_npv']['ppv'][groupname][vendorname]
#             probdis_givenneg = 1 - params['probs_ppv_npv']['npv'][groupname][vendorname]
#             probnondis_givenpos = 1 - params['probs_ppv_npv']['ppv'][groupname][vendorname]
#             probnondis_givenneg = params['probs_ppv_npv']['npv'][groupname][vendorname]            
#         Ses.append(Se)
#         Sps.append(Sp)
#         ## Get effective reading time for positive and negative patients
#         ##        1      PPV     1-PPV
#         ##  e.g. ---- = ----- + -------
#         ##        u+     uD       uND
#         dis_readtime = params['meanServiceTimes'][groupname][diseasename]
#         nondis_readtime = params['meanServiceTimes'][groupname]['non-diseased']
#         pos_means.append(dis_readtime * probdis_givenpos + nondis_readtime * probnondis_givenpos) 
#         neg_means.append(dis_readtime * probdis_givenneg + nondis_readtime * probnondis_givenneg)
#         ## Get the second moments
#         ##   E[X^2] = sum_i (pi * 2 / ratei**2) = sum_i (pi * 2 * timei**2), with i = diseased vs non-diseased
#         dis_2nd_moment_factor = 2 * dis_readtime**2
#         nondis_2nd_moment_factor = 2 * nondis_readtime**2
#         pos_2nd_moment.append(dis_2nd_moment_factor * probdis_givenpos + nondis_2nd_moment_factor * probnondis_givenpos)
#         neg_2nd_moment.append(dis_2nd_moment_factor * probdis_givenneg + nondis_2nd_moment_factor * probnondis_givenneg)
#         ## Get probability of positive across whole population
#         prob_pos_group = (Se * dis_prob + (1 - Sp) * (1 - dis_prob)) * gp_prob
#         prob_pos_groups.append(prob_pos_group)
#         ## Get arrival rate for patients with this specific disease 
#         arrival_rate = prob_pos_group * params['arrivalRates']['non-interrupting']
#         arrival_rates.append(arrival_rate)
#         ## Among all AI-negative, what is the fraction of each disease 
#         prob_thisdis_given_neg = gp_prob * (Sp * (1 - dis_prob) + (1 - Se) * dis_prob)
#         prob_thisdis_given_negs.append(prob_thisdis_given_neg)

#     ## For the AI-negative group (i.e. cases flagged as negative by all AIs).
#     ## Make use of neg_means, neg_2nd_moment, and prob_thisdis_given_negs
#     prob_thisdis_given_negs = numpy.array(prob_thisdis_given_negs) / (1 - sum(prob_pos_groups)) 
#     ## AI_neg_means = effective read time [min] for AI-neg group that may have different disease types
#     AI_neg_mean = sum(prob_thisdis_given_negs * numpy.array(neg_means))
#     AI_neg_2nd_moment  = sum(prob_thisdis_given_negs * numpy.array(neg_2nd_moment))
#     AI_neg_arrival_rate = params['arrivalRates']['negative']

#     ## Calculation of wait-time
#     bs = numpy.array (pos_means + [AI_neg_mean])
#     cs = numpy.array (pos_2nd_moment + [AI_neg_2nd_moment])
#     lambdas = numpy.array (arrival_rates + [AI_neg_arrival_rate])
#     #  Get c (Eq 20) 
#     c = sum (lambdas * cs)
#     ## Apply Eq 21. For the highest priority, only the second summation in the denominator counts.
#     ## These are the mean wait-time of each positive subgroup by each AI in each group. For group
#     ## that don't have an AI, it spit out a mean wait-time all cases are at the end of a sub-system
#     ## with only higher rank AI-positive diseases in front (as if other lower groups do not exist).
#     wait_times = []
#     for j in range(len(lambdas)):
#         wait_time = c / (2 * (1 - sum(lambdas[:j+1] * bs[:j+1]))) ## second summation
#         if j > 0: wait_time /= 1 - sum(lambdas[:j] * bs[:j])
#         wait_times.append (wait_time)
#     wait_times = numpy.array (wait_times)

#     ## Converts the AI+/- wait time into per-disease wait-time. 
#     Ses, Sps, gp_probs = numpy.array (Ses), numpy.array (Sps), numpy.array (gp_probs)
#     diseased_wait_times = wait_times[:-1] * Ses + wait_times[-1] * (1 - Ses)
    
#     ## Populate results into a dictionary
#     priority = {}
#     for index, (disease, vendor, wait_time) in enumerate (zip (aHierarchy.diseaseNames, aHierarchy.AINames, diseased_wait_times)):
#         priority[disease] = {'diseased':wait_time}
#         priority[disease]['negative'] = wait_times[-1]
#         if vendor is None: continue
#         priority[disease]['positive'] = wait_times[index]
        
#     return priority

# fills in AIs that exist in group for diseases that do not have an AI
# def fill_missing_vendors(groupnames, diseasenames, vendornames):
#     group_to_vendor = {}
#     vendor_flag = 0

#     for group, vendor in zip(groupnames, vendornames):
#         if vendor is not None:
#             group_to_vendor[group] = vendor

#     filled_vendornames = []
#     for group, vendor in zip(groupnames, vendornames):
#         if vendor is None:
#             vendor_flag = 1
#             filled_vendornames.append(group_to_vendor.get(group, None))
#         else:
#             filled_vendornames.append(vendor)
    
#     return vendor_flag, filled_vendornames


######  version that should work for multiple diseases per group
# def get_all_waitTime_hierarchical_nonpreemptive (params, aHierarchy):

#     ''' Function to calculate wait-time for hierarchical-NP queue introduced
#         by Michelle based on Eq 21 in "Effects of the Queue Discipline on
#         System Performance" by Raicu et al 2023:
#             https://doi.org/10.3390/appliedmath3010003
        
#                                               sum_i lambda'_i x c_i
#             mean wait-time = -------------------------------------------------------------
#               for class i     2 x (1 - sum_j lamda'_j x b_j) x (1 - sum_j lamda'_j x b_j)
        
#         The numerator is based on Eq 20 of the papr. i goes from 1 to k
#         priority classes. c_i is the 2nd moment of service time distribution.
#         In the denominator, j in the first summation goes from 1 to i - 1.
#         In the second summarion, j goes from 1 to i. i = 1 has the highest
#         priority. b_j is the is the 1st moment (i.e. mean) service rate of
#         class j. Lambda'_j is the individual arrival rate for the j-th class.

#         In our case, AI+ patients in every group with AI form a priority class
#         ranked by the disease rank. For each group with AI, we need to collect
#         the first and second moments of service time distribution for the
#         positive patients by that AI. Because it includes both diseased and
#         non-diseased, the service time distribution is a hyperexponential
#         distribution with probabilities of PPV (or 1-NPV) and 1-PPV (or NPV)
#         to take into account the FP and FN as well. 

#         This method can calculate the mean wait-time of each positive subgroup
#         within each group with AI. It is currently limited to scenarios where
#         each group has only 1 disease and at most 1 AI.

#         inputs
#         ------
#         params (dict): dictionary with user inputs
#         aHierarchy (hierarchy): hierarchy of ranked priority classes

#         outputs
#         -------
#         theories (dict): theorectical predictions for all priority and disease classes.
#     '''
    
#     ## Initialize holders: these values are ordered by disease rank
#     Ses, Sps, gp_probs = [], [], []    
#     #  To store the first and second moments per priority class.
#     neg_means, neg_2nd_moment = [], [] 
#     pos_means, pos_2nd_moment = [], []
#     #  To store the positive rate per priority class w.r.t. whole population
#     prob_pos_groups = []
#     #  To store individual arrival rate of positive cases by the 1 AI in the group
#     arrival_rates = [] 
#     #  To store the fraction of target disease (in this priority class)
#     #  within the overall AI negative class
#     prob_thisgroup_given_negs = []
#     no_vendor_indices = []

#     ## Loop through all groups, get info for each AI-positive subgroup
#     for i in range(len(aHierarchy.groupNames)):
#         groupname = aHierarchy.groupNames[i]
#         diseasename = aHierarchy.diseaseNames[i]
#         vendorname = aHierarchy.AINames[i]
#         ## Get disease prob within the group and group probability
#         dis_idx = params['diseaseGroups'][groupname]['diseaseNames'].index (diseasename)
#         dis_prob = params['diseaseGroups'][groupname]['diseaseProbs'][dis_idx]
#         gp_prob = params['diseaseGroups'][groupname]['groupProb']
#         gp_probs.append(gp_prob)
#         ## Get AI vendor performance. If no vendor in this group,
#         ## Se/Sp = 0, 1 - all cases in this group is counted as AI-.
#         Se, Sp = 0, 1
#         probdis_givenpos, probdis_givenneg = 0, dis_prob
#         probnondis_givenpos, probnondis_givenneg = 0, 1-dis_prob
#         groupnames = aHierarchy.groupNames
#         diseasenames = aHierarchy.diseaseNames
#         vendornames = aHierarchy.AINames
#         vendor_flag, filled_vendornames = fill_missing_vendors(groupnames, diseasenames, vendornames)

#         if aHierarchy.AINames[i] is not None or (aHierarchy.AINames[i] is None and filled_vendornames[i] is None): #If current disease has a vendor OR does not have a vendor but there is no other vendor in the group
#             if aHierarchy.AINames[i] is not None:
#                 Se = params['SeThreshs'][vendorname] 
#                 Sp = params['SpThreshs'][vendorname]
#                 probdis_givenpos = params['probs_ppv_npv']['ppv'][groupname][vendorname]
#                 probdis_givenneg = 1 - params['probs_ppv_npv']['npv'][groupname][vendorname]
#             Ses.append(Se)
#             Sps.append(Sp)
#             dis_readtime = params['meanServiceTimes'][groupname][diseasename]
#             nondis_readtime = params['meanServiceTimes'][groupname]['non-diseased']
 
#             # Initialize holders for looping through other diseases in the group (if they exist)
#             other_probs_givenpos = []
#             other_probs_givenneg = []
#             other_probs_readtime = [] # to get read time and 
#             other_probs_secondmoments = []
#             other_dis_probs = []
#             prob_thisgroup_given_neg = 0
#             for otherdis_in_group in params['diseaseGroups'][groupname]['diseaseNames']:
#                 if otherdis_in_group != diseasename:
#                     dis_idx = params['diseaseGroups'][groupname]['diseaseNames'].index(otherdis_in_group)
#                     this_dis_prob = params['diseaseGroups'][groupname]['diseaseProbs'][dis_idx] # get prevalence of current disease
#                     other_dis_probs.append(this_dis_prob)
#                     other_probs_readtime.append(params['meanServiceTimes'][groupname][otherdis_in_group])
#                     other_probs_secondmoments.append(2 * params['meanServiceTimes'][groupname][otherdis_in_group]**2)
#                     other_probs_givenpos.append((1 - Sp) * (this_dis_prob/(1-dis_prob)) / (Se * dis_prob + (1 - Sp) * (1 - dis_prob))) # get probability that positive AI reading came from current disease
#                     other_probs_givenneg.append((1 - Se) * (this_dis_prob/(1-dis_prob)) / ((1 - Se) * dis_prob + Sp * (1 - dis_prob))) # get probability that negative AI reading came from current disease
#                     prob_thisgroup_given_neg += gp_prob * this_dis_prob  # Variable keeps track of, among all AI-negative, what is the fraction of each group. 

#             # get probability of non diseased given AI positive / AI negative reading
#             probnondis_givenpos = 1 - sum(other_probs_givenpos) - probdis_givenpos
#             probnondis_givenneg = 1 - sum(other_probs_givenneg) - probdis_givenneg

#             # Get effective reading times
#             pos_means.append(dis_readtime * probdis_givenpos + sum(numpy.array(other_probs_readtime)*numpy.array(other_probs_givenpos)) + nondis_readtime * probnondis_givenpos) 
#             neg_means.append(dis_readtime * probdis_givenneg + sum(numpy.array(other_probs_readtime)*numpy.array(other_probs_givenneg)) + nondis_readtime * probnondis_givenneg)
#             ## Get the second moments
#             ##   E[X^2] = sum_i (pi * 2 / ratei**2) = sum_i (pi * 2 * timei**2), with i = diseased vs non-diseased
#             dis_2nd_moment_factor = 2 * dis_readtime**2
#             nondis_2nd_moment_factor = 2 * nondis_readtime**2
#             pos_2nd_moment.append(dis_2nd_moment_factor * probdis_givenpos + sum(numpy.array(other_probs_secondmoments)*numpy.array(other_probs_givenpos)) + nondis_2nd_moment_factor * probnondis_givenpos)
#             neg_2nd_moment.append(dis_2nd_moment_factor * probdis_givenneg + sum(numpy.array(other_probs_secondmoments)*numpy.array(other_probs_givenneg)) + nondis_2nd_moment_factor * probnondis_givenneg)

#             ## Get probability of positive across whole population
#             prob_pos_group = (Se * dis_prob + (1 - Sp) * (1 - dis_prob)) * gp_prob
#             prob_pos_groups.append(prob_pos_group)
#             ## Get arrival rate for patients with this specific disease 
#             arrival_rate = prob_pos_group * params['arrivalRates']['non-interrupting']
#             arrival_rates.append(arrival_rate)
#             ## Among all AI-negative, what is the fraction of each group -- add original disease that has a vendor
#             prob_thisgroup_given_neg += gp_prob * (Sp * (1 - dis_prob) + (1 - Se) * dis_prob)
#             prob_thisgroup_given_negs.append(prob_thisgroup_given_neg)

#         else: # If disease does not have a vendor but there is another vendor in the group
#             # 'Fill in' corresponding vendor in group and get Se / Sp
#             no_vendor_indices.append(i)
#             vendorname = filled_vendornames[i] 
#             Se = params['SeThreshs'][vendorname] 
#             Sp = params['SpThreshs'][vendorname] 
#             Ses.append(Se)
#             Sps.append(Sp)
#             pos_means.append(0)
#             pos_2nd_moment.append(0)
#             arrival_rates.append(0)

#     ## For the AI-negative group (i.e. cases flagged as negative by all AIs).
#     ## Make use of neg_means, neg_2nd_moment, and prob_thisdis_given_negs
#     prob_thisgroup_given_negs = numpy.array(prob_thisgroup_given_negs) / sum(prob_thisgroup_given_negs) 
#     ## AI_neg_means = effective read time [min] for AI-neg group that may have different disease types
#     AI_neg_mean = sum(prob_thisgroup_given_negs * numpy.array(neg_means))
#     AI_neg_2nd_moment  = sum(prob_thisgroup_given_negs * numpy.array(neg_2nd_moment))
#     AI_neg_arrival_rate = (1-sum(prob_pos_groups)) * params['arrivalRates']['non-interrupting']

#     ## Calculation of wait-time
#     bs = numpy.array (pos_means + [AI_neg_mean])
#     cs = numpy.array (pos_2nd_moment + [AI_neg_2nd_moment])
#     lambdas = numpy.array (arrival_rates + [AI_neg_arrival_rate])

#     #  Get c (Eq 20) 
#     c = sum (lambdas * cs)
#     ## Apply Eq 21. For the highest priority, only the second summation in the denominator counts.
#     ## These are the mean wait-time of each positive subgroup by each AI in each group. For group
#     ## that don't have an AI, it spit out a mean wait-time all cases are at the end of a sub-system
#     ## with only higher rank AI-positive diseases in front (as if other lower groups do not exist).
#     wait_times = []
#     for j in range(len(lambdas)):
#         wait_time = c / (2 * (1 - sum(lambdas[:j+1] * bs[:j+1]))) ## second summation
#         if j > 0: wait_time /= 1 - sum(lambdas[:j] * bs[:j])
#         wait_times.append (wait_time)
#     wait_times = numpy.array (wait_times)

#     ## Converts the AI+/- wait time into per-disease wait-time. 
#     Ses, Sps, gp_probs = numpy.array (Ses), numpy.array (Sps), numpy.array (gp_probs)
#     diseased_wait_times = wait_times[:-1] * Ses + wait_times[-1] * (1 - Ses)

#     # Slightly different calculation for diseased wait times for those that do not have a vendor.
#     for i in no_vendor_indices:
#         current_vendor = filled_vendornames[i]
#         index_of_vendor = filled_vendornames.index(current_vendor)
#         Sp = params['SpThreshs'][current_vendor] 
#         diseased_wait_times[i] = wait_times[index_of_vendor] * (1 - Sp) + wait_times[-1] * Sp
    
#     ## Populate results into a dictionary
#     priority = {}
#     for index, (disease, vendor, wait_time) in enumerate (zip (aHierarchy.diseaseNames, aHierarchy.AINames, diseased_wait_times)):
#         priority[disease] = {'diseased':wait_time}
#         priority[disease]['negative'] = wait_times[-1]
#         if vendor is None: continue
#         priority[disease]['positive'] = wait_times[index]
        
#     return priority

def get_all_waitTime_priority_nonpreemptive (params, aHierarchy):

    ''' Function to calculate mean wait-time per disease conditions and across
        all disease conditions. This function re-use the method in 
        get_all_waitTime_hierarchical_nonpreemptive(), except that there are
        only 2 priority classes: overall AI+ and AI- (no hierarchical queue).

        inputs
        ------
        params (dict): dictionary with user inputs
        aHierarchy (hierarchy): hierarchy of ranked priority classes

        outputs
        -------
        theories (dict): theorectical predictions for all priority and disease classes.
    '''
    
    pos_mean, pos_2nd_moment = 0, 0
    for gpname, probinfo in params['probs_condByPriority']['positive'].items():
        for disname, prob in probinfo.items():
            readtime = params['meanServiceTimes'][gpname][disname]
            pos_mean += prob * readtime
            pos_2nd_moment += 2 * readtime**2 * prob
    
    neg_mean, neg_2nd_moment = 0, 0
    for gpname, probinfo in params['probs_condByPriority']['negative'].items():
        for disname, prob in probinfo.items():
            readtime = params['meanServiceTimes'][gpname][disname]
            neg_mean += prob * readtime
            neg_2nd_moment += 2 * readtime**2 * prob

    ## Calculation of wait-time
    bs = numpy.array ([pos_mean, neg_mean])
    cs = numpy.array ([pos_2nd_moment, neg_2nd_moment])
    lambdas = numpy.array ([params['arrivalRates']['positive'], params['arrivalRates']['negative']])
    #  Get c (Eq 20) 
    c = sum (lambdas * cs)
    ## Apply Eq 21.
    wait_times = []
    for j in range(len(lambdas)):
        wait_time = c / (2 * (1 - sum(lambdas[:j+1] * bs[:j+1]))) 
        if j > 0: wait_time /= 1 - sum(lambdas[:j] * bs[:j])
        wait_times.append (wait_time)
    wait_times = numpy.array (wait_times)

    ## Create an equivalent AI, group, and the corresponding diseased probabilities.
    all_se, all_sp, all_pop_prevalence, all_groupProbs, _ = aHierarchy._get_info_for_equivalent_SeSp (deepcopy (params))
    EQSe, _ = aHierarchy._calculate_equivalent_Se_Sp(all_se, all_sp, all_pop_prevalence, all_groupProbs)

    ## Converts the AI+/- wait time into per-disease wait-time. Note that this
    ## should be compared with the mean wait-time from diseased patients that
    ## have AIs (i.e. do not include diseased patients that do not have AIs).
    diseased_wait_times = wait_times[0] * EQSe + wait_times[-1] * (1 - EQSe)

    ## Populate results into a dictionary 
    priority = {'positive': wait_times[0], 'negative':wait_times[1],
                'diseased': diseased_wait_times}
                            
    ## Now, calculate the wait-time for each diseased cases
    for disease, group in zip (aHierarchy.diseaseNames, aHierarchy.groupNames):
        posWeight, negWeight = aHierarchy.get_posNegWeights (params, disease, group)
        priority[disease] = priority['positive']*posWeight + priority['negative']*negWeight
        
    return priority

########################################
## Classic queueing state probabilities
########################################
def get_state_pdf_MMn (ns, s, aLambda, aMu):
    
    ''' Function to obtain a state probability for a M/M/n system.

        inputs
        ------
        ns (array): number of customers in system
        s (int): number of servers
        aLambda (float): customer arrival rate
        aMu (float): server reading rate

        output
        ------
        pred (array): state probability
    '''

    rho = aLambda / s / aMu
    
    p0_first = numpy.sum ([(s*rho)**(i)/numpy.math.factorial (i) for i in numpy.linspace (0, s-1, s)])
    p0_second = (s*rho)**s / numpy.math.factorial  (s) / (1-rho)
    p0 = 1/(p0_first + p0_second)
    
    pred = []
    for n in ns:
        p = (s*rho)**n * p0 / numpy.math.factorial  (n) if n <= s else \
            s**s*rho**n * p0 / numpy.math.factorial  (s)
        pred.append (p)
    
    return numpy.array (pred)

########################################
## RDR method
########################################
def DR_2_phase_coxian (muG1, muG2, muG3):

    ''' Closed form solutions for mapping general distributions to
        quasi-minimal PH distributions with only 2 Coxian phases.

        See https://www.cs.cmu.edu/~harchol/Papers/quasi-minimal-PH.pdf
        for more details.

        inputs
        ------
        muG1 (float): first moment of busy period distribution
        muG2 (float): second moment of busy period distribution
        muG3 (float): third moment of busy period distribution
    '''

    mG2 = muG2 / muG1**2
    mG3 = muG3 / muG1 / muG2
    denom = 3*mG2 - 2*mG3
    if denom == 0: denom += 0.001
    u = (6-2*mG3) / denom
    v = (12-6*mG2) / (mG2*denom)
    if round (u**2 - 4*v, 10) == 0: 
        lambdaX1 = (u + numpy.sqrt (0)) / (2*muG1)
        lambdaX2 = (u - numpy.sqrt (0)) / (2*muG1)
    else:
        lambdaX1 = (u + numpy.sqrt (u**2 - 4*v)) / (2*muG1)
        lambdaX2 = (u - numpy.sqrt (u**2 - 4*v)) / (2*muG1)
    pX = lambdaX2 * (lambdaX1 * muG1 - 1) / lambdaX1
    if lambdaX1 == 0: pX = 1
    # n = 2; p = 1; lambdaY = 0
    return [2, 1, 0, lambdaX1, lambdaX2, pX]

def DR_simple_solution (muG1, muG2, muG3):

    ''' Closed form solutions for mapping general distributions to
        quasi-minimal PH distributions (simple solutions).

        See https://www.cs.cmu.edu/~harchol/Papers/quasi-minimal-PH.pdf
        for more details.

        inputs
        ------
        muG1 (float): first moment of busy period distribution
        muG2 (float): second moment of busy period distribution
        muG3 (float): third moment of busy period distribution
    '''

    mG2 = muG2 / muG1**2
    mG3 = muG3 / muG1 / muG2
    ## In U0 and M0 (shaded area in thesis Fig 2.12)
    ## 2-phase coxian is sufficient
    if mG2 > 2 and mG3 > 2*mG2 - 1:
        return DR_2_phase_coxian (muG1, muG2, muG3)
    # left side (thesis Fig. 2.12): p = 1
    p = 1 if not mG3 < 2*mG2 - 1 else 1/(2*mG2 - mG3)
    muW1 = muG1 / p
    mW2 = p*mG2
    mW3 = p*mG3
    n = numpy.floor (mW2/(mW2 - 1) + 1)
    
    mX2 = ((n-3)*mW2 - (n-2)) / \
          ((n-2)*mW2 - (n-1))
    muX1 = muW1 / \
           ((n-2)*mX2 - (n-3))
    alpha = (n-2)*(mX2-1)*(n*(n-1)*mX2**2 - n*(2*n-5)*mX2 + (n-1)*(n-3))
    beta = ((n-1)*mX2-(n-2))*((n-2)*mX2-(n-3))**2
    mX3 = (beta*mW3 - alpha) / mX2
    
    u = (6-2*mX3) / (3*mX2 - 2*mX3)
    v = (12-6*mX2) / (mX2*(3*mX2 - 2*mX3))
    lambdaX1 = (u + numpy.sqrt (u**2 - 4*v)) / (2*muX1)
    lambdaX2 = (u - numpy.sqrt (u**2 - 4*v)) / (2*muX1)
    pX = lambdaX2 * (lambdaX1 * muX1 - 1) / lambdaX1
    lambdaY = 1/(muX1*(mX2-1))

    return [n, p, lambdaY, lambdaX1, lambdaX2, pX]

def def_cal_MMs (p, highPriority='interrupting', lowPriority='non-interrupting', doDynamic=False):
    
    ''' Obtain a calculator for the lower priority class with only
        two priority classes. Note that this function also applies
        to the second high priority class when more than two priority
        classes in a preemptive-resume setting because the presence
        or absence of lower priority customers do not affect the
        second high priority class.

        inputs
        ------
        params (dict): settings for all simulations
        highPriority (str): name of the higher priority class above the second high class
        lowPriority (str): name of the lower priority class of interest
        doDynamic (bool): use the dynamic matrix formation (esp used when more than
                          3 radiologists).
    '''

    cal = traditional_calculator()
    rhoH = p['rhos'][highPriority]
    bB = p['mus'][highPriority]*p['nRadiologists']
    muG1 = 1 / bB / (1-rhoH)
    muG2 = 2 / bB**2 / (1-rhoH)**3
    muG3 = 6 * (1+rhoH) / bB**3 / (1-rhoH)**5
    DR = DR_2_phase_coxian (muG1, muG2, muG3)

    if doDynamic:
        cal.form_DR_M_M_s_matrix(p['nRadiologists'],
                     p['lambdas'][lowPriority],
                     p['lambdas'][highPriority],
                     p['mus'][lowPriority],
                     p['mus'][highPriority],
                     (1-DR[5])*DR[3], DR[5]*DR[3], DR[4])
    elif p['nRadiologists'] == 1:
        cal.form_DR_M_M_1_matrix(p['lambdas'][lowPriority],
                     p['lambdas'][highPriority],
                     p['mus'][lowPriority],
                     p['mus'][highPriority],
                     (1-DR[5])*DR[3], DR[5]*DR[3], DR[4])
    elif p['nRadiologists'] == 2:
        cal.form_DR_M_M_2_matrix(p['lambdas'][lowPriority],
                     p['lambdas'][highPriority],
                     p['mus'][lowPriority],
                     p['mus'][highPriority],
                     (1-DR[5])*DR[3], DR[5]*DR[3], DR[4])
    elif p['nRadiologists'] == 3:
        cal.form_DR_M_M_3_matrix(p['lambdas'][lowPriority],
                     p['lambdas'][highPriority],
                     p['mus'][lowPriority],
                     p['mus'][highPriority],
                     (1-DR[5])*DR[3], DR[5]*DR[3], DR[4])

    return cal

def get_cal_lowest (params):

    ''' Obtain a calculator for the loweset priority, AI-negative class in the
        with CADt scenario. Note that there is valid theoretical predictions
        when number of radiologists is 1 or 2. None will be returned if
        nRadiologists > 3.

        inputs
        ------
        params (dict): settings for all simulations
    '''

    if params['nRadiologists'] > 3: return None

    ## Get busy periods and cond. probabilities from middle class
    MidCal = MG1_calculator()
    rhoH = params['rhos']['interrupting']
    bB = params['mus']['interrupting']*params['nRadiologists']
    muG1 = 1 / bB / (1-rhoH)
    muG2 = 2 / bB**2 / (1-rhoH)**3
    muG3 = 6 * (1+rhoH) / bB**3 / (1-rhoH)**5
    DR = DR_2_phase_coxian (muG1, muG2, muG3)
    if params['nRadiologists'] == 1:
        MidCal.form_DR_M_M_1_matrix(params['lambdas']['positive'],
                                    params['lambdas']['interrupting'],
                                    params['mus']['positive'],
                                    params['mus']['interrupting'],
                                    (1-DR[5])*DR[3], DR[5]*DR[3], DR[4])
    elif params['nRadiologists'] == 2:
        MidCal.form_DR_M_M_2_matrix(params['lambdas']['positive'],
                                    params['lambdas']['interrupting'],
                                    params['mus']['positive'],
                                    params['mus']['interrupting'],
                                    (1-DR[5])*DR[3], DR[5]*DR[3], DR[4])
        t12_M = DR[5]*DR[3]

    MidCal.get_Zs()
    ## These are non-repeating
    Gs = MidCal._G_nonrep
    Zs = MidCal.Zs
    
    ## For 1 radiologist: 2 busy periods
    if params['nRadiologists'] == 1:
        ## Use [0,0] and [1,0] elements in Zs / Gs for 1 rad
        ## For [0,0]: (0,1,l) -> (0,0,l)
        DR = DR_2_phase_coxian (Zs['Z1'][0][0], Zs['Z2'][0][0], Zs['Z3'][0][0])
        t1_M, t12_M, t2_M = (1-DR[5])*DR[3], DR[5]*DR[3], DR[4]
        p_M = Gs[0][0]
        
        ## For [1,0]: (1,0,l) -> (0,0,l)
        DR = DR_2_phase_coxian (Zs['Z1'][1][0], Zs['Z2'][1][0], Zs['Z3'][1][0])
        t1_H, t12_H, t2_H = (1-DR[5])*DR[3], DR[5]*DR[3], DR[4]
        p_H = Gs[1][0]
    
        ## Get calculator for lower class
        cal = traditional_calculator()
        cal.form_DR_M_M_1_3classes_matrix(params['lambdas']['negative'],
                                          params['lambdas']['positive'],
                                          params['lambdas']['interrupting'],
                                          params['mus']['negative'],
                                          p_M, t1_M, t12_M, t2_M,
                                          p_H, t1_H, t12_H, t2_H)
        return cal
    
    ## For 2 radiologists; 6 busy periods
    if params['nRadiologists'] == 2:
        ## 1. [0,0]: (0,2,l) -> (0,1,l)
        DR = DR_simple_solution (Zs['Z1'][0][0], Zs['Z2'][0][0], Zs['Z3'][0][0])
        t1_2M_1M, t12_2M_1M, t2_2M_1M = (1-DR[5])*DR[3], DR[5]*DR[3], DR[4]
        p_2M_1M = Gs[0][0]
        
        ## 2. [0,1]: (0,2,l) -> (1,0,l) !!!
        DR = DR_simple_solution (Zs['Z1'][0][1], Zs['Z2'][0][1], Zs['Z3'][0][1])
        t0_2M_1H, t01_2M_1H = (1-DR[1])*DR[2], DR[1]*DR[2]
        t1_2M_1H, t12_2M_1H, t2_2M_1H = (1-DR[5])*DR[3], DR[5]*DR[3], DR[4]
        #  By Michelle
        if params['fractionED'] == 0.0:
            t0_2M_1H, t01_2M_1H, t1_2M_1H, t12_2M_1H, t2_2M_1H = 0, 0, 0, 0, 0
        p_2M_1H = Gs[0][1]
        
        ## 3. [1,0]: (1,1,l) -> (0,1,l)
        DR = DR_simple_solution (Zs['Z1'][1][0], Zs['Z2'][1][0], Zs['Z3'][1][0])
        t1_1M1H_1M, t12_1M1H_1M, t2_1M1H_1M = (1-DR[5])*DR[3], DR[5]*DR[3], DR[4]
        p_1M1H_1M = Gs[1][0]
    
        ## 4. [1,1]: (1,1,l) -> (1,0,l)
        DR = DR_simple_solution (Zs['Z1'][1][1], Zs['Z2'][1][1], Zs['Z3'][1][1])
        t1_1M1H_1H, t12_1M1H_1H, t2_1M1H_1H = (1-DR[5])*DR[3], DR[5]*DR[3], DR[4]
        p_1M1H_1H = Gs[1][1]
    
        ## 5. [2,0]: (2+,0,l) -> (0,1,l) !!!
        DR = DR_simple_solution (Zs['Z1'][2][0], Zs['Z2'][2][0], Zs['Z3'][2][0])
        t0_2H_1M, t01_2H_1M = (1-DR[1])*DR[2], DR[1]*DR[2]
        t1_2H_1M, t12_2H_1M, t2_2H_1M = (1-DR[5])*DR[3], DR[5]*DR[3], DR[4]
        p_2H_1M = Gs[2][0]
        
        ## 6. [2,1]: (2+,0,l) -> (1,0,l)
        DR = DR_simple_solution (Zs['Z1'][2][1], Zs['Z2'][2][1], Zs['Z3'][2][1])
        t1_2H_1H, t12_2H_1H, t2_2H_1H = (1-DR[5])*DR[3], DR[5]*DR[3], DR[4]
        p_2H_1H = Gs[2][1]

        ## Get calculator for lower class
        cal = traditional_calculator()
        cal.form_DR_M_M_2_3classes_matrix(params['lambdas']['negative'],
                                          params['lambdas']['positive'],
                                          params['lambdas']['interrupting'],
                                          params['mus']['negative'],
                                          params['mus']['positive'],
                                          params['mus']['interrupting'],
                                          p_2M_1M  , t1_2M_1M  , t12_2M_1M  , t2_2M_1M  ,
                                          p_2M_1H  , t1_2M_1H  , t12_2M_1H  , t2_2M_1H  , t0_2M_1H, t01_2M_1H, 
                                          p_1M1H_1M, t1_1M1H_1M, t12_1M1H_1M, t2_1M1H_1M,
                                          p_1M1H_1H, t1_1M1H_1H, t12_1M1H_1H, t2_1M1H_1H,
                                          p_2H_1M  , t1_2H_1M  , t12_2H_1M  , t2_2H_1M  , t0_2H_1M, t01_2H_1M,
                                          p_2H_1H  , t1_2H_1H  , t12_2H_1H  , t2_2H_1H  )
        
        return cal

############################################
## Calculators 
############################################
class stateProb_calculator (object):

    def __init__ (self):
        
        self._A0, self._A1, self._A2 = None, None, None
        self._B00, self._B01, self._B10 = None, None, None

        self._pis = None

    @property
    def pis (self): return self._pis

    @property
    def A0 (self): return self._A0
    @A0.setter
    def A0 (self, A0):
        if not self._matrix_is_square (A0):
            print ('Please provide a square matrix for A0.')
            return
        self._A0 = A0 
        
    @property
    def A1 (self): return self._A1
    @A1.setter
    def A1 (self, A1):
        if not self._matrix_is_square (A1):
            print ('Please provide a square matrix for A1.')
            return
        self._A1 = A1 
        
    @property
    def A2 (self): return self._A2
    @A2.setter
    def A2 (self, A2):
        if not self._matrix_is_square (A2):
            print ('Please provide a square matrix for A2.')
            return
        self._A2 = A2 
        
    @property
    def B00 (self): return self._B00
    @B00.setter
    def B00 (self, B00):
        good_B00 = self._matrix_is_square (B00) or isinstance (B00, float) \
                   or isinstance (B00, int)
        if not good_B00:
            print ('Please provide a value or square matrix for B00.')
            return
        self._B00 = B00 

    @property
    def B10 (self): return self._B10
    @B10.setter
    def B10 (self, B10): self._B10 = B10 

    @property
    def B01 (self): return self._B01
    @B01.setter
    def B01 (self, B01): self._B01 = B01
    
    #######################################################
    ## Check system
    #######################################################
    def _matrix_is_square (self, matrix):
        
        shape = matrix.shape
        if not len (shape) == 2: return False
        return shape[0] == shape[1]
        
    def _have_matching_shapes (self):
        
        shape = self._A1.shape
        # A0 and A2 must have the same shape
        if not self._A0.shape == shape:
            print ('A0 and A1 have different shape.')
            return False
        if not self._A2.shape == shape:
            print ('A2 and A1 have different shape.')
            return False
        
        # B01 must have the same # columns as A1
        if not self._B01.shape[1] == shape[1]:
            print ('B01 and A1 have different # columns.')
            return False
        
        # B10 must have the same # rows as A1
        if not self._B10.shape[0] == shape[0]:
            print ('B10 and A1 have different # rows.')
            return False

        # B00 must have the same # rows as B01
        # and same # columns as B10
        if not self._B00.shape[0] == self._B01.shape[0]:
            print ('B00 and B01 have different # rows.')
            return False
        if not self._B00.shape[1] == self._B10.shape[1]:
            print ('B00 and B10 have different # columns.')
            return False

        return True

    def _is_valid_system (self):

        for element in ['A0', 'A1', 'A2', 'B10', 'B01', 'B00']:
            matrix = eval ('self._' + element)
            if matrix is None:
                print ('Please provide a valid {0} matrix.'.format (element))
                return False

        # Make sure matrix shapes are matched
        if not self._have_matching_shapes (): return False
        
        return True

    def _negated_sum (self, matrix):
        
        for i, row in enumerate (matrix):
            matrix[i][i] = -numpy.sum (numpy.append (row[:i], row[i+1:]))
 
        return matrix

    def _combine_matrix (self, a00, a01, a10, a11):
        return numpy.vstack([numpy.hstack([a00, a01]), numpy.hstack([a10, a11])])
    
    def _solve_LH_vector (self, p):
        # Find left-handed eigenvalues / vectors
        vals, vecs = eig (p, left=True, right=False)
        # Get the value closest to 1 - should be the largest
        closest_to_1_idx = vals.argsort()[::-1][0]
        val = numpy.round (vals[closest_to_1_idx], 4)
        if val < 0 or val > 1:
            print ('Largest eigenvalue, {0}, found is invalid.'.format (val))
            return None
        # Get the corresponding eigenvector
        vec = vecs[:,closest_to_1_idx]
        # Normalize it
        return vec / numpy.sum (vec)
    
class traditional_calculator (stateProb_calculator):

    def __init__ (self):    
        super(stateProb_calculator, self).__init__ ()
        self._R = None
        
        self._is_MH2C2 = False
        self._is_MC2 = False
        self._nRad = 1

    @property
    def R (self): return self._R

    #######################################################
    ## Form matrix
    #######################################################
    def form_DR_M_M_1_3classes_matrix (self, lambdaL, lambdaM, lambdaH, muL,
                                       p_M, t1_M, t12_M, t2_M,
                                       p_H, t1_H, t12_H, t2_H):
        
        self._B00 = numpy.array ([[-lambdaH*p_H-lambdaM*p_M-lambdaL,         lambdaM*p_M,             0,         lambdaH*p_H,             0],
                                  [                            t1_M, -lambdaL-t1_M-t12_M,         t12_M,                   0,             0],
                                  [                            t2_M,                   0, -lambdaL-t2_M,                   0,             0],
                                  [                            t1_H,                   0,             0, -lambdaL-t1_H-t12_H,         t12_H],
                                  [                            t2_H,                   0,             0,                   0, -lambdaL-t2_H]])
        self._B01 = numpy.array ([[lambdaL,       0,       0,       0,       0],
                                  [      0, lambdaL,       0,       0,       0],
                                  [      0,       0, lambdaL,       0,       0],
                                  [      0,       0,       0, lambdaL,       0],
                                  [      0,       0,       0,       0, lambdaL]])
        self._B10 = numpy.array ([[muL, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0]])
        
        ## In paper, A0 = B; A1 = L; A2 = F
        self._A0 = self._B10
        self._A1 = numpy.array ([[-lambdaH*p_H-lambdaM*p_M-lambdaL-muL,         lambdaM*p_M,             0,         lambdaH*p_H,             0],
                                 [                                t1_M, -lambdaL-t1_M-t12_M,         t12_M,                   0,             0],
                                 [                                t2_M,                   0, -lambdaL-t2_M,                   0,             0],
                                 [                                t1_H,                   0,             0, -lambdaL-t1_H-t12_H,         t12_H],
                                 [                                t2_H,                   0,             0,                   0, -lambdaL-t2_H]])
        self._A2 = self._B01

    def form_DR_M_M_2_3classes_matrix (self, lambdaL, lambdaM, lambdaH, muL, muM, muH,
                                       p_2M_1M  , t1_2M_1M  , t12_2M_1M  , t2_2M_1M  ,
                                       p_2M_1H  , t1_2M_1H  , t12_2M_1H  , t2_2M_1H  , t0_2M_1H, t01_2M_1H, 
                                       p_1M1H_1M, t1_1M1H_1M, t12_1M1H_1M, t2_1M1H_1M,
                                       p_1M1H_1H, t1_1M1H_1H, t12_1M1H_1H, t2_1M1H_1H,
                                       p_2H_1M  , t1_2H_1M  , t12_2H_1M  , t2_2H_1M  , t0_2H_1M, t01_2H_1M,
                                       p_2H_1H  , t1_2H_1H  , t12_2H_1H  , t2_2H_1H ):
        
        self._B00 = numpy.array ([[-lambdaH-lambdaM-lambdaL,                                                                         lambdaM,                                                                         lambdaH,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [                     muM,-lambdaM*p_2M_1M-lambdaM*p_2M_1H-lambdaH*p_1M1H_1M-lambdaH*p_1M1H_1H-lambdaL-muM,                                                                               0,            lambdaM*p_2M_1M,                0,            lambdaM*p_2M_1H,                          0,                0,              lambdaH*p_1M1H_1M,                  0,              lambdaH*p_1M1H_1H,                  0,                          0,                          0,                0,                          0,                0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [                     muH,                                                                               0,-lambdaM*p_1M1H_1M-lambdaM*p_1M1H_1H-lambdaH*p_2H_1M-lambdaH*p_2H_1H-lambdaL-muH,                          0,                0,                          0,                          0,                0,              lambdaM*p_1M1H_1M,                  0,              lambdaM*p_1M1H_1H,                  0,            lambdaH*p_2H_1M,                          0,                0,            lambdaH*p_2H_1H,                0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [                       0,                                                                        t1_2M_1M,                                                                               0,-lambdaL-t1_2M_1M-t12_2M_1M,        t12_2M_1M,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [                       0,                                                                        t2_2M_1M,                                                                               0,                          0,-lambdaL-t2_2M_1M,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [                       0,                                                                               0,                                                                        t0_2M_1H,                          0,                0,-lambdaL-t0_2M_1H-t01_2M_1H,                  t01_2M_1H,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],                                  
                                  [                       0,                                                                               0,                                                                        t1_2M_1H,                          0,                0,                          0,-lambdaL-t1_2M_1H-t12_2M_1H,        t12_2M_1H,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [                       0,                                                                               0,                                                                        t2_2M_1H,                          0,                0,                          0,                          0,-lambdaL-t2_2M_1H,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [                       0,                                                                      t1_1M1H_1M,                                                                               0,                          0,                0,                          0,                          0,                0,-lambdaL-t1_1M1H_1M-t12_1M1H_1M,        t12_1M1H_1M,                              0,                  0,                          0,                          0,                0,                          0,                0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [                       0,                                                                      t2_1M1H_1M,                                                                               0,                          0,                0,                          0,                          0,                0,                              0,-lambdaL-t2_1M1H_1M,                              0,                  0,                          0,                          0,                0,                          0,                0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0],
                                  [                       0,                                                                               0,                                                                      t1_1M1H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,-lambdaL-t1_1M1H_1H-t12_1M1H_1H,        t12_1M1H_1H,                          0,                          0,                0,                          0,                0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0],
                                  [                       0,                                                                               0,                                                                      t2_1M1H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,-lambdaL-t2_1M1H_1H,                          0,                          0,                0,                          0,                0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0],
                                  [                       0,                                                                        t0_2H_1M,                                                                               0,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,-lambdaL-t0_2H_1M-t01_2H_1M,                  t01_2H_1M,                0,                          0,                0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0],                                  
                                  [                       0,                                                                        t1_2H_1M,                                                                               0,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,-lambdaL-t1_2H_1M-t12_2H_1M,        t12_2H_1M,                          0,                0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0],
                                  [                       0,                                                                        t2_2H_1M,                                                                               0,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,-lambdaL-t2_2H_1M,                          0,                0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0],
                                  [                       0,                                                                               0,                                                                        t1_2H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,-lambdaL-t1_2H_1H-t12_2H_1H,        t12_2H_1H,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0],
                                  [                       0,                                                                               0,                                                                        t2_2H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,-lambdaL-t2_2H_1H,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL],
                                  [muL,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-lambdaH-lambdaM-lambdaL-muL,                                                                             lambdaM,                                                                             lambdaH,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                  [  0, muL,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                         muM,-lambdaM*p_2M_1M-lambdaM*p_2M_1H-lambdaH*p_1M1H_1M-lambdaH*p_1M1H_1H-lambdaL-muL-muM,                                                                                   0,            lambdaM*p_2M_1M,                0,            lambdaM*p_2M_1H,                          0,                0,              lambdaH*p_1M1H_1M,                  0,              lambdaH*p_1M1H_1H,                  0,                          0,                          0,                0,                          0,                0],
                                  [  0,   0, muL, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                         muH,                                                                                   0,-lambdaM*p_1M1H_1M-lambdaM*p_1M1H_1H-lambdaH*p_2H_1M-lambdaH*p_2H_1H-lambdaL-muL-muH,                          0,                0,                          0,                          0,                0,              lambdaM*p_1M1H_1M,                  0,              lambdaM*p_1M1H_1H,                  0,            lambdaH*p_2H_1M,                          0,                0,            lambdaH*p_2H_1H,                0],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                            t1_2M_1M,                                                                                   0,-lambdaL-t1_2M_1M-t12_2M_1M,        t12_2M_1M,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                            t2_2M_1M,                                                                                   0,                          0,-lambdaL-t2_2M_1M,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                                   0,                                                                            t0_2M_1H,                          0,                0,-lambdaL-t0_2M_1H-t01_2M_1H,                  t01_2M_1H,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],                                  
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                                   0,                                                                            t1_2M_1H,                          0,                0,                          0,-lambdaL-t1_2M_1H-t12_2M_1H,        t12_2M_1H,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                                   0,                                                                            t2_2M_1H,                          0,                0,                          0,                          0,-lambdaL-t2_2M_1H,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                          t1_1M1H_1M,                                                                                   0,                          0,                0,                          0,                          0,                0,-lambdaL-t1_1M1H_1M-t12_1M1H_1M,        t12_1M1H_1M,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                          t2_1M1H_1M,                                                                                   0,                          0,                0,                          0,                          0,                0,                              0,-lambdaL-t2_1M1H_1M,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                                   0,                                                                          t1_1M1H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,-lambdaL-t1_1M1H_1H-t12_1M1H_1H,        t12_1M1H_1H,                          0,                          0,                0,                          0,                0],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                                   0,                                                                          t2_1M1H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,-lambdaL-t2_1M1H_1H,                          0,                          0,                0,                          0,                0],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                            t0_2H_1M,                                                                                   0,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,-lambdaL-t0_2H_1M-t01_2H_1M,                  t01_2H_1M,                0,                          0,                0],                                  
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                            t1_2H_1M,                                                                                   0,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,-lambdaL-t1_2H_1M-t12_2H_1M,        t12_2H_1M,                          0,                0],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                            t2_2H_1M,                                                                                   0,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,-lambdaL-t2_2H_1M,                          0,                0],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                                   0,                                                                            t1_2H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,-lambdaL-t1_2H_1H-t12_2H_1H,        t12_2H_1H],
                                  [  0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                           0,                                                                                   0,                                                                            t2_2H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,-lambdaL-t2_2H_1H]])
        self._B01 = numpy.array ([[      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0],
                                  [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL]])
        self._B10 = numpy.array ([[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2*muL,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0, muL,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0, muL, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])        
        
        
        ## Repeating starts at level 2
        ## In paper, A0 = B; A1 = L; A2 = F
        self._A0 = numpy.array ([[2*muL,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0, muL,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0, muL, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [    0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self._A1 = numpy.array ([[-lambdaH-lambdaM-lambdaL-2*muL,                                                                             lambdaM,                                                                             lambdaH,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                 [                           muM,-lambdaM*p_2M_1M-lambdaM*p_2M_1H-lambdaH*p_1M1H_1M-lambdaH*p_1M1H_1H-lambdaL-muL-muM,                                                                                   0,            lambdaM*p_2M_1M,                0,            lambdaM*p_2M_1H,                          0,                0,              lambdaH*p_1M1H_1M,                  0,              lambdaH*p_1M1H_1H,                  0,                          0,                          0,                0,                          0,                0],
                                 [                           muH,                                                                                   0,-lambdaM*p_1M1H_1M-lambdaM*p_1M1H_1H-lambdaH*p_2H_1M-lambdaH*p_2H_1H-lambdaL-muL-muH,                          0,                0,                          0,                          0,                0,              lambdaM*p_1M1H_1M,                  0,              lambdaM*p_1M1H_1H,                  0,            lambdaH*p_2H_1M,                          0,                0,            lambdaH*p_2H_1H,                0],
                                 [                             0,                                                                            t1_2M_1M,                                                                                   0,-lambdaL-t1_2M_1M-t12_2M_1M,        t12_2M_1M,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                 [                             0,                                                                            t2_2M_1M,                                                                                   0,                          0,-lambdaL-t2_2M_1M,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                 [                             0,                                                                                   0,                                                                            t0_2M_1H,                          0,                0,-lambdaL-t0_2M_1H-t01_2M_1H,                  t01_2M_1H,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                 [                             0,                                                                                   0,                                                                            t1_2M_1H,                          0,                0,                          0,-lambdaL-t1_2M_1H-t12_2M_1H,        t12_2M_1H,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                 [                             0,                                                                                   0,                                                                            t2_2M_1H,                          0,                0,                          0,                          0,-lambdaL-t2_2M_1H,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                 [                             0,                                                                          t1_1M1H_1M,                                                                                   0,                          0,                0,                          0,                          0,                0,-lambdaL-t1_1M1H_1M-t12_1M1H_1M,        t12_1M1H_1M,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                 [                             0,                                                                          t2_1M1H_1M,                                                                                   0,                          0,                0,                          0,                          0,                0,                              0,-lambdaL-t2_1M1H_1M,                              0,                  0,                          0,                          0,                0,                          0,                0],
                                 [                             0,                                                                                   0,                                                                          t1_1M1H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,-lambdaL-t1_1M1H_1H-t12_1M1H_1H,        t12_1M1H_1H,                          0,                          0,                0,                          0,                0],
                                 [                             0,                                                                                   0,                                                                          t2_1M1H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,-lambdaL-t2_1M1H_1H,                          0,                          0,                0,                          0,                0],
                                 [                             0,                                                                            t0_2H_1M,                                                                                   0,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,-lambdaL-t0_2H_1M-t01_2H_1M,                  t01_2H_1M,                0,                          0,                0],                                 
                                 [                             0,                                                                            t1_2H_1M,                                                                                   0,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,-lambdaL-t1_2H_1M-t12_2H_1M,        t12_2H_1M,                          0,                0],
                                 [                             0,                                                                            t2_2H_1M,                                                                                   0,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,-lambdaL-t2_2H_1M,                          0,                0],
                                 [                             0,                                                                                   0,                                                                            t1_2H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,-lambdaL-t1_2H_1H-t12_2H_1H,        t12_2H_1H],
                                 [                             0,                                                                                   0,                                                                            t2_2H_1H,                          0,                0,                          0,                          0,                0,                              0,                  0,                              0,                  0,                          0,                          0,                0,                          0,-lambdaL-t2_2H_1H]])
        self._A2 = numpy.array ([[lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                 [      0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                 [      0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                 [      0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                 [      0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                 [      0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                 [      0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                 [      0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0,       0],
                                 [      0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0,       0],
                                 [      0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0,       0],
                                 [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0,       0],
                                 [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0,       0],
                                 [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0,       0],
                                 [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0,       0],
                                 [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0,       0],
                                 [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL,       0],
                                 [      0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0, lambdaL]])

    def form_DR_M_M_1_matrix (self, lambdaL, lambdaH, muL, muH, t1, t12, t2):
        
        self._B00 = numpy.array ([[-lambdaH-lambdaL,          lambdaH,          0],
                                  [              t1, -(lambdaL+t1+t12),        t12],
                                  [              t2,                 0, -t2-lambdaL]])
        self._B01 = numpy.array ([[lambdaL,       0,       0],
                                  [      0, lambdaL,       0],
                                  [      0,       0, lambdaL]])
        self._B10 = numpy.array ([[muL, 0, 0],
                                  [  0, 0, 0],
                                  [  0, 0, 0]])
        
        ## In paper, A0 = B; A1 = L; A2 = F
        self._A0 = self._B10
        self._A1 = numpy.array ([[-lambdaH-lambdaL-muL,          lambdaH,          0],
                                 [                  t1, -(lambdaL+t1+t12),        t12],
                                 [                  t2,                 0, -t2-lambdaL]])
        self._A2 = self._B01

    def form_DR_M_M_2_matrix (self, lambdaL, lambdaH, muL, muH, t1, t12, t2):
        
        self._B00 = numpy.array ([[    -lambdaH-lambdaL,              lambdaH,               0,           0,              lambdaL,                        0,               0,               0],
                                  [                 muH, -lambdaH-lambdaL-muH,         lambdaH,           0,                    0,                  lambdaL,               0,               0],
                                  [                   0,                   t1, -lambdaL-t1-t12,         t12,                    0,                        0,         lambdaL,               0],
                                  [                   0,                   t2,               0, -t2-lambdaL,                    0,                        0,               0,         lambdaL],
                                  
                                  [                 muL,                    0,               0,           0, -muL-lambdaH-lambdaL,                  lambdaH,               0,               0],
                                  [                   0,                  muL,               0,           0,                  muH, -muL-lambdaH-muH-lambdaL,         lambdaH,               0],
                                  [                   0,                    0,               0,           0,                    0,                       t1, -t1-t12-lambdaL,             t12],
                                  [                   0,                    0,               0,           0,                    0,                       t2,               0,     -t2-lambdaL]])
        self._B01 = numpy.array ([[      0,       0,       0,       0],
                                  [      0,       0,       0,       0],
                                  [      0,       0,       0,       0],
                                  [      0,       0,       0,       0],
                                  [lambdaL,       0,       0,       0],
                                  [      0, lambdaL,       0,       0],
                                  [      0,       0, lambdaL,       0],
                                  [      0,       0,       0, lambdaL]])
        self._B10 = numpy.array ([[0, 0, 0, 0, 2*muL,   0, 0, 0],
                                  [0, 0, 0, 0,     0, muL, 0, 0],
                                  [0, 0, 0, 0,     0,   0, 0, 0],
                                  [0, 0, 0, 0,     0,   0, 0, 0]])
        
        ## In paper, A0 = B; A1 = L; A2 = F
        self._A0 = numpy.array ([[2*muL,   0, 0, 0],
                                 [    0, muL, 0, 0],
                                 [    0,   0, 0, 0],
                                 [    0,   0, 0, 0]])
        self._A1 = numpy.array ([[-lambdaH-lambdaL-2*muL,                  lambdaH,               0,           0],
                                 [                   muH, -lambdaL-muL-muH-lambdaH,         lambdaH,           0],
                                 [                     0,                       t1, -lambdaL-t1-t12,         t12],
                                 [                     0,                       t2,               0, -t2-lambdaL]])
        self._A2 = numpy.array ([[lambdaL,       0,       0,       0],
                                 [      0, lambdaL,       0,       0],
                                 [      0,       0, lambdaL,       0],
                                 [      0,       0,       0, lambdaL]])

    def form_DR_M_M_3_matrix (self, lambdaL, lambdaH, muL, muH, t1, t12, t2):
        
        self._B00 = numpy.array ([[-lambdaH-lambdaL,             lambdaH,                     0,              0,          0,             lambdaL,                       0,                         0,              0,          0,                     0,                         0,                         0,              0,          0],
                                  [             muH,-lambdaH-lambdaL-muH,               lambdaH,              0,          0,                   0,                 lambdaL,                         0,              0,          0,                     0,                         0,                         0,              0,          0],
                                  [               0,               2*muH,-lambdaH-lambdaL-2*muH,        lambdaH,          0,                   0,                       0,                   lambdaL,              0,          0,                     0,                         0,                         0,              0,          0],
                                  [               0,                   0,                    t1,-lambdaL-t1-t12,        t12,                   0,                       0,                         0,        lambdaL,          0,                     0,                         0,                         0,              0,          0],
                                  [               0,                   0,                    t2,              0,-t2-lambdaL,                   0,                       0,                         0,              0,    lambdaL,                     0,                         0,                         0,              0,          0],
                                  
                                  [             muL,                   0,                     0,              0,          0,-muL-lambdaH-lambdaL,                 lambdaH,                         0,              0,          0,               lambdaL,                         0,                        0,              0,          0],
                                  [               0,                 muL,                     0,              0,          0,                 muH,-muL-lambdaH-muH-lambdaL,                   lambdaH,              0,          0,                     0,                   lambdaL,                         0,              0,          0],
                                  [               0,                   0,                   muL,              0,          0,                   0,                   2*muH,-muL-lambdaH-2*muH-lambdaL,        lambdaH,          0,                     0,                         0,                   lambdaL,              0,          0],
                                  [               0,                   0,                     0,              0,          0,                   0,                       0,                        t1,-t1-t12-lambdaL,        t12,                     0,                         0,                         0,        lambdaL,          0],
                                  [               0,                   0,                     0,              0,          0,                   0,                       0,                        t2,              0,-t2-lambdaL,                     0,                         0,                         0,              0,    lambdaL],
                                  
                                  
                                  [               0,                   0,                     0,              0,          0,               2*muL,                       0,                         0,              0,          0,-2*muL-lambdaH-lambdaL,                   lambdaH,                         0,              0,          0],
                                  [               0,                   0,                     0,              0,          0,                   0,                   2*muL,                         0,              0,          0,                   muH,-2*muL-lambdaH-muH-lambdaL,                   lambdaH,              0,          0],
                                  [               0,                   0,                     0,              0,          0,                   0,                       0,                       muL,              0,          0,                     0,                     2*muH,-muL-lambdaH-2*muH-lambdaL,        lambdaH,          0],
                                  [               0,                   0,                     0,              0,          0,                   0,                       0,                         0,              0,          0,                     0,                         0,                        t1,-t1-t12-lambdaL,        t12],
                                  [               0,                   0,                     0,              0,          0,                   0,                       0,                         0,              0,          0,                     0,                         0,                        t2,              0,-t2-lambdaL]])
        self._B01 = numpy.array ([[      0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0],
                                  [      0,       0,       0,       0,       0],                                  
                                  [lambdaL,       0,       0,       0,       0],
                                  [      0, lambdaL,       0,       0,       0],
                                  [      0,       0, lambdaL,       0,       0],
                                  [      0,       0,       0, lambdaL,       0],
                                  [      0,       0,       0,       0, lambdaL]])
        self._B10 = numpy.array ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3*muL,     0,   0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0, 2*muL,   0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,     0, muL, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,     0,   0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     0,     0,   0, 0, 0]])
        
        ## In paper, A0 = B; A1 = L; A2 = F
        self._A0 = numpy.array ([[3*muL,     0,   0, 0, 0],
                                 [    0, 2*muL,   0, 0, 0],
                                 [    0,     0, muL, 0, 0],
                                 [    0,     0,   0, 0, 0],
                                 [    0,     0,   0, 0, 0]])
        self._A1 = numpy.array ([[-lambdaH-lambdaL-3*muL,                    lambdaH,                         0,               0,           0],
                                 [                   muH, -lambdaL-2*muL-muH-lambdaH,                   lambdaH,               0,           0],
                                 [                     0,                      2*muH,-lambdaL-muL-2*muH-lambdaH,         lambdaH,           0],
                                 [                     0,                          0,                        t1, -lambdaL-t1-t12,         t12],
                                 [                     0,                          0,                        t2,               0, -t2-lambdaL]])
        self._A2 = numpy.array ([[lambdaL,       0,       0,       0,       0],
                                 [      0, lambdaL,       0,       0,       0],
                                 [      0,       0, lambdaL,       0,       0],
                                 [      0,       0,       0, lambdaL,       0],
                                 [      0,       0,       0,       0, lambdaL]])
        
    def form_DR_M_M_s_matrix (self, s, lambdaL, lambdaH, muL, muH, t1, t12, t2):
        
        ## In paper, A0 = B; A1 = L; A2 = F
        shape = (s+2, s+2)
        
        #  A0 i.e. B 
        A0 = numpy.identity (shape[0])
        for i in range (shape[0]):
            A0[i][i] = 0 if i in [shape[0]-1, shape[0]-2] else (s-i)*muL
        self._A0 = A0
        #  A2 i.e. F
        self._A2 = numpy.identity (shape[0])*lambdaL
        #  A1 i.e. L
        T = numpy.array ([[-t1-t12, t12],
                          [      0, -t2]])
        t = -numpy.sum (T, axis=1)
        QB = []
        for i in range (s):
            array = numpy.zeros (s+2)
            array[i+1] = lambdaH
            array[i] = -lambdaH
            if i > 0:
                array[i-1] = i*muH
                array[i] -= i*muH
            QB.append (array)
        last2rows = numpy.hstack ([numpy.zeros ((2, shape[0]-3)),
                                   numpy.vstack ([t, T.T]).T])
        QB = numpy.vstack ([QB, last2rows])
        self._A1 = QB - self._A0 - self._A2
        
        ## At Boundary
        #  B01
        self._B01 = numpy.vstack ([numpy.zeros (shape) for i in range (s-1)] + [self._A2])
        #  B10
        self._B10 = numpy.hstack ([numpy.zeros (shape) for i in range (s-1)] + [self._A0])
        #  B00
        B00 = []
        for i in range (s):
            L = QB - self._A2 if i == 0 else self._A1
            # if s = 1, L0 is B00. No need to stack matrices
            if s==1:
                B00 = L
                break 
            # First row block: L, F, 0, 0, 0, ...
            if i == 0:
                rowblock = numpy.hstack ([L, self._A2] + [numpy.zeros (shape) for i in range (s-2)])
                B00.append (rowblock)
                continue
            # Last row block: 0, 0, 0, ..., B, L
            if i == s-1:
                B = numpy.identity (shape[0]) 
                for r, row in enumerate (B):
                    B[r][r] = 0 if r in [shape[0]-1, shape[0]-2] else min (s-r, i)*muL
                rowblock = numpy.hstack ([numpy.zeros (shape) for i in range (s-2)] + [B, L])
                B00.append (rowblock)
                continue
            
            # Anything in between first and last: 0, 0, ..., 0, B, L, F, 0, ..., 0, 0
            B = numpy.identity (shape[0]) 
            for r, row in enumerate (B):
                B[r][r] = 0 if r in [shape[0]-1, shape[0]-2] else min (s-r, i)*muL
            rowblock = numpy.hstack ([numpy.zeros (shape) for i in range (i-1)] + [B, L, self._A2] +
                                     [numpy.zeros (shape) for i in range (s-3-(i-1))])
            B00.append (rowblock)
        # negated sum
        B00 =  numpy.vstack (B00)
        firstblock = self._negated_sum(numpy.hstack ([B00, self._B01]))
        for r, row in enumerate (B00):
            B00[r][r] = firstblock[r][r]
        self._B00= B00
 
    def _get_R (self):
        #  R is found iteratively
        #     R_(k+1) = -V - R^2_(k) W
        #  where V = A2 inv(A1)
        #        W = A0 inv(A1)
        #  and initial R_0 = 0
        # ** R is non-decreasing i.e. its values will only 
        #    grow until they converge.
        V = numpy.dot (self._A2, inv(self._A1))
        W = numpy.dot (self._A0, inv(self._A1))
        R = numpy.zeros_like (V)
        for niter in range (1000):
            R = -V - numpy.dot (matrix_power (R, 2), W)
        
        return R

    def _solve_boundary_solutions (self, R):
        bound = self._combine_matrix (self._B00, self._B01, self._B10,
                                      self._A1 + numpy.dot (R, self._A0))
        pi_bound = self._solve_LH_vector (bound)
        return pi_bound[:self._B01.shape[0]], pi_bound[self._B01.shape[0]:]

    def _find_norm (self, pi0, pi1, R):
        alpha = sum (pi0.T)
        try:
            alpha += numpy.dot (pi1, sum (inv (numpy.identity (R.shape[0])-R).T))
        except numpy.linalg.LinAlgError:
            for i in range (R.shape[0]):
                if R[i][i] == 1:  R[i][i] = 0.99999
            alpha += numpy.dot (pi1, sum (inv (numpy.identity (R.shape[0])-R).T))
        return alpha
    
    def solve_prob_distributions (self, n=1000):
    
        self._is_valid_system()

        R = self._get_R()
        pi0, pi1 = self._solve_boundary_solutions (R)

        # Normalize boundary states
        alpha = self._find_norm (pi0, pi1, R)
        pi0 /= alpha
        pi1 /= alpha

        pis = [pi0, pi1]
        ## If boundary condition has more states than repeating matrix
        if len (pi0) > self._A1.shape[0]:
            nstats = len (pi0)/self._A1.shape[0]
            pis = numpy.split (pi0, nstats) + [pi1]
            
        ## If M/H2C2/s, only 1 (0) for pi0. 
        if self._is_MH2C2:
            indices = []
            nstats = self._nRad
            for i in range (nstats):
                nCustomer = i
                nsubstats = numpy.math.factorial  (4+nCustomer-1) / (numpy.math.factorial  (nCustomer)*numpy.math.factorial  (4-1))
                index = int (nsubstats)
                if i > 0: index += 1
                indices.append (index)
            pis = numpy.split (pi0, indices[:-1]) + [pi1]

        ## If M/C2/s, only 1 (0) for pi0. 
        if self._is_MC2:
            indices = []
            nstats = self._nRad
            for i in range (nstats):
                nCustomer = i
                nsubstats = numpy.math.factorial  (2+nCustomer-1) / (numpy.math.factorial  (nCustomer)*numpy.math.factorial  (2-1))
                index = int (nsubstats)
                if i > 0: index += 1
                indices.append (index)
            pis = numpy.split (pi0, indices[:-1]) + [pi1]

        for i in range (n-len (pis)):
            pii = numpy.dot (pis[-1], R)
            pis.append (pii)
        
        self._R = R
        self._pis = pis
        
        return pis

class MG1_calculator (stateProb_calculator):

    def __init__ (self):    
        super(stateProb_calculator, self).__init__ ()
        
        self._nRad = None
        self._l_hat = None
        self._gamma = None
        self._Zs = None
        
        self._A3, self._B02 = None, None
        
        self._G = None
        self._G_nonrep = None
        self._G_nonrep2 = None
        self._Astar1, self._Astar2, self._Astar3 = None, None, None
        self._Bstar01, self._Bstar02 = None, None

    @property
    def A3 (self): return self._A3
    @A3.setter
    def A3 (self, A3):
        if not self._matrix_is_square (A3):
            print ('Please provide a square matrix for A3.')
            return
        self._A3 = A3 

    @property
    def B02 (self): return self._B02
    @B02.setter
    def B02 (self, B02): self._B02 = B02

    @property
    def G (self): return self._G

    @property
    def Zs (self): return self._Zs

    @property
    def Astar1 (self): return self._Astar1

    @property
    def Astar2 (self): return self._Astar2

    @property
    def Astar3 (self): return self._Astar3    

    @property
    def Bstar01 (self): return self._Bstar01

    @property
    def Bstar02 (self): return self._Bstar02    

    def form_DR_M_M_1_matrix (self, lambdaL, lambdaH, muL, muH, t1, t12, t2):
        ## This function is different from the same function in traditional_calculator.
        ## This MG1_calculator is used when having m priority classes and/or non-
        ## exponential service process in priority classes.
        
        ## In this case, we keep track of level l-1
        ##  l - 1 = #_middle + Min (#_server, #_high)
        ## The input transition rates are used to formulate the transition probability,
        ## which are used to calculate G. The transition rates are also used to get
        ## the first 3 moments of the forward, local, and backward matrices in order
        ## to obtain the moments of G. 
        
        self._nRad = 1
        self._l_hat = 2
        
        B00 = numpy.array ([[-lambdaH-lambdaL]])
        self._B00 = numpy.array ([[0]])
        
        B01 = numpy.array ([[lambdaL, lambdaH, 0]])
        gamma = -B00[0][0]
        self._B01 = B01 / gamma
        
        B10 = numpy.array ([[muL],
                            [ t1],
                            [ t2]])
        gamma = numpy.array ([[muL+lambdaH+lambdaL], [t1+t12+lambdaL], [t2+lambdaL]])
        self._B10 = B10 / gamma
        
        ## In paper, A0 = B; A1 = L; A2 = F
        A0 = numpy.array ([[muL, 0, 0],
                           [ t1, 0, 0],
                           [ t2, 0, 0]])
        A1 = numpy.array ([[-lambdaH-lambdaL-muL,               0,            0],
                           [                   0, -lambdaL-t1-t12,          t12],
                           [                   0,                0, -t2-lambdaL]])
        A2 = numpy.array ([[lambdaL, lambdaH,       0],
                           [      0, lambdaL,       0],
                           [      0,       0, lambdaL]])
        gamma = numpy.array ([[muL+lambdaH+lambdaL], [t1+t12+lambdaL], [t2+lambdaL]])
        self._gamma = gamma
        self._A0 = A0 / gamma
        self._A1 = A1 / gamma
        self._A2 = A2 / gamma
        
        self._A1[0][0] += 1
        self._A1[1][1] += 1
        self._A1[2][2] += 1

    def form_DR_M_M_2_matrix (self, lambdaL, lambdaH, muL, muH, t1, t12, t2):
        ## This function is different from the same function in traditional_calculator.
        ## This MG1_calculator is used when having m priority classes and/or non-
        ## exponential service process in priority classes.
        
        ## In this case, we keep track of level l-1
        ##  l - 1 = #_middle + Min (#_server, #_high)
        ## The input transition rates are used to formulate the transition probability,
        ## which are used to calculate G. The transition rates are also used to get
        ## the first 3 moments of the forward, local, and backward matrices in order
        ## to obtain the moments of G. 
        
        self._nRad = 2
        self._l_hat = 4
        
        ## L = 1: (0, 0)
        B00 = numpy.array ([[-lambdaH-lambdaL]])
        self._B00 = numpy.array ([[0]])
        
        B01 = numpy.array ([[lambdaL, lambdaH]])
        gamma = -B00[0][0]
        self._B01 = B01 / gamma
        
        self._gamma_1 = gamma

        ## L = 2: (0, 1), (1, 0)
        B10 = numpy.array ([[muL], [muH]])
        gamma = numpy.array ([[muL+lambdaH+lambdaL], [muH+lambdaH+lambdaL]])
        self._B10 = B10 / gamma
        
        B11 = numpy.array ([[-gamma[0][0],            0],
                            [           0, -gamma[1][0]]])
        self._B11 = B11 / gamma + numpy.identity (B11.shape[0])
        
        B12 = numpy.array ([[lambdaL, lambdaH,       0, 0],
                            [      0, lambdaL, lambdaH, 0]])
        self._B12 = B12 / gamma
        
        self._gamma_2 = gamma
        
        ## L = 3: (0, 2), (1, 1), (2+, 0), (x, 0)
        B20 = numpy.array ([[2*muL,   0],
                            [  muH, muL],
                            [    0,  t1],
                            [    0,  t2]])
        gamma = numpy.array ([[2*muL+lambdaH+lambdaL], [muH+muL+lambdaH+lambdaL], [t1+t12+lambdaL], [t2+lambdaL]])
        self._B20 = B20 / gamma
        
        B21 = numpy.array ([[-gamma[0][0],            0,            0,            0],
                            [           0, -gamma[1][0],            0,            0],
                            [           0,            0, -gamma[2][0],          t12],
                            [           0,            0,            0, -gamma[3][0]]])
        self._B21 = B21 / gamma + numpy.identity (B21.shape[0])
        
        B22 = numpy.array ([[lambdaL, lambdaH,       0,       0],
                            [      0, lambdaL, lambdaH,       0],
                            [      0,       0, lambdaL,       0],
                            [      0,       0,       0, lambdaL]])
        self._B22 = B22 / gamma
        
        self._gamma_3 = gamma
        
        ## L = 4 - start repeating: (0, 3), (1, 2), (2+, 1), (x, 1)
        ## In paper, A0 = B; A1 = L; A2 = F
        self._gamma = gamma
        
        A0 = numpy.array ([[2*muL,   0, 0, 0],
                           [  muH, muL, 0, 0],
                           [    0,  t1, 0, 0],
                           [    0,  t2, 0, 0]])
        A1 = numpy.array ([[-lambdaH-lambdaL-2*muL,                        0,               0,           0],
                           [                     0, -lambdaH-lambdaL-muH-muL,               0,           0],
                           [                     0,                        0, -lambdaL-t1-t12,         t12],
                           [                     0,                        0,               0, -t2-lambdaL]])
        A2 = B22
        
        self._A0 = A0 / gamma
        self._A1 = A1 / gamma + numpy.identity (A1.shape[0])
        self._A2 = A2 / gamma
    
    def _get_G (self):
        #  G is found iteratively
        #     G_(k+1) = sum_i=0^inf Ai G^i_k
        #  and initial G_0 = 0
        # ** G is non-decreasing i.e. its values will only 
        #    grow until they converge.
        G = numpy.zeros_like (self._A0)
        for niter in range (1000):
            G = self._A0 + numpy.dot (self._A1, G) + \
                numpy.dot (self._A2, matrix_power (G, 2))
        return G

    def _get_A_moments (self, r):
        
        P = numpy.identity (self._A0.shape[0])
        for nrow in range (self._A0.shape[0]):
            P[nrow][nrow] = numpy.math.factorial (r)/numpy.power (self._gamma[nrow][0], r)
        A0 = numpy.dot (P, self._A0)
        A1 = numpy.dot (P, self._A1)
        A2 = numpy.dot (P, self._A2)
        return A0, A1, A2

    def _get_G_1 (self, G, A0_1, A1_1, A2_1):
        
        G1 = numpy.zeros_like (A0_1)
        for niter in range (1000):
            G1 = A0_1 + numpy.dot (A1_1, G) + numpy.dot (self._A1, G1) + \
                 numpy.dot (A2_1, matrix_power (G, 2)) + \
                 numpy.dot (self._A2, numpy.dot (G1, G)) + \
                 numpy.dot (self._A2, numpy.dot (G, G1))        
        return G1
    
    def _get_G_2 (self, G, G1, A0_1, A1_1, A2_1, A0_2, A1_2, A2_2):
        
        G2 = numpy.zeros_like (A0_2)
        for niter in range (1000):
            G2 = A0_2 + numpy.dot (A1_2, G) + 2*numpy.dot (A1_1, G1) + \
                 numpy.dot (self._A1, G2) + numpy.dot (A2_2, matrix_power (G, 2)) + \
                 2*numpy.dot (A2_1, numpy.dot (G1, G) + numpy.dot (G, G1)) + \
                 numpy.dot (self._A2, numpy.dot (G2, G) + 2*numpy.dot (G1, G1) + numpy.dot (G, G2))      
        return G2

    def _get_G_3 (self, G, G1, G2, A0_1, A1_1, A2_1, A0_2, A1_2, A2_2, A0_3, A1_3, A2_3):
        
        G3 = numpy.zeros_like (A0_3)
        for niter in range (1000):
            G3 = A0_3 + numpy.dot (A1_3, G) + 3*numpy.dot (A1_2, G1) + 3*numpy.dot (A1_1, G2) + \
                 numpy.dot (self._A1, G3) + numpy.dot (A2_3, matrix_power (G, 2)) + \
                 3*numpy.dot (A2_2, numpy.dot (G1, G) + numpy.dot (G, G1)) + \
                 3*numpy.dot (A2_1, numpy.dot (G2, G) + 2*numpy.dot (G1, G1) + numpy.dot (G, G2)) + \
                 numpy.dot (self._A2, numpy.dot (G3, G) + 3*numpy.dot (G2, G1) + 3*numpy.dot (G1, G2) + numpy.dot (G, G3))      
        return G3

    def _get_G_moments (self, G):

        A0_1, A1_1, A2_1 = self._get_A_moments (1)
        A0_2, A1_2, A2_2 = self._get_A_moments (2)
        A0_3, A1_3, A2_3 = self._get_A_moments (3)

        G1 = self._get_G_1 (G, A0_1, A1_1, A2_1)
        G2 = self._get_G_2 (G, G1, A0_1, A1_1, A2_1, A0_2, A1_2, A2_2)
        G3 = self._get_G_3 (G, G1, G2, A0_1, A1_1, A2_1, A0_2, A1_2, A2_2, A0_3, A1_3, A2_3)

        return G1, G2, G3

    def _get_G_nonrepeating (self, G_rep, repeat=False):
        #  G is found iteratively
        #     G_(k+1) = sum_i=0^inf Ai G^i_k
        #  and initial G_0 = 0
        # ** G is non-decreasing i.e. its values will only 
        #    grow until they converge.
        
        ## B10 for 1 radiologist only
        ## l = 2

        B0 = self._B20 if self._nRad == 2 or (self._l_hat == 3 and not repeat) else \
             self._B10 #if self._nRad == 1
        A1 = self._B11 if repeat else self._A1
        A2 = self._B12 if repeat else self._A2
        
        G = numpy.zeros_like (B0)
        for niter in range (1000):
            G = B0 + numpy.dot (A1, G) + \
                numpy.dot (A2, numpy.dot (G_rep, G))
        #for niter in range (1000):
        #    G = B0 + numpy.dot (self._A1, G) + \
        #        numpy.dot (self._A2, numpy.dot (G_rep, G))
        
        return G

    def _get_A_moments_nonrepeating (self, r, repeat=False):
        
        ## At least for exponential, the first non-repeating (one below repeating level)
        ## has the same L (i.e. A1) and F (i.e. A2). Only the B (i.e. A1) is different.
        ## Gamma values are also the same.
        
        B0 = self._B20 if self._nRad == 2 or (self._l_hat == 3 and not repeat) else \
             self._B10 #if self._nRad == 1
        A1 = self._B11 if repeat else self._A1
        A2 = self._B12 if repeat else self._A2
        gamma = self._gamma_2 if repeat else self._gamma              

        P = numpy.identity (A1.shape[0])
        for nrow in range (A1.shape[0]):
            P[nrow][nrow] = numpy.math.factorial (r)/numpy.power (gamma[nrow][0], r)
        A0 = numpy.dot (P, B0)
        A1 = numpy.dot (P, A1)
        A2 = numpy.dot (P, A2)
        
        #P = numpy.identity (self._A1.shape[0])
        #for nrow in range (self._A1.shape[0]):
        #    P[nrow][nrow] = numpy.math.factorial (r)/numpy.power (self._gamma[nrow][0], r)
        #A0 = numpy.dot (P, B0)
        #A1 = numpy.dot (P, self._A1)
        #A2 = numpy.dot (P, self._A2)
        return A0, A1, A2

    def _get_G_1_nonrepeating (self, G_nonrep, G, G1, A0_1, A1_1, A2_1, repeat=False):
        
        A1 = self._B11 if repeat else self._A1
        A2 = self._B12 if repeat else self._A2        
        
        G1_nonrep = numpy.zeros_like (A0_1)
        for niter in range (1000):
            G1_nonrep = A0_1 + numpy.dot (A1_1, G_nonrep) + numpy.dot (A1, G1_nonrep) + \
                        numpy.dot (A2_1, numpy.dot (G, G_nonrep)) + \
                        numpy.dot (A2, numpy.dot (G1, G_nonrep)) + \
                        numpy.dot (A2, numpy.dot (G, G1_nonrep))
        return G1_nonrep
    
    def _get_G_2_nonrepeating (self, G_nonrep, G1_nonrep, G, G1, G2, A0_1, A1_1, A2_1, A0_2, A1_2, A2_2, repeat=False):
        
        A1 = self._B11 if repeat else self._A1
        A2 = self._B12 if repeat else self._A2          
        
        G2_nonrep = numpy.zeros_like (A0_2)
        for niter in range (1000):
            G2_nonrep = A0_2 + numpy.dot (A1_2, G_nonrep) + 2*numpy.dot (A1_1, G1_nonrep) + \
                        numpy.dot (A1, G2_nonrep) + numpy.dot (A2_2, numpy.dot (G, G_nonrep)) + \
                        2*numpy.dot (A2_1, numpy.dot (G1, G_nonrep) + numpy.dot (G, G1_nonrep)) + \
                        numpy.dot (A2, numpy.dot (G2, G_nonrep) + 2*numpy.dot (G1, G1_nonrep) + numpy.dot (G, G2_nonrep))      
        return G2_nonrep

    def _get_G_3_nonrepeating (self, G_nonrep, G1_nonrep, G2_nonrep, G, G1, G2, G3, A0_1, A1_1, A2_1, A0_2, A1_2, A2_2, A0_3, A1_3, A2_3, repeat=False):
        
        A1 = self._B11 if repeat else self._A1
        A2 = self._B12 if repeat else self._A2          
        
        G3_nonrep = numpy.zeros_like (A0_3)
        for niter in range (1000):
            G3_nonrep = A0_3 + numpy.dot (A1_3, G_nonrep) + 3*numpy.dot (A1_2, G1_nonrep) + 3*numpy.dot (A1_1, G2_nonrep) + \
                        numpy.dot (A1, G3_nonrep) + numpy.dot (A2_3, numpy.dot (G, G_nonrep)) + \
                        3*numpy.dot (A2_2, numpy.dot (G1, G_nonrep) + numpy.dot (G, G1_nonrep)) + \
                        3*numpy.dot (A2_1, numpy.dot (G2, G_nonrep) + 2*numpy.dot (G1, G1_nonrep) + numpy.dot (G, G2_nonrep)) + \
                        numpy.dot (A2, numpy.dot (G3, G_nonrep) + 3*numpy.dot (G2, G1_nonrep) + 3*numpy.dot (G1, G2_nonrep) + numpy.dot (G, G3_nonrep))      
        return G3_nonrep

    def _get_G_moments_nonrepeating (self, G_nonrep, G, G1, G2, G3, repeat=False):

        ## l = 2 (repeating starts at l = 3)
        A0_1, A1_1, A2_1 = self._get_A_moments_nonrepeating (1, repeat=repeat)
        A0_2, A1_2, A2_2 = self._get_A_moments_nonrepeating (2, repeat=repeat)
        A0_3, A1_3, A2_3 = self._get_A_moments_nonrepeating (3, repeat=repeat)

        G1_nonrep = self._get_G_1_nonrepeating (G_nonrep, G, G1, A0_1, A1_1, A2_1, repeat=repeat)
        G2_nonrep = self._get_G_2_nonrepeating (G_nonrep, G1_nonrep, G, G1, G2, A0_1, A1_1, A2_1, A0_2, A1_2, A2_2, repeat=repeat)
        G3_nonrep = self._get_G_3_nonrepeating (G_nonrep, G1_nonrep, G2_nonrep, G, G1, G2, G3, A0_1, A1_1, A2_1, A0_2, A1_2, A2_2, A0_3, A1_3, A2_3, repeat=repeat)

        return G1_nonrep, G2_nonrep, G3_nonrep

    def get_Zs (self):
        
        ## Repeating parts
        G = self._get_G()
        self._G = G
        G1, G2, G3 = self._get_G_moments (G)
        
        ## Non-repeating parts - Only the first state below the repeating level
        ## e.g. if 1 radiologist, repeating level starts at 2. So, non-repeating parts return the Zs at 1
        ##      if 2 radiologist, repeating level starts at 4. So, non-repeating parts return the Zs at 3
        G_nonrep = self._get_G_nonrepeating (G)
        self._G_nonrep = G_nonrep
        G1_nonrep, G2_nonrep, G3_nonrep = self._get_G_moments_nonrepeating (G_nonrep, G, G1, G2, G3)
        Z1_nonrep = numpy.nan_to_num (G1_nonrep/G_nonrep)
        Z2_nonrep = numpy.nan_to_num (G2_nonrep/G_nonrep)
        Z3_nonrep = numpy.nan_to_num (G3_nonrep/G_nonrep)
        
        if self._l_hat == 3:
            G_nonrep2 = self._get_G_nonrepeating (G_nonrep, repeat=True)
            self._G_nonrep2 = G_nonrep2
            G1_nonrep, G2_nonrep, G3_nonrep = self._get_G_moments_nonrepeating (G_nonrep2, G_nonrep, G1_nonrep, G2_nonrep, G3_nonrep, repeat=True)
            Z1_nonrep = numpy.nan_to_num (G1_nonrep/G_nonrep2)
            Z2_nonrep = numpy.nan_to_num (G2_nonrep/G_nonrep2)
            Z3_nonrep = numpy.nan_to_num (G3_nonrep/G_nonrep2)        
        
        self._Zs = {'Z1':Z1_nonrep, 'Z2':Z2_nonrep, 'Z3':Z3_nonrep}

    def _get_stars (self, G):

        ## 1. A stars
        Astar1 = numpy.identity (G.shape[0]) - (self._A1 + numpy.dot (self._A2, G))
        Astar2 = - (self._A2)
        Astar3 = None
        if self._A3 is not None:
            Astar1 += - numpy.dot (self._A3, matrix_power (G, 2))
            Astar2 += - numpy.dot (self._A3, matrix_power (G, 1))
            Astar3 = - (self._A3)
        
        Bstar01 = self._B01
        Bstar02 = None
        if self._B02 is not None:
            Bstar01 += numpy.dot (self._B02, G)
            Bstar02 = self._B02

        return Astar1, Astar2, Astar3, Bstar01, Bstar02

    def _solve_boundary_solutions (self, Astar1, Bstar01):
        bound = numpy.identity (self._B00.shape[0]) - self._B00 - \
                numpy.dot (Bstar01, numpy.dot (inv (Astar1), self._B10))
        vecs = eig (bound, left=True, right=False)[1]
        for idx in range (bound.shape[0]):
            pi_bound = numpy.round (vecs[:, idx], 6)
            if not (pi_bound<0).any(): break
        return pi_bound ## Only pi0

    def _find_norm (self, pi0, Astar1, Astar2, Astar3, Bstar01, Bstar02):
        
        AstarSum = sum ([0 if a is None else a for a in [Astar1, Astar2, Astar3]])
        BstarSum = sum ([0 if b is None else b for b in [Bstar01, Bstar02]])
        
        alpha = numpy.sum (pi0) + \
                numpy.dot (pi0, numpy.dot (BstarSum, numpy.sum (inv (AstarSum), axis=1)))
        
        return alpha

    def solve_prob_distributions (self, n=100):
    
        self._is_valid_system()
        #if not self._check_ergodic():
        #    print ('This MC is not ergodic. Cannot solve for its pis.')
        #    return None   

        ## Repeating parts
        G = self._get_G()
        G1, G2, G3 = self._get_G_moments (G)
        Z1 = numpy.nan_to_num (G1/G)
        Z2 = numpy.nan_to_num (G2/G)
        Z3 = numpy.nan_to_num (G3/G)
        
        ## Non-repeating parts - For 1 radiologist only
        G_nonrep = self._get_G_nonrepeating (G)
        G1_nonrep, G2_nonrep, G3_nonrep = self._get_G_moments_nonrepeating (G_nonrep, G, G1, G2, G3)
        Z1_nonrep = numpy.nan_to_num (G1_nonrep/G_nonrep)
        Z2_nonrep = numpy.nan_to_num (G2_nonrep/G_nonrep)
        Z3_nonrep = numpy.nan_to_num (G3_nonrep/G_nonrep)
        
        self._Zs = {'Z1':Z1_nonrep, 'Z2':Z2_nonrep, 'Z3':Z3_nonrep}
        
        Astar1, Astar2, Astar3, Bstar01, Bstar02 = self._get_stars(G)
        pi0 = self._solve_boundary_solutions (Astar1, Bstar01)
        
        # Normalize boundary states
        alpha = self._find_norm (pi0, Astar1, Astar2, Astar3, Bstar01, Bstar02)
        pi0 /= alpha
    
        pis = [pi0]
        for i in range (n-1):
            if i == 0: # state = 1
                pii = numpy.dot (pis[-1], numpy.dot (Bstar01, inv (Astar1)))
            elif i == 1: # state = 2
                pii = -numpy.dot (pis[-1], numpy.dot (Astar2, inv (Astar1)))
                if not self._B02 is None:
                    pii += numpy.dot (pis[0], numpy.dot (Bstar02, inv (Astar1)))
            else:
                pii = -numpy.dot (pis[-1], numpy.dot (Astar2, inv (Astar1)))
                if not self._A3 is None:
                    pii += -numpy.dot (pis[-2], numpy.dot (Astar3, inv (Astar1)))
            pis.append (pii)
        
        self._G = G
        self._Astar1  = Astar1
        self._Astar2  = Astar2
        self._Astar3  = Astar3
        self._Bstar01 = Bstar01
        self._Bstar02 = Bstar02
        self._pis = pis
        
        return pis