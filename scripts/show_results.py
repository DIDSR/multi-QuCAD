
##
## By Elim Thompson 11/10/2024
##
## This script includes function that displays results from a stats.p output file.
#######################################################################################################

################################
## Import packages
################################ 
import pandas, numpy
pandas.set_option('display.precision', 2)

################################
## Define constants
################################ 
queues = ['fifo', 'priority', 'hierarchical']

#######################################
## Function that shows theory results
#######################################
def get_theory_results (params, display=False):

    theory = params['theory']
    rankedDiseases = params['rankedDiseases']

    columns = ['queue_subgroup', 'wait_time_theory', 'time_diff_theory']
    summary = {key:[] for key in columns}

    for queue in queues:
        queuename = 'without CADt' if queue == 'fifo' else \
                    'with CADt (priority)' if queue == 'priority' else \
                    'with CADt (hierarchical)'
        if queue == 'fifo':
            summary['queue_subgroup'].append (queuename + ' ' + 'non-interrupting')
            summary['wait_time_theory'].append (theory[queue]['non-interrupting'])
            summary['time_diff_theory'].append (numpy.nan)
            fifo_wait_time = theory[queue]['non-interrupting']
            continue
        for disease in rankedDiseases:
            summary['queue_subgroup'].append (queuename + ' ' + disease)
            summary['wait_time_theory'].append (theory[queue][disease]['diseased'])
            summary['time_diff_theory'].append (theory[queue][disease]['diseased'] - fifo_wait_time)
        
    df = pandas.DataFrame (summary).set_index ('queue_subgroup')
    if display: print (df)
    return df

def get_sim_results (adict, display=False):

    wpatients = adict['wpatients']
    rankedDiseases = adict['params']['rankedDiseases']

    columns = ['queue_subgroup', 'wait_time_sim', 'time_diff_sim']
    summary = {key:[] for key in columns}

    for queue in queues:
        queuename = 'without CADt' if queue == 'fifo' else \
                    'with CADt (priority)' if queue == 'priority' else \
                    'with CADt (hierarchical)'
        if queue == 'fifo':
            summary['queue_subgroup'].append (queuename + ' ' + 'non-interrupting')
            subgroup = wpatients[~wpatients['is_interrupting']] 
            summary['wait_time_sim'].append (subgroup['fifo'].mean())
            summary['time_diff_sim'].append (numpy.nan)
            fifo_wait_time = subgroup['fifo'].mean()
            continue
        for disease in rankedDiseases:
            summary['queue_subgroup'].append (queuename + ' ' + disease)
            subgroup = wpatients[wpatients['disease_name']==disease]
            summary['wait_time_sim'].append (subgroup[queue].mean())
            summary['time_diff_sim'].append (subgroup[queue].mean() - fifo_wait_time)
        
    df = pandas.DataFrame (summary).set_index ('queue_subgroup')
    if display: print (df)
    return df

def display_results (adict):

    theory = get_theory_results (adict['params'], display=False)
    sim = get_sim_results (adict, display=False)

    merged = pandas.merge(sim, theory, left_index=True, right_index=True)
    print (merged)
