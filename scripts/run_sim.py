
##
## By Elim Thompson 12/15/2020
##
## This is the main python script to simulate radiology reading workflow at a specific clinical
## setting with a CADt diagnostic performance. This simulation software handles a simplified
## scenario with 1 AI that is trained to identify 1 disease condition from 1 modality and
## anatomy. Patients in the reading queue either have the disease condition or not. User can
## either use argument flags like below or use `../inputs/config.dat` to specify user input values.
##
## $ python run_sim.py --configFile ../inputs/config.dat --verbose
##
## 05/08/2023
## ----------
## * Add in properties for multi-AI scenario
##
## 05/20/2024
## ----------
## * Incorporated Rucha's work for hierarchical queue
##
## 05/24/2024
## ----------
## * Incorporated Michelle's method for non-preemptive hierarchical queue
## * Extended it to non-preemptive priority queue
#######################################################################################################

################################
## Import packages
################################ 
import pickle, time, os, sys, cProfile, io, pstats, argparse, pandas
from datetime import datetime

sys.path.insert(0, os.getcwd()+'\\tools')
from tools import inputHandler, trialGenerator, plotter

################################
## Function
################################ 
def sim_theory_summary (trial_wdf, theory, diseaseGroups, aDiseaseTree):

    ''' Function to quickly print out wait-time and its difference
        between simulation and theory for different diseased populations.

        inputs
        ------
        trial_wdf (dataframe): wait-time dataframe from trialGenerator
        theory (dict): theoretical prediction from calculator.get_theory_waitTime()
        diseaseGroups (dict): user-input disease groups

        output
        ------
        results (dataframe): 4 columns: sim_waittime, theory_waittime,
                                        sim_delta = sim_waittime - sim_fifo
                                        theory_delta = theory_waittime - theory_fifo
                             for each disease condition for each queue type
                             (priority and hierarchical)
    '''

    diseases = [gp['diseaseNames'][i] for _, gp in diseaseGroups.items()
                for i in range (len (gp['diseaseNames']))]

    sim_wdf = trial_wdf[trial_wdf['trial_id']=='trial_000']
    wdf = sim_wdf[sim_wdf['is_interrupting']==False]

    adict = {'n_sim_pateints':[], 'sim_waittime':[], 'theory_waittime':[]}
    #rows = ['fifo_non-interrupting'] + \
    rows = ['priority '+ disease for disease in diseases] + \
            ['hierarchical '+ disease for disease in diseases]

    for row in rows:
        if 'fifo' in row:
            nvalue = len (wdf.fifo)
            simvalue = wdf.fifo.mean()
            theoryvalue = theory['fifo']['non-interrupting']
        else:
            qtype, disease = row.split()
            nvalue = len (wdf[wdf['disease_name']==disease][qtype])

            ## If want sim and theory values for true diseased groups:
            simvalue = wdf[wdf['disease_name']==disease][qtype].mean()
            theoryvalue = theory[qtype][disease] if qtype == 'priority' else \
                          theory[qtype][disease]['diseased']
            
            ## If want sim and theory values for AI-positive groups rather than true diseased,
            ## uncomment below.
            
            # for aGroup in aDiseaseTree.diseaseGroups:
            #     vendors_in_group = []
            #     for anAI in aGroup.AIs:
            #         vendors_in_group.append(anAI.AIname)
            #         if disease==anAI.targetDisease:
            #             vendorname = anAI.AIname
            #             condition = wdf[vendorname] == True
            #             # Create conditions for all other vendors being False
            #             for vendor in vendors_in_group:
            #                 if vendor != vendorname:
            #                     condition &= wdf[vendor] == False
            # simvalue = wdf[condition][qtype].mean()
            # theoryvalue =  theory[qtype][disease] if qtype == 'priority' else \
            #                theory[qtype][disease]['positive']

        adict['n_sim_pateints'].append (nvalue)            
        adict['sim_waittime'].append (simvalue)
        adict['theory_waittime'].append (theoryvalue)

    ## Add in sim_delta and theory_delta 
    adict['sim_delta'] = adict['sim_waittime'] - adict['sim_waittime'][0]
    adict['theory_delta'] = adict['theory_waittime'] - adict['theory_waittime'][0]
    adict['columns'] = rows

    return pandas.DataFrame (adict).set_index ('columns')

def generate_plots (params, trialGen):

    ''' Function to generate plots.

        inputs
        ------
        params (dict): dictionary with user inputs
        trialGen (trialGenerator): all results from simulations
    '''

    if params['isPreemptive']:
        outFile = params['plotPath'] + 'nPatientsDistributions.png'
        plotter.plot_n_patient_distributions (outFile, trialGen.n_patients_system_df, trialGen.n_patients_system_stats, params)

################################
## Script starts here!
################################ 
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("--configFile", dest='config_file', help="Path to the configuration file")
    args = parser.parse_args()
    configFile = args.config_file

    ## Gather user-specified settings
    params, AIs, aDiseaseTree = inputHandler.read_args(configFile)

    ## Track runtime profile if turned on
    pr = None
    if params['doRunTime']:
        pr = cProfile.Profile()
        pr.enable()

    ## Run trials
    t0 = time.time()
    trialGen = trialGenerator.trialGenerator ()
    trialGen.set_params (params, AIs)
    trialGen.simulate_trials (AIs, aDiseaseTree)
    params['runTimeMin'] = (time.time() - t0)/60 # minutes
    print ('{0} trials took {1:.2f} minutes.\n'.format (params['nTrials'], params['runTimeMin']))
    
    ## Results
    print ('Quick results from 1 trial')

    #trialGen._get_sim_probabilities(trialGen.waiting_times_df, aDiseaseTree, params)
    results = sim_theory_summary (trialGen.waiting_times_df, params['theory'], params['diseaseGroups'], aDiseaseTree)
    print (results)
    if params['doPlots']: generate_plots (params, trialGen)

    ## Gather data for dict
    data = {'params':params,
            'lpatients':trialGen.n_patients_system_df,
            'wpatients':trialGen.waiting_times_df,
            'lstats':trialGen.n_patients_system_stats,
            'wtstats':trialGen.waiting_times_stats}
    with open (params['statsFile'], 'wb') as f:
        pickle.dump (data, f)
    f.close()
    
    if params['doRunTime']:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
        ps.print_stats()

        with open (params['runTimeFile'], 'w+') as f:
            f.write (s.getvalue())
        f.close()