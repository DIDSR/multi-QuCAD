
##
## By Elim Thompson 12/15/2020
##
## This is the main python script to simulate radiology reading workflow at a specific clinical
## setting with a CADt diagnostic performance. This simulation software handles radiologist
## workflows with multiple disease conditions and CADt devices. Patients in the reading queue may
## have different disease conditions. User must provide `../inputs/config.dat` to define the clinical
## workflow.
##
## $ python run_sim.py --configFile ../inputs/config.dat
##
## Disclaimer
## ----------
## This software and documentation (the "Software") were developed at the Food and Drug
## Administration (FDA) by employees of the Federal Government in the course of their official
## duties. Pursuant to Title 17, Section 105 of the United States Code, this work is not subject
## to copyright protection and is in the public domain. Permission is hereby granted, free of
## charge, to any person obtaining a copy of the Software, to deal in the Software without
## restriction, including without limitation the rights to use, copy, modify, merge, publish,
## distribute, sublicense, or sell copies of the Software or derivatives, and to permit persons
## to whom the Software is furnished to do so. FDA assumes no responsibility whatsoever for use
## by other parties of the Software, its source code, documentation or compiled executables, and
## makes no guarantees, expressed or implied, about its quality, reliability, or any other
## characteristic. Further, use of this code in no way implies endorsement by the FDA or confers
## any advantage in regulatory decisions. Although this software can be redistributed and/or
## modified freely, we ask that any derivative works bear some notice that they are derived from
## it, and any modified versions bear some notice that they have been modified.
##
#######################################################################################################

################################
## Import packages
################################ 
import pickle, time, os, sys, cProfile, io, pstats, argparse, pandas, numpy
from datetime import datetime
import scipy.stats as stats

sys.path.insert(0, os.getcwd()+'\\tools')
from tools import inputHandler, trialGenerator, plotter, simulator

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

    adict = {'n_sim_pateints':[], 'sim_waittime':[], 'sim_waittime_low': [], 'sim_waittime_high': [], 'theory_waittime':[]}
    rows =  ['fifo_non-interrupting'] + \
            ['priority '+ disease for disease in diseases] + \
            ['hierarchical '+ disease for disease in diseases]

    for row in rows:
        if 'fifo' in row:
            nvalue = len (wdf.fifo)
            simvalue = wdf.fifo.mean()
            simvalue_low, simvalue_high = stats.t.interval(0.95, len(wdf.fifo)-1, loc=simvalue, scale=stats.sem(wdf.fifo))
            theoryvalue = theory['fifo']['non-interrupting']
        else:
            qtype, disease = row.split()
            nvalue = len (wdf[wdf['disease_name']==disease][qtype])

            ## If want sim and theory values for true diseased groups:
            simvalue = wdf[wdf['disease_name']==disease][qtype].mean()
            simvalue_low, simvalue_high = stats.t.interval(0.95, len(wdf[wdf['disease_name']==disease][qtype])-1, loc=simvalue, scale=stats.sem(wdf[wdf['disease_name']==disease][qtype]))
            theoryvalue = theory[qtype][disease]['diseased']
            
        adict['n_sim_pateints'].append (nvalue)            
        adict['sim_waittime'].append (simvalue)        
        adict['sim_waittime_low'].append (simvalue_low)
        adict['sim_waittime_high'].append (simvalue_high)
        adict['theory_waittime'].append (theoryvalue)

    ## Add in sim_delta and theory_delta 
    #adict['sim_delta'] = adict['sim_waittime'] - adict['sim_waittime'][0]
    #adict['theory_delta'] = adict['theory_waittime'] - adict['theory_waittime'][0]
    adict['columns'] = rows

    return pandas.DataFrame (adict).set_index ('columns')

def generate_plots (params, trialGen, results):

    ''' Function to generate plots.

        inputs
        ------
        params (dict): dictionary with user inputs
        trialGen (trialGenerator): all results from simulations
    '''

    if params['isPreemptive']:
        outFile = params['plotPath'] + 'nPatientsDistributions.png'
        plotter.plot_n_patient_distributions (outFile, trialGen.n_patients_system_df,
                                              trialGen.n_patients_system_stats, params)
   
    ## If do-plots, generate plots for one simulation
    if params['doPlots']:
        # Timing flow with first 200 cases for both with and withoutCADt
        diseases = [gp['diseaseNames'][i] for _, gp in params['diseaseGroups'].items()
                for i in range (len (gp['diseaseNames']))]

        for qtype in params['qtypes']:
            outFile = params['plotPath'] + 'patient_timings_' + qtype + '.pdf'
            #Plot patient timings
            plotter.plot_timing (outFile, trialGen.waiting_times_df,
                                 params['startTime'], params['hierDict'],
                                 params['AIinfo'], n=200, qtype=qtype)

        #Plot wait-time simulation and theory for each true-diseased group
        outFile = params['plotPath'] + 'sim_theory.pdf'
        plotter.plot_sim_theory_diseased (outFile, diseases, results)

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
    results = sim_theory_summary (trialGen.waiting_times_df, params['theory'], params['diseaseGroups'], aDiseaseTree)
    print (results)

    if params['doPlots']: generate_plots (params, trialGen, results)
            
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