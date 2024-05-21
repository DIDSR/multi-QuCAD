
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
## 05/20/2023
## ----------
## * Cleaned up for publishing multi-QuCAD 
#######################################################################################################

################################
## Import packages
################################ 
import numpy, pickle, time, os, sys, cProfile, io, pstats, argparse

sys.path.insert(0, os.getcwd()+'\\tools')
from tools import inputHandler, diseaseTree, AI, simulator, trialGenerator, plotter, hier

#import logging
#logging.basicConfig(level=logging.DEBUG)

################################
## Define lambdas
################################ 
get_n_positive_patients = lambda oneSim, qtype:len (oneSim.get_positive_records(qtype))
get_n_negative_patients = lambda oneSim, qtype:len (oneSim.get_negative_records(qtype))
get_n_interrupting_patients = lambda oneSim, qtype:len (oneSim.get_interrupting_records(qtype))

################################
## Define functions
################################ 
def create_disease_tree (diseaseGroups, meanServiceTimes, AIs):

    ## AIs should be already updated with user-set threshold 
    aDiseaseTree = diseaseTree.diseaseTree ()
    aDiseaseTree.build_diseaseTree (diseaseGroups, meanServiceTimes, AIs)

    return aDiseaseTree

def create_AI (AIname, AIinfo, doPlots=False, plotPath=None):

    ''' Function to create a CADt device either at a threshold or from 
        an input ROC file. If provided an ROC File, the file should have
        two columns. First is false positive fraction (FPF), and second
        is true positive fraction (TPF). Note that either FPFThresh or
        rocFile should be provided. If both are provided, use FPFThresh
        and ignore ROC file.

        inputs
        ------
        TPFThresh (float): CADt Se operating point to be used for simulation
        FPFThresh (float): CADt 1-Sp operating point to be used for simulation
        rocFile (str): File to ROC curve that will be parameterized
        doPlot (bool): If true, generate plots for ROC parameterization 
        plotPath (str): Path where plots generated will live

        output
        ------
        anAI (AI): CADt with a diagnostic performance from user input
    '''

    ## When FPF is provided, use a single operating point
    if AIinfo['FPFThresh'] is not None:
        return AI.AI.build_from_opThresh(AIname, AIinfo['groupName'], AIinfo['targetDisease'],
                                         AIinfo['TPFThresh'], 1-AIinfo['FPFThresh'])

    ## If FPF is not provided, but emperical ROC is provided, parameterize
    ## the ROC curve based on bi-normal distribution.
    anAI = AI.AI.build_from_empiricalROC (AIname, AIinfo['groupName'], AIinfo['targetDisease'],
                                          AIinfo['rocFile'], AIinfo['TPFThresh'])
    anAI.fit_ROC (doPlots=doPlots, outPath=plotPath)
    ## Make sure the CADt operates at the user-input Se Threshold 
    anAI.SeThresh = AIinfo['TPFThresh']
    return anAI

def print_sim_performance (oneSim, AIs, params):

    ## Check with group prob and disease prevalence
    patient_groups = numpy.array ([p.group_name for p in oneSim.get_noninterrupting_records('fifo')])
    for groupName, groupInfo in params['diseaseGroups'].items():
        
        # 1. check group prob
        groupProb = groupInfo['groupProb']
        patient_groupnames_in_this_group = patient_groups[patient_groups==groupName]
        g_sim = len (patient_groupnames_in_this_group) / len (patient_groups)
        print ('+-----------------------------------------------')
        print ('| Group {0} ({1}):'.format (groupName, len (patient_groupnames_in_this_group)))
        print ('|   * Group prob (theory, sim): {0:.3f}, {1:.3f}'.format (groupProb, g_sim))
        
        # 2. check if any disease that don't belong to this group
        patient_diseases = numpy.array ([p.disease_name for p in oneSim.get_noninterrupting_records('fifo')
                                         if p.group_name == groupName])
        disease_names = numpy.array (groupInfo['diseaseNames'] + ['non-diseased'])
        unexpected_patient_diseases = patient_diseases[~numpy.in1d (patient_diseases, disease_names)]
        if len (unexpected_patient_diseases) > 0: 
            print ('|   * Unexpected disease in this group: {0}'.format (unexpected_patient_diseases))
        
        # 3. check disease prob within this group
        disease_probs = numpy.array (groupInfo['diseaseProbs'] + [1-sum(groupInfo['diseaseProbs'])])
        for diseaseName, diseaseProb in zip (disease_names,  disease_probs):
            patient_diseasenames_in_this_group = patient_diseases[patient_diseases==diseaseName]
            d_sim = len (patient_diseasenames_in_this_group) / len (patient_diseases)
            print ('|   * Disease {0} ({1}):'.format (diseaseName, len (patient_diseasenames_in_this_group)))
            print ('|       -- Disease prob (theory, sim): {0:.3f}, {1:.3f}'.format (diseaseProb, d_sim))

    print ('+-----------------------------------------------\n')

    ## Check with AI performance
    for AIname, anAI in AIs.items():
        
        groupName = anAI.groupName
        targetDisease = anAI.targetDisease

        # Extract patients within the target group
        patients_in_gp = numpy.array ([p for p in oneSim.get_noninterrupting_records('fifo')
                                       if p.group_name == groupName])        
        # Divide patients into TP, FP, TN, FN based on target diseased
        TP = len ([p for p in patients_in_gp if p.disease_name==targetDisease and p.is_positives[AIname]])
        FP = len ([p for p in patients_in_gp if not p.disease_name==targetDisease and p.is_positives[AIname]])
        FN = len ([p for p in patients_in_gp if p.disease_name==targetDisease and not p.is_positives[AIname]])
        TN = len ([p for p in patients_in_gp if not p.disease_name==targetDisease and not p.is_positives[AIname]])        
    
        ppv_sim = 0 if TP + FP == 0 else TP / (TP + FP)
        npv_sim = 0 if TN + FN == 0 else TN / (TN + FN)
        
        print ('+-----------------------------------------------')
        print ('| AI {0} ({1}) in group {2}:'.format (AIname, targetDisease, groupName))
        print ('|    * PPV (theory, sim): {0:.4f}, {1:.4f}'.format (params['probs_ppv_npv']['ppv'][groupName][AIname], ppv_sim))
        print ('|    * NPV (theory, sim): {0:.4f}, {1:.4f}'.format (params['probs_ppv_npv']['npv'][groupName][AIname], npv_sim))
        
    print ('+-----------------------------------------------')
    
    #from hier.py
    print('hr: ', hier.df_hr_mean, hier.df_95_ci_hr)
    print('pr: ', hier.df_pr_mean, hier.df_95_ci_pr)
    print('fifo: ', hier.df_fifo_mean, hier.df_95_ci_fifo)
    print('theory: ', hier.df_theory)

################################
## Script starts here!
################################ 
if __name__ == '__main__':

    verbose = True
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("--configFile", dest='config_file', help="Path to the configuration file")
    args = parser.parse_args()
    configFile = args.config_file

    ## Gather user-specified settings
    params = inputHandler.read_args(configFile, verbose)

    pr = None
    if params['doRunTime']:
        pr = cProfile.Profile()
        pr.enable()

    ## Create an AI object
    AIs = {AIname:create_AI (AIname, AIinfo, doPlots=params['doPlots'], plotPath=params['plotPath'])
           for AIname, AIinfo in params['AIinfo'].items()}
    
    ## Create a disease tree
    aDiseaseTree = create_disease_tree (params['diseaseGroups'],
                                        params['meanServiceTimes'], AIs)
    
    ## Add additional params
    params = inputHandler.add_params (AIs, params, aDiseaseTree)

    ## Check AI performance
    oneSim = simulator.simulator ()
    oneSim.set_params (params)
    oneSim.track_log = False
    hier_classes_dict = {}
    for ii, vendor_name in enumerate(hier.vendor_hierarchy):
    # For hierarchical queuing, disease number starts at 3 (most time-sensitive class).
        hier_classes_dict.update({vendor_name : {'groupName': hier.AI_group_hierarchy[ii], 'disease_num':ii+3}})
    oneSim.simulate_queue (AIs, aDiseaseTree, hier_classes_dict)
    print_sim_performance (oneSim, AIs, params)
    params['n_patients_per_class'] = {qtype:{aclass:get_n_interrupting_patients (oneSim, qtype) if aclass=='interrupting' else \
                                                    get_n_positive_patients  (oneSim, qtype) if aclass=='positive' else \
                                                    get_n_negative_patients  (oneSim, qtype)
                                             for aclass in ['interrupting', 'positive', 'negative']}
                                      for qtype in params['qtypes'][1:]}
    
    ## If do-plots, generate 1 trial to plot case diagram and histograms
    #if params['doPlots']:
    #    # Plot case diagram
    #    for qtype in params['qtypes']:
    #        outFile = plotPath + 'patient_timings_' + qtype + '.pdf'
    #        plotter.plot_timing (outFile, oneSim.all_records, params['startTime'], n=200, qtype=qtype)

    ## Run trials
    t0 = time.time()
    trialGen = trialGenerator.trialGenerator ()
    trialGen.set_params (params, AIs)
    #allTrials, maxSim = trialGen.simulate_trials (anAI)
    trialGen.simulate_trials (AIs, aDiseaseTree)
    params['runTimeMin'] = (time.time() - t0)/60 # minutes
    print ('{0} trials took {1:.2f} minutes'.format (params['nTrials'], params['runTimeMin']))

    # Plot waiting time histograms (will come back to plot)
    # if params['doPlots']:
    #     print("plot_paths")
    #     # Plot n patients histograms
    #     # 1. using all patients from all trials
    #     ext = 'allstats_default_test.pdf'
    #     plotter.plot_n_patient_distributions (params['plotPath'], ext, trialGen.n_patients_system_df,
    #                                           trialGen.n_patients_system_stats, params, oneTrial=False,
    #                                           include_theory=True)
    #     # 2. using all patients from one trial (show mean of means from all trials)
    #     ext = 'trialstats_default_test.pdf'
    #     plotter.plot_n_patient_distributions (params['plotPath'], ext, trialGen.n_patients_system_df,
    #                                           trialGen.n_patients_system_stats_from_trials,
    #                                           params, oneTrial=True, include_theory=True)
        
        # Plot waiting time histograms
        # # 1. using all patients from one trial (show mean of means from all trials)
        # ext = 'trialstats_default_sorted_minDrTime_em0.005_p0.30_r0.80_2.pdf'
        # df = trialGen.waiting_times_df[trialGen.waiting_times_df.trial_id=='trial_000']
        # df = df.drop(axis=1, columns=['trial_id', 'patient_id'])
        # plotter.plot_waiting_time_distributions (params['plotPath'], ext, df, trialGen.waiting_times_stats_from_trials,
        #                                          params, doDiff=False)
        # plotter.plot_waiting_time_distributions (params['plotPath'], ext, df, trialGen.waiting_times_stats_from_trials,
        #                                          params, doDiff=True)
        # # 2. using all patients from all trials
        # ext = 'allstats_default_sorted_minDrTime_em0.005_p0.30_r0.80_2.pdf'
        # df = trialGen.waiting_times_df.drop(axis=1, columns=['trial_id', 'patient_id'])
        # plotter.plot_waiting_time_distributions (params['plotPath'], ext, df, trialGen.waiting_times_stats, params, doDiff=False)
        # plotter.plot_waiting_time_distributions (params['plotPath'], ext, df, trialGen.waiting_times_stats, params, doDiff=True)
        
    ## Gather data for dict
    data = {'params':params,
            'lpatients':trialGen.n_patients_system_df,
            'wpatients':trialGen.waiting_times_df,
            'lstats':trialGen.n_patients_system_stats,
            'wtstats':trialGen.waiting_times_stats}
    with open (params['statsFile'], 'wb') as f:
        pickle.dump (data, f)
    f.close()

    for AIname, AIinfo in params['AIinfo'].items():
        trialGen.waiting_times_df.to_csv(f'{AIinfo['FPFThresh']}_{AIinfo['TPFThresh']}.csv')
    
    if params['doRunTime']:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
        ps.print_stats()

        with open (params['runtimeFile'], 'w+') as f:
            f.write (s.getvalue())
        f.close()