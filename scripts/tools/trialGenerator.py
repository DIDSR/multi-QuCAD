
##
## By Elim Thompson (12/16/2020)
##
## This script defines a class that generate trials. This class passes
## user input parameters and calls the simulator class to simulate
## reading flow. This script is used when you want the results from
## one single computer. If you have a cluster, you can run individual
## trials in parallel without using this script.
##
## 05/21/2024
## ----------
## * Add in
########################################################################

################################
## Import packages
################################ 
import pandas, numpy, logging
from . import simulator

################################
## Define constants
################################
## Names of queue types for with and without CADt scenarios. Due to
## historical reasons, fifo means without-CADt scenario. In the past,
## without-CADt scenario did not include interrupting subgroup. But now,
## with this highest-priority group, without-CADt scenario is a
## 2-priority-class system and is no longer fifo. However, this
## software still calls this without-CADt scenario "fifo". 
queuetypes = ['fifo', 'priority', 'hierarchical'] 
## Names of priority classes in with CADt scenario. Positive
## and negative patients will be lumped into one priority
## class in the without-CADt scenario.
priorityClasses = {'interrupting':1, 'positive':2, 'negative':99}

## Simulation default parameters
##  1. number of days during simulation. Default: 1 month
timeWindowDays = 30
##  2. number of radiologists. Default: 1. Same number for both
##     with and without CADt scenario. For now, average reading
##     time per patient subgroup is the same for all radiologists
##     if multiple radiologists are simulated. 
nRadiologists = 1
##  2. number of patients to be padded before and after the
##     simulation periods. It is found to have no impacts
##     on the state probability distribution.
nPatientsPads = [100, 100]
##  3. simulation start timestamp. Actual time doesn't
##     really matter, but it is needed for the first
##     patient's arrival.
startTime = pandas.to_datetime ('2020-01-01 00:00')
##  4. number of trials to be simulated. Default: 100 trials
nTrials = 100

################################
## Define class
################################ 
class trialGenerator (object):
    
    def __init__ (self):
    
        ''' A trialGenerator doesn't need any parameters to be initialized.
            To update the parameters f a trialGenerator instance, use
            set_params().
        '''
    
        ## Parameters related to the queues themselves. These parameters
        ## will not be updated when a user changes input parameters.
        self._classes = priorityClasses
        self._qtypes = queuetypes 
    
        ## Parameters related to debugging
        self._logger = logging.getLogger ('trialGenerator.trialGenerator')

        self._sim_performance = {}

        ## Parameters related to simulation results. These parameters will
        ## not be updated when a user changes input parameters.
        #   1. results from individual simulations 
        self._waiting_times_df = None
        self._n_patients_queue_df = None
        self._n_patients_system_df = None
        #   2. statistics from all simulations
        self._waiting_times_stats = None
        self._n_patients_queue_stats = None
        self._n_patients_system_stats = None
        self._waiting_times_stats_from_trials = None
        self._n_patients_queue_stats_from_trials = None
        self._n_patients_system_stats_from_trials = None

        ## Parameters that will be updated when a user changes input params.
        self._nTrials = nTrials                    # number of trials to be run
        self._startTime = startTime                # simulation start timestamp
        self._nRadiologists = nRadiologists        # number of radiologists 
        self._timeWindowDays = timeWindowDays      # simulation duration in days
        self._nPatientsPadStart = nPatientsPads[0] # number of patients to be padded before counting results 
        self._nPatientsPadEnd = nPatientsPads[1]   # number of patients to be padded after counting results
        self._hierDict = None                      # disease ranking in hierarchy queue
        self._isPreemptive = True                  # priority queue type: is it preemptive? Default True        

        self._fractionED = None                    # fraction of emergent patient in all patients
        self._arrivalRate = None                   # overall patient arrival rate regardless of subgroups 
        self._serviceTimes = None                  # mean reading time by interrupting, diseased, and non-diseased
        self._diseaseGroups = None                 # disease prevalences within each group

        self._probs_ppv_npv = None                 # For printing sim performance

    ## +----------------------------------------
    ## | Class properties
    ## +----------------------------------------
    @property
    def qtypes (self): return self._qtypes
    @property
    def classes (self): return self._classes
    @property
    def n_patients_queue_df (self): return self._n_patients_queue_df
    @property
    def n_patients_system_df (self): return self._n_patients_system_df
    @property
    def waiting_times_df (self): return self._waiting_times_df
    @property
    def n_patients_queue_stats (self): return self._n_patients_queue_stats
    @property
    def n_patients_system_stats (self): return self._n_patients_system_stats
    @property
    def waiting_times_stats (self): return self._waiting_times_stats
    @property
    def n_patients_queue_stats_from_trials (self): return self._n_patients_queue_stats_from_trials
    @property
    def n_patients_system_stats_from_trials (self): return self._n_patients_system_stats_from_trials
    @property
    def waiting_times_stats_from_trials (self): return self._waiting_times_stats_from_trials
    @property
    def sim_performance(self): return self._sim_performance    
    @property
    def nTrials (self): return self._nTrials
    @nTrials.setter
    def nTrials (self, nTrials):
        if not isinstance (nTrials, int):
            raise IOError ('Input nTrial must be an integer.')
        self._nTrials = nTrials
    @property
    def startTime (self): return self._startTime
    @startTime.setter
    def startTime (self, startTime):
        if isinstance (startTime, str):
            try:
                startTime = pandas.to_datetime (startTime)
            except:
                raise IOError ('Input startTime string is invalid.')
        if not isinstance (startTime, pandas._libs.tslibs.timestamps.Timestamp):
            raise IOError ('Input startTime must be a pandas Timestamp object.')
        self._startTime = startTime
    @property
    def endTime (self):
        if self._timeWindowDays >= 1:
            return self._startTime + pandas.offsets.Day (self._timeWindowDays)
        return self._startTime + pandas.offsets.Minute (int (self._timeWindowDays*24*60))
    @property
    def timeWindowDays (self): return self._timeWindowDays
    @timeWindowDays.setter
    def timeWindowDays (self, timeWindowDays):
        self._timeWindowDays = timeWindowDays
    @property
    def nPatientPadsStart (self): return self._nPatientPadsStart
    @nPatientPadsStart.setter
    def nPatientPadsStart (self, nPatientPadsStart):
        if not isinstance (nPatientPadsStart, int):
            raise IOError ('Input nPatientPadsStart must be an integer.')         
        self._nPatientPadsStart = nPatientPadsStart
    @property
    def nPatientPadsEnd (self): return self._nPatientPadsEnd
    @nPatientPadsEnd.setter
    def nPatientPadsEnd (self, nPatientPadsEnd):
        if not isinstance (nPatientPadsEnd, int):
            raise IOError ('Input nPatientPadsEnd must be an integer.')            
        self._nPatientPadsEnd = nPatientPadsEnd         
    @property
    def diseaseGroups (self): return self._diseaseGroups
    @diseaseGroups.setter
    def diseaseGroups (self, diseaseGroups):
        self._diseaseGroups = diseaseGroups
    @property
    def fractionED (self): return self._fractionED
    @fractionED.setter
    def fractionED (self, fractionED):
        if not isinstance (fractionED, float):
            raise IOError ('Input fractionED must be a float.')
        if not (fractionED >= 0.0 and fractionED <= 1.0):
            raise IOError ('Input fractionED must be between 0 and 1.')
        self._fractionED = fractionED 
    @property
    def arrivalRate (self): return self._arrivalRate
    @arrivalRate.setter
    def arrivalRate (self, arrivalRate):
        if not isinstance (arrivalRate, float):
            raise IOError ('Input arrivalRate must be a float.')
        self._arrivalRate = arrivalRate
    @property
    def serviceTimes (self): return self._serviceTimes
    @serviceTimes.setter
    def serviceTimes (self, serviceTimes):
        # 1. It must be a dictionary
        if not isinstance (serviceTimes, dict):
            raise IOError ('Input serviceTimes must be a dictionary.')
        self._serviceTimes = serviceTimes
    @property
    def hierDict (self): return self._hierDict
    @hierDict.setter
    def hierDict (self, hierDict):
        if not isinstance (hierDict, dict):
            raise IOError ('Input hierDict must be a dictionary.')        
        self._hierDict = hierDict
    @property
    def nRadiologists (self): return self._nRadiologists
    @nRadiologists.setter
    def nRadiologists (self, nRadiologists):
        if not isinstance (nRadiologists, int):
            raise IOError ('Input nRadiologists must be an integer.')        
        self._nRadiologists = nRadiologists
    @property
    def isPreemptive (self): return self._isPreemptive
    @isPreemptive.setter
    def isPreemptive (self, isPreemptive):
        if not isinstance (isPreemptive, bool):
            raise IOError ('Input isPreemptive must be a boolean.')        
        self._isPreemptive = isPreemptive      

    ## +---------------------------------------------
    ## | Private functions to generate trials
    ## +---------------------------------------------
    def _reset_sim (self, sim):

        ''' Reset simulator instance and erase data from previous
            simulation trial. This is used in simulate_trials,
            which should be called before set_params().
            
            input
            -----
            sim (simulator): simulator instance to generate 1 trial
                             with data from the previous trial
         
            output
            ------
            sim (simulator): simulator instance to generate 1 trial
                             with no data (i.e. ready to run simulation)
        '''
        
        ## Reset simulator by emptying data holders 
        sim.reset()
        
        ## Set parameters
        sim.startTime = self._startTime
        sim.fractionED = self._fractionED
        sim.arrivalRate = self._arrivalRate
        sim.serviceTimes = self._serviceTimes
        sim.nRadiologists = self._nRadiologists
        sim.diseaseGroups = self._diseaseGroups
        sim.timeWindowDays = self._timeWindowDays
        sim.nPatientPadsEnd = self._nPatientPadsEnd
        sim.nPatientPadsStart = self._nPatientPadsStart
        sim.hierDict = self._hierDict
        sim.isPreemptive = self._isPreemptive
        
        return sim
        
    def _get_stats (self, values, weights=None):
    
        ''' Obtain statistics from a distribution of input values.
            Statistics include 95% C.I. and 1 sigma ranges, as
            well as median and mean.
            
            inputs
            ------
            values (array): values from which a distribution is built
            weights (array): weights of elements in `values`. Must
                             have same length as `values`.
                             
            output
            ------
            stats (dict): statistics of the distribution from values.
                          Keys: 'lower_95cl', 'lower_1sigma', 'median',
                                'mean', 'upper_1sigma', 'upper_95cl'
        '''
    
        ## If no valid values, return 0's
        sample = numpy.array(values)[numpy.isfinite(values)]
        if len (sample) == 0:
            return {'lower_95cl':0, 'lower_1sigma':0, 'median':0,
                    'upper_1sigma':0, 'upper_95cl':0, 'mean':0}
        
        ## Massage input values and apply weights 
        indices = numpy.argsort (sample)
        sample = numpy.array ([sample[i] for i in indices])
        if weights is None: weights = numpy.ones (len (sample))
        w = numpy.array ([weights[i] for i in indices])
            
        ## Extract statistics from cumulative PDF
        stat = {}
        cumulative = numpy.cumsum(w)/numpy.sum(w)
        stat['lower_95cl']   = sample[cumulative>0.025][0] # lower part is at 50%-47.5% = 2.5%        
        stat['lower_1sigma'] = sample[cumulative>0.16][0]  # lower part is at 50%-34% = 16%
        stat['median']       = sample[cumulative>0.50][0]  # median
        stat['mean']         = numpy.average (sample, weights=w)
        stat['upper_1sigma'] = sample[cumulative>0.84][0]  # upper is 50%+34% = 84%
        stat['upper_95cl']   = sample[cumulative>0.975][0] # upper is 50%+47.5% = 97.5%
        stat['sample_size']   = len (sample) # added RD
        stat['sum']   = numpy.sum(sample) # added RD
        
        return stat

    def _get_waiting_time_stats (self, df):

        ''' Obtain statistics from waiting time simulated data. Each patient
            is treated independently. In each of the two scenarios (with and
            without CADt), for each subgroups (all, interrupting, non-interrupting,
            diseased, non-diseased, positive, negative, TP, TN, FP, FN),
            the statistics from both the absolute waiting times and the
            difference in waiting time between the two scenarios are obtained. 
            
            input
            -----
            df (pandas.DataFrame): waiting time per patient in each subgroup
            
            output
            ------
            stats (dict): statistics of waiting time simulated data for
                          each subgroup in each of the two scenarios.
                          stats[qtype][either 'waitTime' or 'diff'][subgroup]
        '''

        ## Holder for statistics
        stats = {}
    
        ## Get statistics for each scenario
        for qtype in self._qtypes:
            # Create a holder for this scenario
            stats[qtype] = {}
            # Extract the waiting time from non-interrupting patients because
            # 
            noninterruptingDF = df[~df.is_interrupting]
            
            ## Collect absolute waiting time
            stats[qtype]['waitTime'] = {'all'          : self._get_stats(df[qtype]),
                                        'non-interrupting': self._get_stats(noninterruptingDF[qtype]),
                                        'interrupting'    : self._get_stats(df[df.is_interrupting][qtype]),
                                        'diseased'     : self._get_stats(noninterruptingDF[noninterruptingDF.is_diseased][qtype]),
                                        'non-diseased' : self._get_stats(noninterruptingDF[noninterruptingDF.is_diseased==False][qtype]),
                                        'positive'     : self._get_stats(noninterruptingDF[noninterruptingDF.is_positive][qtype]),
                                        'negative'     : self._get_stats(noninterruptingDF[~noninterruptingDF.is_positive][qtype]),                
                                        'TP'           : self._get_stats(noninterruptingDF[numpy.logical_and (noninterruptingDF.is_diseased, noninterruptingDF.is_positive)][qtype]),
                                        'TN'           : self._get_stats(noninterruptingDF[numpy.logical_and (noninterruptingDF.is_diseased==False, ~noninterruptingDF.is_positive)][qtype]),
                                        'FP'           : self._get_stats(noninterruptingDF[numpy.logical_and (noninterruptingDF.is_diseased==False, noninterruptingDF.is_positive)][qtype]),
                                        'FN'           : self._get_stats(noninterruptingDF[numpy.logical_and (noninterruptingDF.is_diseased, ~noninterruptingDF.is_positive)][qtype])}
        
            # Handle differences
            if qtype == 'fifo': continue
            all_values  = df[qtype] - df.fifo
            none_values = all_values[~df.is_interrupting]
            e_values    = all_values[df.is_interrupting]
            d_values    = none_values[noninterruptingDF.is_diseased]
            n_values    = none_values[noninterruptingDF.is_diseased==False]
            pv_values   = none_values[noninterruptingDF.is_positive]
            nv_values   = none_values[~noninterruptingDF.is_positive]
            tp_values   = none_values[numpy.logical_and (noninterruptingDF.is_diseased, noninterruptingDF.is_positive)]
            fn_values   = none_values[numpy.logical_and (noninterruptingDF.is_diseased, ~noninterruptingDF.is_positive)]
            tn_values   = none_values[numpy.logical_and (noninterruptingDF.is_diseased==False, ~noninterruptingDF.is_positive)]
            fp_values   = none_values[numpy.logical_and (noninterruptingDF.is_diseased==False, noninterruptingDF.is_positive)]            
            stats[qtype]['diff'] = {'all': self._get_stats(all_values),
                                    'non-interrupting': self._get_stats(none_values),
                                    'interrupting': self._get_stats(e_values),
                                    'diseased': self._get_stats(d_values),
                                    'non-diseased': self._get_stats(n_values),
                                    'positive': self._get_stats(pv_values),
                                    'negative': self._get_stats(nv_values),                
                                    'TP': self._get_stats(tp_values),
                                    'TN': self._get_stats(tn_values),
                                    'FP': self._get_stats(fp_values),
                                    'FN': self._get_stats(fn_values)}      
    
        return stats       
    
    def _get_n_patients_stats (self, df):

        ''' Obtain statistics from number of patients as observed by each
            in-coming patient image in simulated data. 

            input
            -----
            df (dict): number of patients observed by each in-coming
                       patient images for both with and without CADt
                       scenarios.
            
            output
            ------
            stats (dict): statistics of waiting time simulated data for
                          each subgroup in each of the two scenarios.
                          stats[qtype][subgroup]
        '''

        stats = {}
        for qtype in self._qtypes:
            if qtype == 'hierarchical': continue
            stats[qtype] = {}
            for gtype in df.keys():
                if qtype == 'fifo' and gtype in ['positive', 'negative']: continue
                stats[qtype][gtype] = self._get_stats(df[gtype][qtype])
        return stats
        
    def _set_stats (self, nPatientsQueuedfs, nPatientsSystemdfs, waitTimedfs):
        
        ''' Obtain statistics for 
                1. number of patients in system observed by each in-coming patient
                2. number of patients in queue observed by each in-coming patient
                   (this is not used because number of patients in system is
                    the state probability, instead of the number in queue)
                3. waiting time of each patient

            input
            -----
            nPatientsQueuedfs (pandas DataFrame): number of patients in queue
                                                  observed by each in-coming
                                                  patient images for both with and
                                                  without CADt scenarios.
            nPatientsSystemdfs (pandas DataFrame): number of patients in system
                                                   observed by each in-coming
                                                   patient images for both with and
                                                   without CADt scenarios.                                                  
            waitTimedfs (pandas DataFrame): waiting time of each patients in both
                                            with and without CADt scenarios.
        '''

        self._waiting_times_df = pandas.concat (waitTimedfs, ignore_index=True)
        self._n_patients_queue_df = {gtype: pandas.concat (df, ignore_index=True)
                                     for gtype, df in nPatientsQueuedfs.items()}
        self._n_patients_system_df = {gtype: pandas.concat (df, ignore_index=True)
                                      for gtype, df in nPatientsSystemdfs.items()}
        
        # Gather stats from all patients in all trials
        self._waiting_times_stats = self._get_waiting_time_stats (self._waiting_times_df)
        self._n_patients_queue_stats = self._get_n_patients_stats (self._n_patients_queue_df)
        self._n_patients_system_stats = self._get_n_patients_stats (self._n_patients_system_df)

    def _print_timeStats (self, oneSim, trialId):
        
        ''' A function for debugging. 
        '''

        print ('================================ {0} ================================'.format (trialId))
        
        for qtype in ['fifo', 'priority']:
            TP = numpy.array ([[a.caseID, a.total_service_duration, a.wait_time_duration]
                               for a in oneSim.get_TP_records (qtype)])
            FN = numpy.array ([[a.caseID, a.total_service_duration, a.wait_time_duration]
                               for a in oneSim.get_FN_records (qtype)])
            FP = numpy.array ([[a.caseID, a.total_service_duration, a.wait_time_duration]
                               for a in oneSim.get_FP_records (qtype)])
            TN = numpy.array ([[a.caseID, a.total_service_duration, a.wait_time_duration]
                               for a in oneSim.get_TN_records (qtype)]) 
            
            for timeType in ['service', 'wait']:
                
                tindex = 1 if timeType == 'service' else 2 
                
                TP_times = [numpy.min  (TP.T[tindex].astype (float)),
                            numpy.mean (TP.T[tindex].astype (float)),
                            numpy.max  (TP.T[tindex].astype (float))]
                FN_times = [numpy.min  (FN.T[tindex].astype (float)),
                            numpy.mean (FN.T[tindex].astype (float)),
                            numpy.max  (FN.T[tindex].astype (float))]
                FP_times = [numpy.min  (FP.T[tindex].astype (float)),
                            numpy.mean (FP.T[tindex].astype (float)),
                            numpy.max  (FP.T[tindex].astype (float))]
                TN_times = [numpy.min  (TN.T[tindex].astype (float)),
                            numpy.mean (TN.T[tindex].astype (float)),
                            numpy.max  (TN.T[tindex].astype (float))]
                
                print ('+-{0:9}-+-{0:9}-+-{0:9}-+-{0:9}-+-{0:9}-+'.format ('-'*9))
                print ('| {0:9} | {1:21} | {2:21} |'.format (qtype, 'Diseased', 'Non-diseased'))
                print ('+-{0:9}-+-{0:9}-+-{0:9}-+-{0:9}-+-{0:9}-+'.format ('-'*9))
                print ('| {0:9} | {1:9} | {2:9} | {3:9} | {4:9} |'.format (timeType, 'TP', 'FN', 'FP', 'TN'))
                print ('+-{0:9}-+-{0:9}-+-{0:9}-+-{0:9}-+-{0:9}-+'.format ('-'*9))
                print ('| {0:9} | {1:9.4f} | {2:9.4f} | {3:9.4f} | {4:9.4f} |'.format ('min', TP_times[0],
                                                                                       FN_times[0], FP_times[0],
                                                                                       TN_times[0]))
                print ('+-{0:9}-+-{0:9}-+-{0:9}-+-{0:9}-+-{0:9}-+'.format ('-'*9))
                print ('| {0:9} | {1:9.4f} | {2:9.4f} | {3:9.4f} | {4:9.4f} |'.format ('mean', TP_times[1],
                                                                                       FN_times[1], FP_times[1],
                                                                                       TN_times[1]))                    
                print ('+-{0:9}-+-{0:9}-+-{0:9}-+-{0:9}-+-{0:9}-+'.format ('-'*9))
                print ('| {0:9} | {1:9.4f} | {2:9.4f} | {3:9.4f} | {4:9.4f} |'.format ('max', TP_times[2],
                                                                                       FN_times[2], FP_times[2],
                                                                                       TN_times[2]))                    
                print ('+-{0:9}-+-{0:9}-+-{0:9}-+-{0:9}-+-{0:9}-+'.format ('-'*9))
                print ('')
      
    ## +--------------------------------------------------------------
    ## | Public functions to set parameters and start trials
    ## +--------------------------------------------------------------
    def _get_sim_performance (self, oneSim, AIs):
    
        ## Check with group prob and disease prevalence
        patient_groups = numpy.array ([p.group_name for p in oneSim.get_noninterrupting_records('fifo')])
        for groupName, groupInfo in self.diseaseGroups.items():
            
            # 1. check group prob
            patient_groupnames_in_this_group = patient_groups[patient_groups==groupName]
            g_sim = len (patient_groupnames_in_this_group) / len (patient_groups)
            self._sim_performance['group'][groupName]['groupFraction'].append (g_sim)
            print ('+-----------------------------------------------')
            print ('| Group {0} ({1}):'.format (groupName, len (patient_groupnames_in_this_group)))
            print ('|   * Group prob (theory, sim): {0:.3f}, {1:.3f}'.format (groupInfo['groupProb'], g_sim))            
            
            # 2. check disease prob within this group
            patient_diseases = numpy.array ([p.disease_name for p in oneSim.get_noninterrupting_records('fifo')
                                             if p.group_name == groupName])
            disease_names = numpy.array (groupInfo['diseaseNames'] + ['non-diseased'])
            unexpected_patient_diseases = patient_diseases[~numpy.in1d (patient_diseases, disease_names)]
            if len (unexpected_patient_diseases) > 0: 
                print ('|   * Unexpected disease in this group: {0}'.format (unexpected_patient_diseases))            

            disease_probs = numpy.array (groupInfo['diseaseProbs'] + [1-sum(groupInfo['diseaseProbs'])])
            for diseaseName, diseaseProb in zip (disease_names,  disease_probs):
                patient_diseasenames_in_this_group = patient_diseases[patient_diseases==diseaseName]
                d_sim = len (patient_diseasenames_in_this_group) / len (patient_diseases)
                self._sim_performance['group'][groupName]['diseases'][diseaseName].append (d_sim)
                print ('|   * Disease {0} ({1}):'.format (diseaseName, len (patient_diseasenames_in_this_group)))
                print ('|       -- Disease prob (theory, sim): {0:.3f}, {1:.3f}'.format (diseaseProb, d_sim))

        print ('+-----------------------------------------------\n')        

        ## Check with AI performance
        for AIname, anAI in AIs.items():
            
            groupName = anAI.groupName
            targetDisease = anAI.targetDisease
            Se, Sp = anAI.SeThresh, anAI.SpThresh
        
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
            
            self._sim_performance['AI'][AIname]['ppv'].append (ppv_sim)
            self._sim_performance['AI'][AIname]['npv'].append (npv_sim)

            print ('+-----------------------------------------------')
            print ('| AI {0} ({1}) in group {2}:'.format (AIname, targetDisease, groupName))
            print ('|    * PPV (theory, sim): {0:.4f}, {1:.4f}'.format (self._probs_ppv_npv['ppv'][groupName][AIname], ppv_sim))
            print ('|    * NPV (theory, sim): {0:.4f}, {1:.4f}'.format (self._probs_ppv_npv['npv'][groupName][AIname], npv_sim))
            
        print ('+-----------------------------------------------\n')

    def set_params (self, params, AIs):
        
        ''' Function to set parameters for all trials.

            inputs
            ------
            params (dict): dictionary capsulating all simulation parameters
            AIs (dict): dictionary of all AIs involved in the queue            
        '''

        self.nTrials = params['nTrials']
        self.startTime = params['startTime']
        self.timeWindowDays = params['timeWindowDays']
        self.doPlots = params['doPlots']
        self.diseaseGroups = params['diseaseGroups']
        self.fractionED = params['fractionED']
        self.arrivalRate = 1/params['meanArrivalTime']
        self.serviceTimes = params['meanServiceTimes'] 
        self.nRadiologists = params['nRadiologists']
        self.nPatientPadsStart = params['nPatientsPads'][0]
        self.nPatientPadsEnd = params['nPatientsPads'][1]
        self.hierDict = params['hierDict']
        self.isPreemptive = params['isPreemptive']
        self._probs_ppv_npv = params['probs_ppv_npv']
    
        ## Once we know the disease groups, we can initialize sim_performanace.
        self._sim_performance['group'] = {}
        for groupName, groupInfo in params['diseaseGroups'].items():
            self._sim_performance['group'][groupName] = {'groupFraction': [], 'diseases':{}}
            for diseaseName in numpy.array (groupInfo['diseaseNames'] + ['non-diseased']):
                self._sim_performance['group'][groupName]['diseases'][diseaseName] = []
            
        self._sim_performance['AI'] = {}
        for AIname in AIs.keys():
            self._sim_performance['AI'][AIname] = {'ppv':[], 'npv':[]}

    def simulate_trials (self, AIs, aDiseaseTree):
        
        ''' Function to simulate as many as trials asked.

            inputs
            ------
            AIs (dict): dictionary of all AIs involved in the queue
            aDiseaseTree (diseaseTree): a diseaseTree that have all group/disease
                                        probabilities                     
        '''

        waitTimedfs = []
        subgroups = ['non-interrupting', 'interrupting', 'positive', 'negative']
        ## Add TP, FN, etc
        nPatientsSystemdfs = {group:[] for group in subgroups}
        nPatientsQueuedfs = {group:[] for group in subgroups}
        sim = simulator.simulator()
        ## Generate trials
        for i in range (self._nTrials):
            
            if i%10 == 0 :
                self._logger.debug (' -- {0} / {1} simulations'.format (i, self._nTrials))

            ## Generate a trial
            sim = self._reset_sim (sim)
            sim.track_log = False
            sim.simulate_queue (AIs, aDiseaseTree)
            if i==0: self._get_sim_performance (sim, AIs)

            # Get waiting time data frame (one frame per trial)
            df = sim.waiting_time_dataframe
            df['trial_id'] = 'trial_' + str (i).zfill (3)
            df['patient_id'] = df.index
            df['delta'] = df['priority'] - df['fifo'] # added RD
            waitTimedfs.append (df)
            
            # Handle n customers per class 
            for group in subgroups:
                df = sim.get_n_patients_in_queue(group)
                df['trial_id'] = 'trial_' + str (i).zfill (3)
                df['patient_id'] = df.index
                nPatientsQueuedfs[group].append (df)
                
                df = sim.get_n_patients_in_system(group)
                df['trial_id'] = 'trial_' + str (i).zfill (3)
                df['patient_id'] = df.index
                nPatientsSystemdfs[group].append (df)

        self._set_stats (nPatientsQueuedfs, nPatientsSystemdfs, waitTimedfs)

    