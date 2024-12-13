
##
## By Elim Thompson (11/27/2020)
##
## This script encapsulates a server (radiologist) class. Each radiologist
## has a name and the current patient (an instance that contains all the
## information about the patient case in this radiologist's hand). The
## "current patient" instance will be updated as the radiologist gets
## interrupted or reads a new case. 
###########################################################################

################################
## Import packages
################################ 
import numpy, pandas
from copy import deepcopy

################################
## Define reader class
################################ 
minutes_to_microsec = lambda minute: round (minute * 60 * 10**3 * 10**3)

class radiologist (object):
    
    def __init__ (self, name, serviceTimes):
        
        ''' A radiologist is initialized with a name and the average reading
            (service) times for diseased, non-diseased and interrupting subgroups.
            
            inputs
            ------
            name (str): a unique name for this radiologist
            serviceTimes (dict): {subgroup: average reading time} where the
                                  subgroup keys are diseased, non-diseased,
                                  and interrupting. The values (mean reading
                                  times) are in minutes.
        '''
        
        self._radiologist_name = name
        self._serviceTimes = serviceTimes
        
        ## Keep a record of the patient that is currently read by this radiologist 
        self._current_patient = None
        
    @property
    def _get_serviceRate (self): return self._serviceTimes
    
    @property
    def radiologist_name (self): return self._radiologist_name

    @property
    def current_patient (self): return self._current_patient

    def is_busy (self, thisTime):
        
        ''' Is this radiologist busy at the input time? Yes if the current
            patient has a closing time after the input time.
            
            input
            -----
            thisTime (pandas.datetime): the time to check if this rad is busy
            
            output
            ------
            is_busy (bool): True if this radiologist is busy at input time
                            False if this radiologist is in idle at that time
        '''
        
        # If no patient, s/he is not busy
        if self._current_patient is None: return False 
        
        # If the current patient's closing time is after the input time,
        # s/he is busy. Otherwise, s/he is in idle.
        return self._current_patient.latest_close_time > thisTime
    
    def determine_service_duration (self, disease_name=None,
                                    group_name=None, is_interrupting=False):
        
        ''' Randomly generate a service reading time for a patient with input
            interrupting status and disease (truth) status.
            
            inputs
            ------
            disease_name (str): Name of disease (or "non-diseased")
            group_name (str): Name of the AI group
            is_interrupting (bool): True if interrupting status is interrupting
            
            output
            ------
            reading time (float): random reading time in minutes from an
                                  exponential distribution
        '''
        
        # If interrupting patients, it has its own rate
        if is_interrupting:
            return numpy.random.exponential (self._serviceTimes['interrupting'])
        # For non-interruping, it is based on ground truth condition
        meanTime = self._serviceTimes[group_name][disease_name]
        return numpy.random.exponential (meanTime)
        
    def read_new_patient (self, apatient, hi_priority=None, given_open_time=None):
        
        ''' Called when reading a brand new patient or when resuming a previously
            interrupted patient. This function updates the patient information,
            including this radiologist name as well as the open and close times
            by this radiologist. The updated patient instance is kept as in this
            radiologist instance for later time comparison. 
            
            input
            -----
            apatient (patient): a patient instance to be read by this radiologist
        '''
        
        # Determine case open time of the patient. 
        #  * If radiologist doesn't have any current patient, openTime is the new
        #    patient's trigger time.
        #  * If the new patient's trigger time is after the current patient's close
        #    time, it is also the new patient's trigger time.
        #  * If the new patient's trigger time is before the current patient's close
        #    time, this new patient has to wait until the current case is closed.

        # Note: High priority patient is already set as the current patient if this
        # function is called within read_newest_urgent_patient. Not so otherwise.

        # Previous code before hierarchical queuing was implemented is commented
        # out right below ----
        # openTime = apatient.trigger_time if self.current_patient is None else \
        #            apatient.trigger_time if self.current_patient.latest_close_time <= apatient.trigger_time else \
        #            self.current_patient.latest_close_time

        ##############################################################################
        # The following code was updated as part of hierarchical queuing. 

        # current_patient is hi priority and was never read before. current_patient is apatient as well.
        if hi_priority and (len(apatient.open_times) == 0): 
            openTime = apatient.trigger_time

        # current patient is hi priority, but was also partially read before, and is now reopened. current_patient is apatient as well.
        elif hi_priority and (len(apatient.open_times) > 0) and given_open_time: 
            openTime = given_open_time

        # system is empty, apatient is not hi priority. No current patient. apatient is the only patient to be read.
        elif (self.current_patient is None) and (len(apatient.open_times) == 0): 
            openTime = apatient.trigger_time

        # apatient is not hi priority and was never read before. current_patient and apatient are different patients.
        # current_patient was completed before apatient arrived. 
        elif (self.current_patient.latest_close_time <= apatient.trigger_time) and (len(apatient.open_times) == 0): 
            openTime = apatient.trigger_time

        # apatient is not hi priority. current_patient and apatient are different patients.
        # apatient is opened when current patient is complete since apatient is not higher priority than current_patient.
        else: 
            openTime = self.current_patient.latest_close_time

        # Update the timing information of this new patient
        #  1. the open time
        apatient.open_times.append (openTime)
        #  2. the close time
        readTimeDuration = minutes_to_microsec (apatient.service_duration)
        apatient.close_times.append (openTime + pandas.offsets.Micro (readTimeDuration))
        #  3. this radiologist's name
        apatient.add_doctor (self._radiologist_name)
        
        # Update current patient to the new patient. 
        # This is redundant for hi_priority patients (because stop_reading sets the hi_priority
        # patient to current_patient), but not redundant in other cases.
        self._current_patient = apatient

    def stop_reading (self, hi_priority_trigger_time, hi_priority_patient):

        ''' Called when the reading of current patient is interrupted (by another
            patient of higher priority).
            
            input 
            -----
            stop_reading_time (pandas.datetime): time when the current patient is
                                                 interrupted, likely the arrival
                                                 time of a higher-priority patient
                                                 
            output
            ------
            apatient (patient): the current (lower-priority) patient with updated
                                remaining service duration and close time
        '''
        # Flag to check if the interrupted patient was at least partially read before being interrupted.
        # Update to False if found otherwise.
        atleast_partially_read = True 
        # Make a copy of the current patient to avoid changing data
        apatient = deepcopy (self._current_patient) # interrupted patient
        
        # 1. If the interrupting patient is seen for the first time, use their trigger time as the stop reading time. 
        # 2. Else, use the start reading time of the interrupted patient, i.e., the interrupted patient is not seen at all, but replaced by the interrupting patient.
        # The second occurs in hierarchical queuing when multiple interruptions can occur.
        # This may lead to a choice between reading a low-priority apatient or resuming a previously interrupted, but now hi-priority, hi_priority_patient.
        stop_reading_time = hi_priority_trigger_time if len(hi_priority_patient._open_times) == 0 else apatient.latest_open_time

        # For preemptive resume, we need to check the remaining service time
        # i.e. service duration of this patient - (diff between open times of
        #      next and this patient)
        new_service_duration = apatient.service_duration - \
                               (stop_reading_time - apatient.latest_open_time).total_seconds()/60
        # For debugging purposes, need to take a look if the new duraction is negative
        if new_service_duration < 0:
            print ('New service time for Case {0} is negative?'.format (apatient.caseID))
        
        if not stop_reading_time == apatient.latest_open_time: # if open and close times are not the same.
            # Update the timing information of the lower-priority patient
            #  1. remaining service time 
            apatient.service_duration = new_service_duration
            #  2. case close time
            apatient.add_close_time (stop_reading_time)

        else:
            atleast_partially_read = False # if open and close times are the same. Corresponds to 2. in stop_reading_time.
        
        # Re-set this radiologist current patient to the interrupted patient
        # self._current_patient = None 
        # Commented out so that during interruption, current_patient is not None / system is not empty. 
        # If uncommented, this will lead to a bug and negative wait-times for some patients in hierarchical queues.

        self._current_patient = hi_priority_patient # updated RD
        
        # Return the lower-priority patient instance and reading info about this patient.
        return apatient, stop_reading_time, atleast_partially_read, (stop_reading_time - apatient.latest_open_time).total_seconds()/60