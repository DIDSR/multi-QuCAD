## 
## Elim Thompson (03/10/2023)
##
## This script contains functions that handle input user values, either as
## argument flags or as an input file. Based on that, an output params is
## returned for simulation to run on.
##
## 05/08/2023
## ----------
## * Add in properties for multi-AI scenario
##
## 05/20/2024
## ----------
## * Cleaned up for publishing multi-QuCAD 
################################################################################

################################
## Import packages
################################
import numpy, pandas, os, argparse, sys
from copy import deepcopy

sys.path.insert(0, os.getcwd()+'\\tools')
from . import diseaseTree, AI, hierarchy, calculator

################################
## Define constants
################################
## +------------------------
## | Workflow setting
## +------------------------
qtypes = ['fifo', 'priority', 'hierarchical'] # 'fifo' = without CADt scenario; 'priority' = with CADt scenario
rhoThresh = 0.99               # Maximum allowed hospital busyness
nPatientsPads = [0, 1]         # Chop off the first and last 100 patients
startTime = pandas.to_datetime ('2020-01-01 00:00') # Simulation Start time 

## +------------------------
## | Config file titles
## +------------------------
#configFile = '../scripts/config_resday4.dat'
configTitles = numpy.array (['Clinical setting', 'Group and disease parameters',
                             'CADt AI diagnostic performance', 'Simulation setting'])

################################
## Define lambdas
################################ 
get_ppv = lambda p, Se, Sp: 0 if Se==0 else p*Se / (p*Se + (1-p)*(1-Sp))
get_npv = lambda p, Se, Sp: 0 if Se==1 else 1 - p*(1-Se) / (p*(1-Se) + (1-p)*Sp)

get_n_positive_patients = lambda oneSim, qtype:len (oneSim.get_positive_records(qtype))
get_n_negative_patients = lambda oneSim, qtype:len (oneSim.get_negative_records(qtype))
get_n_interrupting_patients = lambda oneSim, qtype:len (oneSim.get_interrupting_records(qtype))

get_timeWindowDay = lambda arrivalRate, nPatientsTarget: int (numpy.ceil (nPatientsTarget / arrivalRate / (24*60)))

get_service_rate = lambda service_time: numpy.nan_to_num (numpy.inf) if service_time==0 else 1/service_time
get_service_rates = lambda meanServiceTime: get_service_rate (meanServiceTime) if not isinstance (meanServiceTime, dict) else \
                                            {disease: get_service_rate (aTime) for disease, aTime in meanServiceTime.items()}

get_mu_effective = lambda params: 1/((1-params['prob_isInterrupting']) / params['mus']['non-interrupting'] + \
                                     params['prob_isInterrupting'] / params['mus']['interrupting'])
get_lambda_effective = lambda params: params['traffic']*params['nRadiologists']*params['mu_effective']

get_service_rate = lambda service_time: numpy.nan_to_num (numpy.inf) if service_time==0 else 1/service_time

################################
## Define functions
################################ 
def _check_between_0_and_1 (key, value):

    if value > 1:
        print ('ERROR: Input {0} {1:.3f} is too high.'.format (key, value))
        raise IOError ('Please provide a {0} between 0 and 1.'.format (key))
    if value < 0:
        print ('ERROR: Input {0} {1:.3f} is too low.'.format (key, value))
        raise IOError ('Please provide a {0} between 0 and 1.'.format (key))            

def check_user_inputs (params):

    ''' Function to check input traffic, FPFThresh, rocFile, and existence of
        input ROC file (rocFile) and output files / paths (plotPlot, statsFile,
        runtimeFile).

        input
        -----
        params (dict): dictionary capsulating all user inputs
    '''

    ## Checks on traffic
    if params['traffic'] > rhoThresh:
        print ('ERROR: Input traffic {0:.3f} is too high.'.format (params['traffic']))
        raise IOError ('Please limit traffic below {0:.3f}.'.format (rhoThresh))

    ## Checks on values that are between 0 and 1
    for key in ['traffic', 'fractionED']: _check_between_0_and_1 (key, params[key])
    #  AI: TPF and FPF
    for name, info in params['AIinfo'].items():
        # Check TPF
        _check_between_0_and_1 ('AIinfo {0} {1}'.format (name, 'TPFThresh'), info['TPFThresh'])
        # Check FPF
        if info['FPFThresh'] is not None: 
            _check_between_0_and_1 ('AIinfo {0} {1}'.format (name, 'FPFThresh'), info['FPFThresh'])
        # Check FPFThresh and rocFile
        #  1. Both are provided
        if info['FPFThresh'] is not None and info['rocFile'] is not None:
            print ('WARN: {0} has both FPFThresh and rocFile. Taking FPFThresh and ignoring rocFile.'.format (name))
            params['rocFile'] = None
        #  2. Neither are provided
        if info['FPFThresh'] is None and info['rocFile'] is None:
            print ('ERROR: Neither FPFThresh nor rocFile is provided for {0}.'.format (name))
            raise IOError ('Please provide either FPFThresh or rocFile.')
        # Check ROC file
        if info['rocFile'] is not None:
            if not os.path.exists (info['rocFile']):
                print ('ERROR: Input rocFile does not exist.')
                raise IOError ('Please provide a valid rocFile with two columns (first is true-positive fraction, and second is false-positive fraction).')

    #  disease group: groupProb and diseaseProbs
    for groupname, info in params['diseaseGroups'].items():
        _check_between_0_and_1 ('diseaseGroups {0} groupProb'.format (groupname), info['groupProb'])
        for diseasename, value in zip (info['diseaseNames'], info['diseaseProbs']):
            _check_between_0_and_1 ('diseaseGroups {0} disease {1}'.format (groupname, diseasename), value)

    ## Checks on number of radiologists
    if params['nRadiologists'] > 2 and params['fractionED'] > 0.0:
        print ('WARN: There are more than 2 radiologits with presence of interrupting images.')
        print ('WARN: Theoretical values for AI negative and diseased/non-diseased subgroups will not be available.')

    ## Checks if file/folders exists
    for location in ['statsFile', 'runTimeFile', 'plotPath']:
        if params[location] is not None:
            if not os.path.exists (os.path.dirname (params[location])):
                print ('ERROR: Path does not exist: {0}'.format (os.path.dirname (params[location])))
                try:
                    print ('ERROR: Trying to create the folder ...')
                    os.mkdir (os.path.dirname (params[location]))
                except:
                    raise IOError ('Cannot create the folder.\nPlease provide a valid {0} path.'.format (location))

def read_args (configFile): 

    ''' Function to read user inputs. Adding doPlot and doRunTime based on whether
        user provides the paths.

        input
        -----
        configFile (str): path to the config file

        output
        ------
        params (dict): dictionary capsulating all user inputs
    '''

    ## Put everything in a dictionary
    params = {'configFile':configFile, 'qtypes':qtypes,
              'nPatientsPads':nPatientsPads, 'startTime':startTime}
    params.update (read_configFile (configFile))

    if params['verbose']:
        print ('Reading user inputs:')
        print ('+------------------------------------------')
        ## Put user inputs into `params` 
        for key in params.keys():
            if key in ['qtypes', 'nPatientsPads', 'startTime']: continue
            if key in ['meanServiceTimes', 'diseaseGroups', 'AIinfo']:
                for subgroup in params[key].keys():
                    print ('| {0} {1}: {2}'.format (key, subgroup, params[key][subgroup]))
                continue
            print ('| {0}: {1}'.format (key, params[key]))

    ## Add a few flags 
    params['doPlots'] = params['plotPath'] is not None
    if params['verbose']: print ('| doPlots: {0}'.format (params['doPlots']))
    params['doRunTime'] = params['runTimeFile'] is not None
    if params['verbose']: print ('| doRunTime: {0}'.format (params['doRunTime']))

    if params['verbose']:
        print ('+------------------------------------------')
        print ('')

    check_user_inputs (params)
    return add_params (params) 

def extract_clinical_simulation_settings (content):

    ''' Function to extract clinical and simulation settings. The returned variable must
        should contain the following keys and their values:
                    * 'isPreemptive'
                    * 'traffic'
                    * 'fractionED'
                    * 'nRadiologists'
                    * 'meanServiceTimeInterruptingMin'
                    * 'nTrials'
                    * 'nPatientsTarget'
                    * 'verbose'
                    * 'statsFile'
                    * 'runTimeFile'
                    * 'plotPath'
        
        inputs 
        ------
        content (dict): Dictionary with title keys of "Clinical setting" and "Simulation setting"

        outputs
        -------
        inputs (dict): parameters and their values 
    '''

    ## Extract clinical and simulation settings as inputs
    inputs = {line.split()[0]:line.split()[1] for line in content['Clinical setting']}
    inputs.update ({line.split()[0]:line.split()[1] for line in content['Simulation setting']})

    #  Check the inputs and change value types accordingly based on parameters name
    for key, value in inputs.items():
        # These keys should be float
        if key in ['traffic', 'fractionED', 'meanServiceTimeInterruptingMin']:
            inputs[key] = float (value)
        # These keys should be integers
        if key in ['nRadiologists', 'nTrials', 'nPatientsTarget']:
            inputs[key] = int (value)
        # These keys should be boolean
        if key in ['isPreemptive', 'verbose']:
            if not inputs[key] in ['True', 'False']:
                raise IOError ('ERROR: {0} must be either True or False.'.format (key))
            inputs[key] = eval (value)
        # These keys may be None
        if key in ['rocFile', 'runTimeFile', 'plotPath']:
            if value == 'None': inputs[key] = None

    return inputs    

def extract_group_and_disease_parameters (groupDiseaseParameters):

    ''' Function to extract parameters related to groups and target diseases
        
        inputs 
        ------
        groupDiseaseParameters (array): every non-empty lines in "Group and disease parameters"

        outputs
        -------
        diseaseGroups (dict): parameters and their values by group
                              e.g. {'GroupCT':{'groupProb':0.4, 'diseaseNames':['A'],
                                               'diseaseRanks':[1], 'diseaseProbs':[0.3]},
                                    'GroupUS':{'groupProb':0.6, 'diseaseNames':['B'],
                                               'diseaseRanks':[2], 'diseaseProbs':[0.6]}}
        meanServiceTimes (dict): radiologists' service time by groups and diseases
                                 e.g. {'GroupCT':{'A':10, 'non-diseased':7},
                                       'GroupUS':{'B':6, 'non-diseased':7}}
    '''

    ## Extract each sub-section by group name
    diseaseGroups, meanServiceTimes = {}, {}

    currentGroup = None
    for line in groupDiseaseParameters:
        # Detect/update the current group - only one word in this line
        if not ' ' in line.strip():
            currentGroup = line
            diseaseGroups[currentGroup] = {'diseaseNames':[], 'diseaseProbs':[], 'diseaseRanks':[]}
            meanServiceTimes[currentGroup] = {}
            continue
        ## For the rest, the first element is the parameter name followed by the values
        parameter, values = line.split (' ', 1)
        parameter = parameter.strip()
        values = values.strip()
        #  "disease" has 3 values
        if parameter=='disease':
            name, rank, prob, readTime = [v.strip() for v in values.split (' ') if len (v.strip())>0]
            diseaseGroups[currentGroup]['diseaseNames'].append (name.strip())
            diseaseGroups[currentGroup]['diseaseRanks'].append (int (rank))
            diseaseGroups[currentGroup]['diseaseProbs'].append (float (prob))
            meanServiceTimes[currentGroup][name] = float (readTime)
            continue
        #  "groupProb" has 1 value 
        if parameter=='groupProb':
            diseaseGroups[currentGroup]['groupProb'] = float (values)
            continue
        #  "meanServiceTimeNonDiseasedMin" has 1 value
        meanServiceTimes[currentGroup]['non-diseased'] = float (values)

    return diseaseGroups, meanServiceTimes    

def extract_cadt_diagnostic_performance (cadtPerformance):

    ''' Function to extract parameters related to CADt AI diagnostic performance
        
        inputs 
        ------
        cadtPerformance (array): every non-empty lines in "CADt AI diagnostic performance"

        outputs
        -------
        AIinfo (dict): parameters and their values by each AI
                       e.g. {'Vendor1':{'groupName':'GroupCT', 'targetDisease':'A',
                                        'TPFThresh':0.95, 'FPFThresh':0.15, 'rocFile':None}}
    '''

    ## Extract each sub-section by AI name
    AIinfo = {}

    currentAI = None
    for line in cadtPerformance:
        # Detect/update the current AI - only one word in this line
        if not ' ' in line.strip():
            currentAI = line
            AIinfo[currentAI] = {}
            continue
        ## For the rest, the first element is the parameter name followed by 1 value
        parameter, value = [l.strip() for l in line.split (' ') if len (l.strip())>0]
        parameter = parameter.strip()
        value = value.strip()
        #  "groupName", "targetDisease" are strings
        if parameter in ['groupName', 'targetDisease']:
            AIinfo[currentAI][parameter] = value
            continue
        #  "TPFThresh" is a float
        if parameter == 'TPFThresh':
            AIinfo[currentAI][parameter] = float (value)
            continue
        #  "FPFThresh" may be a float or None
        if parameter == 'FPFThresh':
            value = None if value.lower() == 'none' else float (value)
            AIinfo[currentAI][parameter] = value
            continue
        #  "rocFile" may be a string or None
        if parameter == 'rocFile':
            if value.lower() == 'none': value = None
            AIinfo[currentAI][parameter] = value

    return AIinfo    

def read_configFile (configFile):

    ''' Function with all user input values for user to feed to simulation software.

        input
        -----
        configFile (str): path and filename of the user input file
                          For an example, see ../../inputs/config.dat
        
        outputs
        -------
        inputs (dict): all user inputs from config file
    '''

    with open (configFile, 'r') as f:
        config = [line.strip() for line in f.readlines()]
    f.close ()

    ## Extract each section
    content = {}
    currentTitle = None
    for line in config:
        # Skip this line if empty
        if len (line.strip())==0: continue
        # Detect/update the current title as I scan through each line
        isTitle = numpy.array ([title in line for title in configTitles])
        if isTitle.any():
            currentTitle = configTitles[isTitle][0]
            content[currentTitle] = []
            continue
        # If current title is None, it is the beginning of the config file
        if currentTitle is None: continue
        # Skip line if it starts with '#'
        line = line.strip()
        if line[0] == '#': continue
        # Append this line to this section according to current title
        content[currentTitle].append (line.split('#')[0].strip())

    ## Extract clinical and simulation settings as inputs
    inputs = extract_clinical_simulation_settings (content)
    inputs['AIinfo'] = extract_cadt_diagnostic_performance (content['CADt AI diagnostic performance'])

    diseaseGroups, meanServiceTimes = extract_group_and_disease_parameters (content['Group and disease parameters'])
    meanServiceTimes['interrupting'] = inputs['meanServiceTimeInterruptingMin']
    del inputs['meanServiceTimeInterruptingMin']
    inputs['diseaseGroups'] = diseaseGroups
    inputs['meanServiceTimes'] = meanServiceTimes

    return inputs

def get_ppvs_npvs (aDiseaseTree):
    
    ''' Each AI algorithm only detects one disease. Even if an AI device claims to
        detect multiple diseases, there are several underlying algorithm - one per
        disease. 

        For each algorithm targeting at one disease, we have a ppv. For a group
        with multiple AIs, the ppv and npv are calculated individually per AI based
        on the targeted disease only. PPV and NPV is based on disease prevalence
        within the group. This aligns with values typically reported by sponsors
        in a submission.

        input
        -----
        aDiseaseTree (diseaseTree): a diseaseTree instance that encapsulates
                                    all group/disease/AI/reading time info.                

        output
        ------
        probs (dict): PPV and NPV for all AIs involved
                      e.g. {'ppv': {'groupA':{'AI_A1':ppv1, 'AI_A2':ppv2},
                                    'groupB':{'AI_B1':ppv3}              },
                            'npv': {'groupA':{'AI_A1':npv1, 'AI_A2':npv2},
                                    'groupB':{'AI_B1':npv3}              }}
    '''
    
    probs = {'ppv':{}, 'npv':{}}

    for aGroup in aDiseaseTree.diseaseGroups:
        # Gather all the AIs in this group
        AIinGroup = aGroup.AIs
        # Positive/negative probabilities are valid quantities only if there is
        # at least one AI in the group.
        if len (AIinGroup) == 0: continue
        # Get group name for later use
        groupName = aGroup.groupName
        # Set up dict for this group
        for probkey in probs.keys():
            if not groupName in probs[probkey]: probs[probkey][groupName] = {}  
        # Each AI has one target disease. Loop through AI to get prob for its
        # target disease.
        for anAI in AIinGroup:
            Se, Sp = anAI.SeThresh, anAI.SpThresh
            AIName = anAI.AIname     
            # Figure out the prevalence of the target disease in this group.
            for aDisease in aGroup.diseases:
                if aDisease.diseaseName == anAI.targetDisease:
                    prevalence = aDisease.diseaseProb
            # With prevalence, Se, Sp, calculate probabilities within group.
            probs['ppv'][groupName][AIName] = get_ppv (prevalence, Se, Sp)
            probs['npv'][groupName][AIName] = get_npv (prevalence, Se, Sp)
            
    return probs

def get_prob (aDiseaseTree):

    ''' There is a probability associated with each disease state that takes
        into account all AIs within the same group.
    
    
                                  AI-A
                                  ====
                             / A   TP
                            /                     
                        / + -- B   FP
                       /    \  
                      /      \ ND  FP
            / Group 1       
           / (AI-A)   \      / A   FN
          /            \    /
         /              \ - -- B   TN
        /                   \
       /                     \ ND  TN
       \
        \                              AI-C    AI-D
         \                             ====    ====
          \                        / C  TP      FP 
           \                      /
            \                 / + -- D  FP      TP
             \               /  \ \
              \             /    \ \ E  FP      FP
               \           /      \
                \         /        \ ND FP      FP
                 \ Group 2              
                  (AI-C,  \
                   AI-D)   \ 
                            \      / C  FN      TN
                             \    /
                              \ - -- D  TN      FN
                                \ \
                                 \ \ E  TN      TN
                                  \
                                   \ ND TN      TN
    
        The probability calculates here is the last layers i.e. Given a
        group and given a positive/negative subgroup, what is the prob
        of a disease state. For example, in group 1+, what is the prob
        of getting a truly diseased with A? Same for getting a truly
        diseased with B (that AI-A is not trained to identify)? And that
        for truly non-diseased patients?
    
        For a group with two AIs, P(+) of a disease includes the TP by
        one AI and FP by another AI. Assuming that all diseases within the
        group are uncorrelated and that there are enough patients for each
        disease condition subgroup, the positive probability is just the
        sum of P(+A) for all diseases. Therefore, for each AI in this group,
        we need to sum up the probabilities. In cases where one or more
        disease conditions in the group do not have a dedicated AI to look
        for that condition, no extra term is needed; its effect is included
        in the (1-Sp)*(1-pA) term. Note that there is an extra factor of 2
        (or more depending on the number of AIs involved in the group) bec
        the same patient is viewed twice, each by a different AI.
    
        input
        -----
        aDiseaseTree (diseaseTree): a diseaseTree instance that encapsulates
                                    all group/disease/AI/reading time info.                

        output
        ------
        probs (dict): Given a group and given a positive/negative subgroup,
                      what is the prob of a disease state. Based on the
                      above example, the output of this function has this
                      structure.
                      e.g. {'group1':{'+':{diseaseA'    : {'AI-A_TP':p1},
                                           diseaseB'    : {'AI-A_FP':p2},
                                           non-diseased': {'AI-A_FP':p3}},
                                     {'-':{diseaseA'    : {'AI-A_FN':p4},
                                           diseaseB'    : {'AI-A_TN':p5},
                                           non-diseased': {'AI-A_TN':p6}}},
                            'group2':{'+':{diseaseC'    : {'AI-C_TP':p7 , 'AI-D_FP':p15},
                                           diseaseD'    : {'AI-C_FP':p8 , 'AI-D_TP':p16},
                                           diseaseE'    : {'AI-C_FP':p9 , 'AI-D_FP':p17},
                                           non-diseased': {'AI-C_FP':p10, 'AI-D_FP':p18}},
                                     {'-':{diseaseC'    : {'AI-C_FN':p11, 'AI-D_TN':p19},
                                           diseaseD'    : {'AI-C_TN':p12, 'AI-D_FN':p20},
                                           diseaseE'    : {'AI-C_TN':p13, 'AI-D_TN':p21},
                                           non-diseased': {'AI-C_TN':p14, 'AI-D_TN':p22}}}}
    '''

    probs = {}

    for aGroup in aDiseaseTree.diseaseGroups:
        
        probs[aGroup.groupName] = {}
        diseases = aGroup.diseases
        diseaseProbs = {aDisease.diseaseName:aDisease.diseaseProb for aDisease in diseases}
        
        for priorityClass in ['positive', 'negative']:
            probs[aGroup.groupName][priorityClass] = {}
        
            for aDisease in diseases:
                prevalence = diseaseProbs[aDisease.diseaseName]
                probs[aGroup.groupName][priorityClass][aDisease.diseaseName] = {}
                ## If no AI in this group, positive prob = 0, negative prob= 1
                ## They all go to the negative queue
                if len (aGroup.AIs) == 0:
                    prob = 0 if priorityClass=='positive' else 1
                    probs[aGroup.groupName][priorityClass][aDisease.diseaseName]['noAI'] = prob*prevalence
                for anAI in aGroup.AIs:
                    ## Either true positive or false negative
                    if anAI.targetDisease == aDisease.diseaseName:
                        if priorityClass=='positive': # True positive
                            key = anAI.AIname + '_TP'
                            prob = prevalence*anAI.SeThresh
                        else: # False negative
                            key = anAI.AIname + '_FN'
                            prob = prevalence*(1-anAI.SeThresh)
                    else: ## Either false positive or true negative
                        if priorityClass=='positive': # False positive
                            key = anAI.AIname + '_FP'
                            prob = prevalence*(1-anAI.SpThresh)
                        else: # True negative
                            key = anAI.AIname + '_TN'
                            prob =  prevalence*anAI.SpThresh
                    probs[aGroup.groupName][priorityClass][aDisease.diseaseName][key] = prob
                    
    return probs

def get_positive_negative_probs (aDiseaseTree, probs):
    
    ''' This function gets the probability (per group) that a patient with
        disease name is called positive by any AIs involved in the group.
    
        inputs
        ------
        aDiseaseTree (diseaseTree): a diseaseTree instance that encapsulates
                                    all group/disease/AI/reading time info.
        probs (dict): Given a group and given a positive/negative subgroup,
                      what is the prob of a disease state. This is expected
                      to be the outputs of get_prob().
                      e.g. {'group1':{'+':{diseaseA'    : {'AI-A_TP':p1},
                                           diseaseB'    : {'AI-A_FP':p2},
                                           non-diseased': {'AI-A_FP':p3}},
                                     {'-':{diseaseA'    : {'AI-A_FN':p4},
                                           diseaseB'    : {'AI-A_TN':p5},
                                           non-diseased': {'AI-A_TN':p6}}},
                            'group2':{'+':{diseaseC'    : {'AI-C_TP':p7 , 'AI-D_FP':p15},
                                           diseaseD'    : {'AI-C_FP':p8 , 'AI-D_TP':p16},
                                           diseaseE'    : {'AI-C_FP':p9 , 'AI-D_FP':p17},
                                           non-diseased': {'AI-C_FP':p10, 'AI-D_FP':p18}},
                                     {'-':{diseaseC'    : {'AI-C_FN':p11, 'AI-D_TN':p19},
                                           diseaseD'    : {'AI-C_TN':p12, 'AI-D_FN':p20},
                                           diseaseE'    : {'AI-C_TN':p13, 'AI-D_TN':p21},
                                           non-diseased': {'AI-C_TN':p14, 'AI-D_TN':p22}}}}

        output
        ------
        newProbs (dict): Probability that a patient of a disease condition is
                         positive by any AIs in the queue.
                         e.g. {'group1':{'+':{'diseaseA'    :{'is_positive':b1},
                                              'diseaseB'    :{'is_positive':b2},
                                              'non-diseased':{'is_positive':b3}},
                                         '-':{'diseaseA'    :{'is_negative':b4},
                                              'diseaseB'    :{'is_negative':b5},
                                              'non-diseased':{'is_negative':b6}}},
                               'group2':{'+':{'diseaseC'    :{'is_positive':b7},
                                              'diseaseD'    :{'is_positive':b8},
                                              'diseaseE'    :{'is_positive':b9},
                                              'non-diseased':{'is_positive':b10}},
                                         '-':{'diseaseC'    :{'is_negative':b11},
                                              'diseaseD'    :{'is_negative':b12},
                                              'diseaseE'    :{'is_negative':b13},
                                              'non-diseased':{'is_negative':b14}}}}
    '''
    
    newProbs = deepcopy (probs)
    
    for aGroup in aDiseaseTree.diseaseGroups:
        groupName = aGroup.groupName
        nAI = len (aGroup.AIs)
        
        for aDisease in aGroup.diseases:
            
            diseaseName = aDisease.diseaseName
            diseaseProb = aDisease.diseaseProb

            for priorityClass in ['positive', 'negative']:
                
                if nAI <= 1:
                    prob_isClass = list (probs[groupName][priorityClass][diseaseName].values())[0]
                else:
                    summed = sum (probs[groupName][priorityClass][diseaseName].values())
                    multiplied = numpy.product (list (probs[groupName][priorityClass][diseaseName].values()))
                    prob_isClass = summed - multiplied/diseaseProb**(nAI-1) if priorityClass == 'positive' else \
                                   multiplied/diseaseProb**(nAI-1)
                newProbs[groupName][priorityClass][diseaseName]['is_'+priorityClass] = prob_isClass 
                
    return newProbs

def get_isPos_isNeg (aDiseaseTree, probs):

    ''' This is different from get_positive_negative_probs() in a sense
        that this function outputs the overall probability of a patient
        being negative or positive (regardless of the disease conditions
        or disease truth or group that it belongs to).

        inputs
        ------
        aDiseaseTree (diseaseTree): a diseaseTree instance that encapsulates
                                    all group/disease/AI/reading time info.
        probs (dict): Probability that a patient of a disease condition is
                      positive by any AIs in the queue. This is expected
                      to be the outputs of get_positive_negative_probs().
                      e.g. {'group1':{'+':{'diseaseA'    :{'is_positive':b1},
                                           'diseaseB'    :{'is_positive':b2},
                                           'non-diseased':{'is_positive':b3}},
                                      '-':{'diseaseA'    :{'is_negative':b4},
                                           'diseaseB'    :{'is_negative':b5},
                                           'non-diseased':{'is_negative':b6}}},
                            'group2':{'+':{'diseaseC'    :{'is_positive':b7},
                                           'diseaseD'    :{'is_positive':b8},
                                           'diseaseE'    :{'is_positive':b9},
                                           'non-diseased':{'is_positive':b10}},
                                      '-':{'diseaseC'    :{'is_negative':b11},
                                           'diseaseD'    :{'is_negative':b12},
                                           'diseaseE'    :{'is_negative':b13},
                                           'non-diseased':{'is_negative':b14}}}}
                                           
        output
        ------
        PosNegProb (dict): probability that a patient being negative or positive
                           (regardless of the disease conditions or disease truth
                           or group that it belongs to).
                           e.g. {'positive':q1, 'negative':q2}
    '''

    PosNegProb = {}
    
    for priorityClass in ['positive', 'negative']:
        prob = 0
        for aGroup in aDiseaseTree.diseaseGroups:
            # Get the group information
            groupProb = aGroup.groupProb
            groupName = aGroup.groupName
            # Gather positive/negative prob
            summed = sum ([probs[groupName][priorityClass][aDisease.diseaseName]['is_' + priorityClass]
                           for aDisease in aGroup.diseases])
            prob += summed*groupProb
        PosNegProb[priorityClass] = prob

    return PosNegProb

def get_mu (aDiseaseTree, probs, probs_pos_neg, probs_ppv_npv, mus, doNeg=False):
    
    ''' Function to calculate 1. effective reading time for this priority class and
        2. probability that a patient belongs to this priority class with respect to
        all patients (priority class: positive or negative). 

        inputs
        ------
        aDiseaseTree (diseaseTree): a diseaseTree instance that encapsulates
                                    all group/disease/AI/reading time info.
        probs (dict): Given a group and given a positive/negative subgroup,
                      what is the prob of a disease state. This is expected
                      to be the outputs of get_prob().
        probs_pos_neg (dict): probability that a patient being negative or positive
                              (regardless of the disease conditions or disease truth
                              or group that it belongs to). This is expected to be
                              the outputs of get_isPos_isNeg().
        probs_ppv_npv (dict): PPV and NPV for all AIs involved. This is expected to be
                              the outputs of get_ppvs_npvs().
        mus (dict): service rate (i.e. 1/meanReadTime) 
                    e.g. {'GroupCT':{'A':1/10, 'non-diseased':1/7},
                          'GroupUS':{'F':1/6, 'non-diseased':1/7}}
        doNeg (bool): True if calculation is for AI-negative. False if for AI-positive.

        outputs
        -------
        effective mu (float): effective reading rate [1/min] for this priority class
        condProbs (dict): probability that a patient belongs to this priority
                          class with respect to all patients
    '''

    meanReadingTime = 0
    priorityClass = 'negative' if doNeg else 'positive'
    condProbs = {}

    for aGroup in aDiseaseTree.diseaseGroups:

        if not doNeg and len (aGroup.AIs) == 0: continue
        
        # Get the group information
        groupProb = aGroup.groupProb
        groupName = aGroup.groupName
        nAI = len (aGroup.AIs)        
        if not groupName in condProbs: condProbs[groupName] = {}
        
        # Gather the probabilities for AI label subgroup
        p_pclass_group = sum ([probs[groupName][priorityClass][aDisease.diseaseName]['is_'+priorityClass]
                               for aDisease in aGroup.diseases])
        p_group_pclass = p_pclass_group * groupProb / probs_pos_neg[priorityClass]
        
        # For each disease, calculate the p/mu
        for aDisease in aGroup.diseases:
            # Get this disease name
            diseaseName = aDisease.diseaseName
            diseaseProb = aDisease.diseaseProb                        
            # Calculate the prob for this disease in this group
            pvs = []
            for anAI in aGroup.AIs:
                AIName = anAI.AIname
                targetDiseaseProb = [aDisease.diseaseProb for aDisease in aGroup.diseases
                                     if aDisease.diseaseName==anAI.targetDisease][0]
                
                pv = probs_ppv_npv['npv'][groupName][AIName] if doNeg else \
                     probs_ppv_npv['ppv'][groupName][AIName]
                
                if doNeg:
                    thisPV = 1-pv if anAI.targetDisease == diseaseName else \
                             pv*diseaseProb / (1-targetDiseaseProb)
                else:                
                    thisPV = pv if anAI.targetDisease == diseaseName else \
                             (1-pv)*diseaseProb / (1-targetDiseaseProb)
                
                pvs.append (thisPV)
            
            if nAI == 0:
                p_disease_group = diseaseProb
            elif nAI == 1:
                p_disease_group = pvs[0] 
            else:
                summed = numpy.sum (pvs)
                multiplied = numpy.product (pvs)
                            
                p_disease_group = summed - multiplied/diseaseProb if priorityClass == 'positive' else \
                                  multiplied/diseaseProb
        
            thisProb = p_group_pclass * p_disease_group
            condProbs[groupName][diseaseName] = thisProb
            # print ('+---------------------------------------------------------')
            # print ('{0} {1} {2}:'.format (groupName, diseaseName, priorityClass))
            # print ('  -- pvs: {0}'.format (pvs))
            # print ('  -- p_disease_group: {0}'.format (p_disease_group))
            # print ('  -- p_group_pclass: {0}'.format (p_group_pclass))
            # print ('  -- thisProb: {0}'.format (thisProb))
            mu = mus[groupName][diseaseName]
            meanReadingTime += thisProb/mu
        
    # Inverse to get the effective service rate
    return 1/meanReadingTime, condProbs

def get_muNonEm (prob_pos_neg, mus):

    ''' Function to compute effective reading rate for the entire queue.

        inputs
        ------
        probs_pos_neg (dict): probability that a patient being negative or positive
                        (regardless of the disease conditions or disease truth
                        or group that it belongs to). This is expected to be
                        the outputs of get_isPos_isNeg().
        mus (dict): effective reading rate for both priority class

        output
        ------
        effective mu (float): effective reading rate [1/min] for the entire queue
    '''

    meanReadingTime = 0

    for priorityClass in ['positive', 'negative']:
        mu = mus[priorityClass]
        prob = prob_pos_neg[priorityClass]
        meanReadingTime += prob/mu
        
    # Inverse to get the effective service rate
    return 1/meanReadingTime

def create_disease_tree (diseaseGroups, meanServiceTimes, AIs):

    ''' Function to create a disease tree instance that encapsulate all
        the group/disease information including group probability, disease
        prevalence, mean reading times, which AI associate to which group
        and disease, and their operating thresholds. 

        inputs
        ------
        diseaseGroups (dict): group information from config file
            e.g. {'GroupCT':{'groupProb':0.4, 'diseaseNames':['A'], 'diseaseProbs':[0.3]},
                  'GroupUS':{'groupProb':0.6, 'diseaseNames':['F'], 'diseaseProbs':[0.6]}}
        meanServiceTimes (dict): radiologists' service time by groups and diseases
                                 e.g. {'GroupCT':{'A':10, 'non-diseased':7},
                                       'GroupUS':{'F':6, 'non-diseased':7}}
        AIs (dict): directary of all AIs involved in the queue
                    e.g. {AIname: an AI object}
        
        outputs
        -------
        aDiseaseTree (diseaseTree): a diseaseTree instance that encapsulates
                                    all group/disease/AI/reading time info.
        
    '''

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

def create_hierarchy (diseaseDict, AIinfo):

    ''' Function to create a hierarchy of disease conditions.

        inputs
        ------
        diseaseDict (dict): group information from config file
            e.g. {'GroupCT':{'groupProb':0.4, 'diseaseNames':['A'],
                             'diseaseRanks':[1], 'diseaseProbs':[0.3]},
                  'GroupUS':{'groupProb':0.6, 'diseaseNames':['F', 'E'],
                             'diseaseRanks':[3, 2], 'diseaseProbs':[0.6, 0.1]}}
        AIinfo (dict): parameters and their values by each AI
                    e.g. {'Vendor1':{'groupName':'GroupCT', 'targetDisease':'A',
                                     'TPFThresh':0.95, 'FPFThresh':0.15, 'rocFile':None}}
        
        output
        ------
        aHierarchy (hierarhcy): info about hierarchical-priority queue.
    '''

    aHierarchy = hierarchy.hierarchy()
    aHierarchy.build_hierarchy (diseaseDict, AIinfo)
    return aHierarchy

def add_params (params, include_theory=True):

    ''' Function to add additional parameters from user inputs. This includes
        probabilities that the next patient belongs to a certain priority class,
        as well as the arrival and service rates.

        inputs
        ------
        params (dict): dictionary with user inputs
        include_theory (bool): if true, also calculate analytical mean wait-time

        output
        ------
        params (dict): dictionary capsulating all simulation parameters
                       and analytical time-savings.
        anAI (AI): CADt with a diagnostic performance from user input
        aDiseaseTree (diseaseTree): a diseaseTree instance that encapsulates
                                    all group/disease/AI/reading time info.
    '''

    ## Create an array of AI objects, a disease tree, and a hierarchy
    AIs = {AIname:create_AI (AIname, AIinfo, doPlots=params['doPlots'], plotPath=params['plotPath'])
           for AIname, AIinfo in params['AIinfo'].items()}
    
    aDiseaseTree = create_disease_tree (params['diseaseGroups'],
                                        params['meanServiceTimes'], AIs)
    
    aHierarchy = create_hierarchy (params['diseaseGroups'], params['AIinfo'])
    params['hierDict'] = aHierarchy.hierDict

    ## Probabilities
    params['SeThreshs'] = {AIname:anAI.SeThresh for AIname, anAI in AIs.items()}
    params['SpThreshs'] = {AIname:anAI.SpThresh for AIname, anAI in AIs.items()}
    params['prob_isInterrupting'] = params['fractionED']
    params['prob_isDiseased'] = aDiseaseTree.get_diseased_prevalence()*(1-params['fractionED'])
    params['prob_isNonDiseased'] = aDiseaseTree.get_nondiseased_prevalence()*(1-params['fractionED'])    
    
    probs = get_prob (aDiseaseTree)
    params['probs_ppv_npv'] = get_ppvs_npvs (aDiseaseTree)
    params['probs'] = get_positive_negative_probs (aDiseaseTree, probs)
    params['probs_pos_neg'] = get_isPos_isNeg (aDiseaseTree, params['probs'])
    params['prob_isPositive'] = params['probs_pos_neg']['positive']*(1-params['fractionED'])
    params['prob_isNegative'] = params['probs_pos_neg']['negative']*(1-params['fractionED'])
    
    ## Service rates
    params['serviceRates'] = {key: get_service_rates (value) for key, value in params['meanServiceTimes'].items()}
    params['mus'] = params['serviceRates']
    params['probs_condByPriority'] = {}
    params['mus']['positive'], params['probs_condByPriority']['positive'] = get_mu (aDiseaseTree, params['probs'],
                                                                                    params['probs_pos_neg'],
                                                                                    params['probs_ppv_npv'],
                                                                                    params['mus'], doNeg=False)
    params['mus']['negative'], params['probs_condByPriority']['negative'] = get_mu (aDiseaseTree, params['probs'],
                                                                                    params['probs_pos_neg'],
                                                                                    params['probs_ppv_npv'],
                                                                                    params['mus'], doNeg=True)
    params['mus']['non-interrupting'] = get_muNonEm (params['probs_pos_neg'], params['mus'])
    params['mu_effective'] = get_mu_effective (params)
    
    ## Arrival rates
    params['lambda_effective'] = get_lambda_effective (params)
    params['meanArrivalTime'] = 1/params['lambda_effective']
    params['arrivalRates'] = {'interrupting':params['prob_isInterrupting']*params['lambda_effective'],
                              'non-interrupting':(1-params['prob_isInterrupting'])*params['lambda_effective'],
                              'positive':params['prob_isPositive']*params['lambda_effective'],
                              'negative':params['prob_isNegative']*params['lambda_effective']}             
    
    ## Simulation times
    nPatientsPerTrial = params['nPatientsTarget'] + sum (params['nPatientsPads'])
    params['timeWindowDays'] = get_timeWindowDay (params['lambda_effective'], nPatientsPerTrial)
    params['endTime'] = params['startTime'] + pandas.offsets.Day (params['timeWindowDays'])
    
    ## Setting per priority class    
    params['lambdas'] = params['arrivalRates']
    params['rhos'] = {key: params['lambdas'][key]/params['mus'][key]/params['nRadiologists']
                      for key in params['lambdas'].keys()}

    ## Get theoretical waiting time and delta time (i.e. wait-time-saving)
    if include_theory:
        params['theory'] = calculator.get_theory_waitTime (params, aHierarchy) 

    return params, AIs, aDiseaseTree
