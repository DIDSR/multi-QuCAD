##
## By Elim Thompson (05/21/2024)
##
## This script encapsulates all properties/calculations of a hierarchical queue
## based on Rucha and Michelle works. This class also includes the calculation
## of hierarchical queue. This calculation would not work if ...
##  * if multiple diseased in a group (e.g. A and B) but there are more than 1 AI.
###################################################################################

################################
## Import packages
################################ 
import numpy
from copy import deepcopy

from . import calculator, inputHandler

################################
## Define constants
################################
## Priority = 1  -> interrupting
## Priority = 2  -> AI-positive in non-hierarchical queue
## Priority = 3+ -> AI-positive in hierarchical queue
## Priority = 99 -> AI-negative
start_priority = 3

################################
## Define hierarchy
################################
class hierarchy (object):

    ''' A hierarchy object that encapsulates properties to define
        a hierarchical queue.
    '''

    def __init__ (self):       
        
        self._diseaseNames = None
        self._groupNames = None
        self._AINames = None
        self._hierDict = {}

    @property
    def diseaseNames (self): return self._diseaseNames
    @property
    def groupNames (self): return self._groupNames
    @property
    def AINames (self): return self._AINames
    @property
    def diseaseNamesWithAIs (self): return self.diseaseNames[self.AINames!=None]
    @property
    def groupNamesWithAIs (self): return self.groupNames[self.AINames!=None]
    @property
    def AINamesWithAIs (self): return self.AINames[self.AINames!=None]
    @property
    def hierDict (self): return self._hierDict

    def build_hierarchy (self, diseaseDict, AIinfo):

        ''' Public function to define hierarchy for disease names,
            group names, AI names based on disease rank.

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
        '''

        order, diseaseNames, groupNames, AINames = [], [], [], []

        for groupname, aGroup in diseaseDict.items():
            order.extend(aGroup['diseaseRanks'])
            diseaseNames.extend(aGroup['diseaseNames'])

            for diseaseName in aGroup['diseaseNames']:

                groupNames.append(groupname)

                thisAI = [ainame for ainame, aiinfo in AIinfo.items() if aiinfo['targetDisease']==diseaseName]
                aiName = None if len (thisAI) == 0 else thisAI[0]
                AINames.append(aiName)
        
        order = numpy.argsort (order)
        diseaseNames = numpy.array (diseaseNames)
        groupNames = numpy.array (groupNames)
        AINames = numpy.array (AINames)

        self._diseaseNames = diseaseNames[order]       
        self._groupNames = groupNames[order]
        self._AINames = AINames[order]

        ## Build a hierarchy class dictionary
        self._hierDict = {ainame : o + start_priority
                          for ainame, o in zip (self._AINames, order[order])}
        
    def _calculate_equivalent_Se_Sp (self, se_array, sp_array, all_pop_d_prevalence, all_groupProbs):
        
        ''' Private function to compute equivalent sensitivity and specificity
            from the lists of Se and Sp corresponding to the AIs to be combined.
            This function is only valid if disease groups for all AIs are independent
            and if there is at most 1 AI in each group.

            As an example, if only 1 AI in each group,
            e.g. GroupCTA has disease A & B with 1 AIa looking for A and
                 GroupCX  has disease C with 1 AIc looking for C
                 GroupXR  has disease D with no AI
            And we are interested in [GroupCTA and GroupCX] (high periority) vs
            GroupXR (low-priority), the equivalent Se is ...
        
                  #A+byAIa + #C+byAIc     #A+byAIa/#A * #A/N + #C+byAIc/#C * #C/N  
            Se = --------------------- = ------------------------------------------ 
                        #A + #C                         #A/N + #C/N                

                  SeA * piA + SeC * piC
               = -----------------------
                        piA + piC

            If multiple AIs in each group, e.g. GroupCTA also has AIb looking for B,
            then the numerator needs to include cases with AIa falsely flagged B-diseased
            cases and vice versa.
        
            For Sp, the "ND"-equivalent are the truly non-diseased patients *and* the
            conditions that no AI looks for.

                  #(B and ND)-byAIa + #ND-byAIc     
            Sp = -------------------------------
                           N - #A - #C             
        
                  #(B and ND)-byAIa/#(B and ND) * #(B and ND)/#CTA * #CTA/N + #ND-byAIc/#ND * #ND/#CX * #CX/N
               = ---------------------------------------------------------------------------------------------
                                                            1 - piA - piC

                  SpA * (1-piA) * groupProbCTA + SpC * (1-piC) * groupProbCX
               = ------------------------------------------------------------
                                      1 - piA - piC
                    
            where groupProbCTA, groupProbCX, piA and piC are with respect to number of
            cases in group CTA and CX (groups that the AIs in high-priority are involved).

            inputs
            ------
            se_array (array): sensitivity of the AIs involved        
            sp_array (array): specificity of the AIs involved
            all_pop_d_prevalence (array): disease prevalences with respect to the entire population
            all_groupProbs (array): group probabilities
            Note: Each element of all arrays must correspond to the same disease name
            
            outputs
            -------
            equt_se (float): Equivalent sensitivity considering all AIs
            equt_sp (float): Equivalent specificity considering all AIs
        '''

        eqvt_se = numpy.sum ([ se_array[i] * all_pop_d_prevalence[i]/all_groupProbs[i] 
                                * all_groupProbs[i]/numpy.sum (all_groupProbs)
                                for i in range (len (se_array))])
        eqvt_se /= numpy.sum (all_pop_d_prevalence)/numpy.sum (all_groupProbs)

        eqvt_sp = numpy.sum ([ sp_array[i] * (1-all_pop_d_prevalence[i]/all_groupProbs[i]) 
                                * all_groupProbs[i]/numpy.sum (all_groupProbs)
                                for i in range (len (se_array))])
        eqvt_sp /= 1 - numpy.sum (all_pop_d_prevalence)/numpy.sum (all_groupProbs)

        return eqvt_se, eqvt_sp
    
    def _calculate_equivalent_readtime (self, meanServiceTimes, diseaseGroups, AIinfo, groups_wAI):

        ''' Private function to compute equivalent reading time when the AIs are
            combined. Disease 'Z' is composed of all disease conditions that the
            AIs are involved in the equivalent group.
        
            As an example, if only 1 AI in each group,
            e.g. GroupCTA has disease A & B with 1 AIa looking for A and
                 GroupCX  has disease C with 1 AIc looking for C
                 GroupXR  has disease D with no AI
        
            Let's say, Z is made of A and C. and ND is B and ND-CTA and ND-CX. The
            read-time for Z (TZ) is... 
                1/TZ = pA / TA + pC / TC
            where pA = A / (A+C) and TA is the mean read time for A diseased cases
            Note that the probabilities (pA and pC) are with respect to the sum of
            cases with the disease conditions of interests i.e. A and C only. 

            The equivalent read-time for 'non-diseased' depends on disease conditions
            that may or may not have an AI involved, as well as the non-diseased in
            multipled groups that have different group probabilities.

            inputs
            ------
            meanServiceTimes (dict): original mean service time provided by the users
                                     for all diseases in all groups
            diseaseGroups (dict): groups of diseases with their rank, probabilities
            AIinfo (dict): AI info provided with all AIs of interests
            groups_wAI (list): a list of group names that have AI involved.

            outputs
            -------
            TZ (float): mean reading time of 'diseased' for group Z (equivalent group)
            TNZ (float): mean reading time for 'non-diseased' group Z (equivalent group)
            * Note: 'diseased' here refers to the disease conditions that the AI(s)
                    of interests are trained to identify. 'non-diseased' refers to 
                    disease conditions that do not have any AI or have AI that are
                    not in the list of AIs of interests.
        '''

        ## For each AI of interests, pull out the target disease and group name
        diseases, groupsdis = [], []
        for _, aiinfo in AIinfo.items():
            diseases.append (aiinfo['targetDisease'])
            groupsdis.append (aiinfo['groupName'])

        ## A normalization for the probability to only include the groups 
        ## that have the AIs of interests
        normProb = sum ([diseaseGroups[gp]['groupProb'] for gp in numpy.unique (groupsdis)])

        ## To obtain the mean read time for the 'diseased' patients, loop through
        ## each target disease and get the disease prob w.r.t. the 'diseased' population.
        probs, Ts = [], []
        for disease, group in zip (diseases, groupsdis):
            dis_idx = numpy.where (numpy.array (diseaseGroups[group]['diseaseNames'])==disease)[0][0]
            prob = diseaseGroups[group]['diseaseProbs'][dis_idx]* diseaseGroups[group]['groupProb'] / normProb
            probs.append (prob)
            Ts.append (meanServiceTimes[group][disease])
        probs = numpy.array (probs)
        Ts = numpy.array (Ts)
        ps = probs / numpy.sum (probs)
        TZ = 1/ numpy.sum ([p/T for p, T in zip (ps, Ts)])

        ## For non-diseased, need to go through all groups that have the AIs involved
        ## and get the disease names and group names. For non-diseased, make sure we
        ## differentiate the different 'non-diseased' in different groups because the
        ## group probabilities can be different.
        nondiseases, groupsdis = [], []
        for gpname, group in diseaseGroups.items():
            ## Ignore groups without AI of interests
            if not gpname in groups_wAI: continue
            for disname in group['diseaseNames']:
                ## Ignore those that has an AI to identify
                if disname in diseases: continue
                nondiseases.append (disname)
                groupsdis.append (gpname)
            ## Now deal with non-diseased
            disname = 'non-diseased_' + gpname
            nondiseases.append (disname)
            groupsdis.append (gpname)

        ## A normalization for the probability for the nondiseased 
        normProb = sum ([diseaseGroups[gp]['groupProb'] for gp in numpy.unique (groupsdis)])

        ## To obtain the mean read time for the 'non-diseased' patients, loop
        ## through each non-target disease and get the disease prob w.r.t. the
        ## 'non-diseased' population. For non-diseased, the prob is 1 minus
        ## sum of all disease prevalences within the group.
        probs, Ts = [], []
        for nondisease, group in zip (nondiseases, groupsdis):
            if nondisease in diseaseGroups[group]['diseaseNames']:
                dis_idx = numpy.where (numpy.array (diseaseGroups[group]['diseaseNames'])==nondisease)[0][0]
                prob = diseaseGroups[group]['diseaseProbs'][dis_idx]
            else:
                prob = 1 - sum (diseaseGroups[group]['diseaseProbs'])
            probs.append (prob* diseaseGroups[group]['groupProb'] / normProb)

            nondiseasename = 'non-diseased' if 'non-diseased' in nondisease else nondisease
            Ts.append (meanServiceTimes[group][nondiseasename])
        probs = numpy.array (probs)
        Ts = numpy.array (Ts)
        ps = probs / numpy.sum (probs)
        TNZ = 1/ numpy.sum ([p/T for p, T in zip (ps, Ts)])    

        return TZ, TNZ
    
    def _update_newParams (self, newParams, keep_ai): 
        
        ''' Private function to compute and return params after creating equivalent
            AIs from multiple AIs. Equivalent Se, Sp and disease group probabilities
            are computed. Other parameters may be updated just for consistency in
            access to params[variables].

            inputs
            ------
            newParams (dict): original params dict that will be updated
            groups_wAI (list): a list of group names that have AI involved.
            groups_noAI (list): a list of group names that does not have AI involved.
            
            output
            ------
            newParams (dict): params dict that is updated            
        '''

        if keep_ai is not 'all':
            disease = keep_ai[-1]
        else:
            disease = None

        ## Loop through all groups, get info for each AI-positive subgroup
        arr = list(newParams['meanServiceTimes'].keys())
        arr = arr[:-1]   

        new_lambdas = {'interrupting': newParams['lambdas']['interrupting'],
                       'non-interrupting': newParams['lambdas']['non-interrupting'],
                       'positive': newParams['lambdas']['positive'],
                       'negative': newParams['lambdas']['negative']}
        new_rhos = {'interrupting': newParams['rhos']['interrupting'],
                    'non-interrupting': newParams['rhos']['non-interrupting'],
                    'positive': newParams['rhos']['positive'],
                    'negative': newParams['rhos']['negative']}

        for groupname in newParams['diseaseGroups']:
            for diseasename in newParams['diseaseGroups'][groupname]['diseaseNames']:
                if diseasename == disease:
                    new_lambdas['positive'] = newParams['lambdas'][f'{diseasename}_H']
                    new_rhos['positive'] = new_lambdas['positive']/newParams['mus']['positive']
                if disease == None:
                    new_lambdas['positive'] = newParams['lambdas']['positive']
                    new_rhos['positive'] = newParams['lambdas']['positive']/newParams['mus']['positive']

        for groupname in newParams['diseaseGroups']:
            for diseasename in newParams['diseaseGroups'][groupname]['diseaseNames']:
                if diseasename == disease:
                    new_lambdas['negative'] = newParams['lambdas'][f'{diseasename}_L']
                    new_rhos['negative'] = new_lambdas['negative']/newParams['mus']['negative']
                if disease == None:
                    new_lambdas['negative'] = newParams['lambdas']['non-interrupting'] * newParams['prob_AI_neg_group']
                    new_rhos['negative'] = newParams['lambdas']['non-interrupting'] * newParams['prob_AI_neg_group']/newParams['mus']['negative']

        ## Actually update the dictionary
        newParams['lambdas'] = new_lambdas
        newParams['rhos'] = new_rhos
        
        return newParams

    def _get_equivalent_params (self, params, keep_ai='all'):
        
        ''' Private function to get a set of params for equivalent group that involves
            the AIs of interests. This function will be called multiplet times for each
            equivalent partition. This method was proposed by Rucha.

            As an example, if only 1 AI in each group,
            e.g. GroupCTA has disease A & B with 1 AIa looking for A and
                 GroupCX  has disease C with 1 AIc looking for C
                 GroupXR  has disease D with no AI
                 Ranking: A > B > C > D

            Partitions correspond to the diseased prevalence and group prob

               GroupCTA                GroupCX                  GroupXR
            +--------------+----------------------------------+----------+
            |   "    "     |         "                        |   "      |
            | A " B  " ND  |    C    "          ND            | D "  ND  |
            |   "    "     |         "                        |   "      |
            +--------------+----------------------------------+----------+

            With 2 AIs for A and C, the cases in hierarchical order are below. Each
            big block is a priority class, and all cases within the big block are
            in FIFO (random arrival).

                  AI-A+               AI-C+                AI-A- and AI-C- and no-AI
            +---------------+------------------------+----------------------------------+
            | A  " B  " ND  | C    "       ND        | A  " B  " ND  " C  " ND " D " ND |
            | TP " FP " FP  | TP   "       FP        | FN " TN " TN  " FN " TN "   "    |
            |    "    " CTA |      "       CX        |    "    " CTA "    " CX "   " XR |
            +---------------+------------------------+----------------------------------+

            This function will be called twice.
            1. AI-A+ as high priority & the last two classes as low priority where the 
               input `keep_ai` = ['AI-A']
            2. (AI-A+ and AI-C+) as high priority & the last class as low priority where
               the input `keep_ai` = ['AI-A', 'AI-C']
            Making use of the "mean" wait-time, the absolute mean wait-time for A-patient
            is from #1, and that for C-patient is from (#2-#1)/number of diseased AI+.
            
            Each time it is called, a GroupEQ is created with an equivalent Se/Sp/read-
            time to represent the two priority classes.

            inputs
            ------
            params (dict): original params with user-defined settings
            keep_ai (list or 'all'): If a list of AI names, update AIinfo with just the
                                     AIs of interests. If 'all', consider all AIs.

            outputs
            -------
            params_out (dict): updated params with equivalent group
        '''

        ## Make a new copy to avoid contamination
        params_out = deepcopy (params) 

        params_out = self._update_newParams (params_out, keep_ai)

        return params_out

    def _define_high_low_classes (self, diseaseDivide):
        
        ''' Private function to divide AIs of different priority classes.
            Given the hierarchy of diseases with AI and the chosen disease,
            return two lists of AI: hi and lo -- these lists will be used
            to compute equivalent AIs.
            e.g, for chosen disease B and hierarchy A>B>C>D,
                 hi : [A] and lo: [A, B]

            input
            -----
            diseaseDivide (str): disease name at which AIs are divided

            outputs
            -------
            hi_AIs (array): AIs in the "high" priority class
            lo_AIs (array): AIs in the "low" priority class
        '''

        idx = list(self.diseaseNames).index(diseaseDivide)
        hi_AIs = self.diseaseNames[:idx]
        lo_AIs = self.diseaseNames[:idx+1]
        
        return hi_AIs, lo_AIs

    def _get_positive_rate(self, params, disease, diseaseOnly):

        ''' Private function to compute all positive rate as a sum of TP and
            FP. This is used to weight the positive patient groups for hi and
            lo subsets in the theoretical calculations. If group/AI/disease
            names are provided, it is calculated for a single disease and its
            probabilities are directly pulled from params['diseaseGroups'].
            Otherwise, use the probablilities from the equivalent diseases/ AIs.

            inputs
            ------
            params (dict): params dict to pull information from
            groupname (str): the group of interest
            AIname (str): the AI of interest
            diseasename (str): the disease of interest
            * Note: if group/AI/disease names are provided, the params is
                    expected to be the original one from users. Otherwise,
                    params should be the hi/lo ones.

            outputs
            -------
            positive_rate (float): sum of TP and FP.
        '''
        
        if diseaseOnly:
            for groupname in params['rankedGroups']:
                diseasenames = numpy.array (params['diseaseGroups'][groupname]['diseaseNames'])
                diseaseorders = numpy.argsort (params['diseaseGroups'][groupname]['diseaseRanks']) 
                for diseasename in diseasenames[diseaseorders]:
                    if diseasename == disease:
                        pos_rate = params['prob_pos_i_neg_higher_AIs'][groupname][diseasename]

            return pos_rate

        else:
            found = False
            pos_rate = 0
            for groupname in params['rankedGroups']:
                if found: break
                diseasenames = numpy.array (params['diseaseGroups'][groupname]['diseaseNames'])
                diseaseorders = numpy.argsort (params['diseaseGroups'][groupname]['diseaseRanks'])
                for diseasename in diseasenames[diseaseorders]:
                    if diseasename == disease:
                        pos_rate += params['prob_pos_i_neg_higher_AIs'][groupname][diseasename]
                        found = True
                        break
                    else:
                        pos_rate += params['prob_pos_i_neg_higher_AIs'][groupname][diseasename]

        return pos_rate

    def _predict (self, params, keep_ai='all', pclass='negative'):

        ''' Private function to predict mean wait time for the hierarchical
            equivalent priority class.

            inputs
            ------
            params (dict): original params with user-defined settings
            keep_ai (list or 'all'): If a list of AI names, update AIinfo with just the
                                     AIs of interests. If 'all', consider all AIs.
            pclass (str): either 'positive' or 'negative'

            output
            ------
            updatedParams (dict): params updated for the equivalent group
            meanWaitTime (float): mean wait time of this equivalent group in minutes
        '''

        updatedParams = self._get_equivalent_params (params, keep_ai=keep_ai)
        # To match the expected format in get_theory_waitTime
        updatedParams['SeThresh'] = list(updatedParams['SeThreshs'].values())[0] 
        updatedParams['SpThresh'] = list(updatedParams['SpThreshs'].values())[0]

        meanWaitTime = calculator.get_theory_waitTime_fifo_preresume (pclass, 'preresume', updatedParams)
        ## Another function if non-preemptive
        return updatedParams, meanWaitTime
    
    def predict_mean_waitTime_dis (self, params, aHierarchy, hier_flag):
        
        ''' Function to convert waiting time for all AI+ classes and AI- class in preresume priority and hierarchical scheme
            to wait-time for true-diseased patients. 

            inputs
            ------
            params (dict): dictionary with user inputs
            aHierarchy (hierarchy): hierarchy of ranked priority classes
            hier_flag: flag for priority scheme (False) or hierarchical scheme (True).

            outputs
            -------
            theories (dict): theoretical predictions for all disease classes.
        '''

        theories = {}

        probs_for_conversion = params['probs_for_waittime_conversion']

        if hier_flag == True:
            pos_neg_theories = self.get_all_waitTime_hier_posneg(params)
        else: 
            pos_neg_theories = self.get_all_waitTime_preresume_posneg(params)

        if hier_flag == True:
            for j in range(len(probs_for_conversion)):
                diseaseName = aHierarchy.diseaseNames[j]
                groupname = aHierarchy.groupNames[j]
            
                probs_for_conversion_thisdis = probs_for_conversion[diseaseName]
                disease_in_group_times = []
                for disease_in_group in params['diseaseGroups'][groupname]['diseaseNames']:
                    disease_in_group_times.append(pos_neg_theories[disease_in_group]['positive'])

                theories[diseaseName] = {}
                theories[diseaseName]['diseased'] = sum(disease_in_group_times * probs_for_conversion_thisdis) + pos_neg_theories[diseaseName]['negative'] * (1-sum(probs_for_conversion_thisdis))
        else:
            probs_for_conversion_thisdis = 0
            for j in range(len(probs_for_conversion)):
                diseaseName = aHierarchy.diseaseNames[j]
                groupname = aHierarchy.groupNames[j]
            
                probs_for_conversion_thisdis = probs_for_conversion[diseaseName]

                theories[diseaseName] = {}
                theories[diseaseName]['diseased'] = sum(probs_for_conversion_thisdis)*pos_neg_theories['positive'] + pos_neg_theories['negative'] * (1-sum(probs_for_conversion_thisdis))

        return theories

    def get_all_waitTime_preresume_posneg (self, params):

        ''' Function to calculate waiting time for all AI+ classes and AI- class in preresume priority scheme. 

            inputs
            ------
            params (dict): dictionary with user inputs
            aHierarchy (hierarchy): hierarchy of ranked priority classes

            outputs
            -------
            theories (dict): theorectical predictions for all priority and disease classes.
        '''
        theory_neg = calculator.get_theory_waitTime_fifo_preresume ('negative', 'preresume', params) 
        theory_pos = calculator.get_theory_waitTime_fifo_preresume ('positive', 'preresume', params) 

        priority = {}

        priority['positive'] = theory_pos
        priority['negative'] = theory_neg

        return priority

    def get_all_waitTime_hier_posneg (self, params):

        ''' Function to calculate waiting time for all AI+ classes and AI- class in preresume hierarchical scheme. 

            inputs
            ------
            params (dict): dictionary with user inputs
            aHierarchy (hierarchy): hierarchy of ranked priority classes

            outputs
            -------
            theories (dict): theorectical predictions for all priority and disease classes.
        '''

        ## For the lowst lowest priority class without an AI. This is the wait-time
        ## for all AI-negative patients and patients that has no AIs to review.
        _, theory_neg = self._predict (params, keep_ai='all', pclass='negative')

        ## Store the predicted wait-time for each disease/priority class
        theories = {}

        ## Now, loop through each disease condition to get the theorectical predictions
        #for disease in self.diseaseNames:
        for disease in params['rankedDiseases']:

            theories[disease] = {}

            HiAIs, LoAIs = self._define_high_low_classes (disease)
            ## If there is no HiAIs, i.e. working on the disease (with an AI) of
            ## the highest rank, just calculate the mean wait time of the low class.
            if len (HiAIs)==0:
                params_lo, theory_lo = self._predict (params, keep_ai=LoAIs, pclass='positive')
                theory_pos = theory_lo
                theories[disease]['positive'] = theory_pos
                theories[disease]['negative'] = theory_neg

            else:
                params_hi, theory_hi = self._predict (params, keep_ai=HiAIs, pclass='positive')
                params_lo, theory_lo = self._predict (params, keep_ai=LoAIs, pclass='positive')
                
                ## Get the rate of positive patients in each case: hi, lo, and chosen disease.
                loPosRate = self._get_positive_rate (params_lo, disease, diseaseOnly=False)
                hiPosRate = self._get_positive_rate (params_hi, prev_disease, diseaseOnly=False)                 
                posRate = self._get_positive_rate (params, disease, diseaseOnly=True) 
                
                theory_pos = (theory_lo*loPosRate - theory_hi*hiPosRate) / posRate

                if posRate == 0:
                    theory_pos = theory_neg
                
                if posRate > 0:
                    theories[disease]['positive'] = theory_pos
                else: theories[disease]['positive'] = prev_theory_pos
                theories[disease]['negative'] = theory_neg

            prev_disease = disease
            prev_theory_pos = theory_pos
    
        return theories
    

