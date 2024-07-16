
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
        self._hierDict = {ainame : order + start_priority
                          for ainame, order in zip (self._AINames, order)}
        
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

    def _update_disease_names (self, newParams, groups_wAI):
        
        ''' Private function to updates disease and group names for consistency in
            names in the theoretical calculations. Note only one AI is expected.
            This function replaces the group name by 'GroupEQ' and disease name by
            'Z'. This is called when only one disease group is present in the list
            of hi_AIs.

            inputs
            ------
            newParams (dict): original params dict that will be updated
            groups_wAI (list): a list of group names that have AI involved.
                               This should just be 1 element in this case.
            
            output
            ------
            newParams (dict): params dict that is updated
        '''

        vendor = list(newParams['AIinfo'].keys())[0] 
        gp = newParams['AIinfo'][vendor]['groupName']
        dis = newParams['AIinfo'][vendor]['targetDisease']

        # 1. New AIinfo = the 1 AI
        new_AIinfo = {'Vendor0': {'groupName': 'GroupEQ', 'targetDisease': 'Z',
                                  'TPFThresh': newParams['AIinfo'][vendor]['TPFThresh'],
                                  'FPFThresh': newParams['AIinfo'][vendor]['FPFThresh'],
                                  'rocFile': newParams['AIinfo'][vendor]['rocFile']}}

        # 2. Update disease group names
        #    This group may have multiple disease conditions. But the diseaseProb
        #    should be the target disease of this 1 AI
        disInd = list (newParams['diseaseGroups'][gp]['diseaseNames']).index (dis)
        diseaseProb = newParams['diseaseGroups'][gp]['diseaseProbs'][disInd]
        new_diseaseGroup = {'GroupEQ': {'diseaseNames': ['Z'], 'diseaseRanks':[1],
                                        'diseaseProbs': [diseaseProb], 
                                        'groupProb': newParams['diseaseGroups'][gp]['groupProb']}}

        # 3. Update meanServiceTime
        new_meanServiceTimes = {}
        new_meanServiceTimes['interrupting'] = newParams['meanServiceTimes']['interrupting']
        for groupname in newParams['diseaseGroups'].keys():
            if not groupname in newParams['meanServiceTimes']: continue
            new_meanServiceTimes[groupname] = newParams['meanServiceTimes'][groupname] 

        TDZ, TNDZ = self._calculate_equivalent_readtime (newParams['meanServiceTimes'],
                                                         newParams['diseaseGroups'],
                                                         newParams['AIinfo'], groups_wAI)
        new_meanServiceTimes['GroupEQ'] = {'Z': TDZ, 'non-diseased': TNDZ}

        # Actually update newParams
        del newParams['meanServiceTimes'][gp]
        newParams['meanServiceTimes'].update(new_meanServiceTimes)
        newParams['AIinfo'] = new_AIinfo
        del newParams['diseaseGroups'][gp]
        newParams['diseaseGroups'].update(new_diseaseGroup)

        return newParams
    
    def _get_info_for_equivalent_SeSp (self, newParams):

        ## Create an equivalent AI, group, and the corresponding diseased probabilities.
        all_se, all_sp, all_pop_prevalence, all_groupProbs, all_groupNames = [], [], [], [], []
        for vendor in newParams['AIinfo'].keys():
            all_se.append(newParams['AIinfo'][vendor]['TPFThresh'])
            all_sp.append(1 - newParams['AIinfo'][vendor]['FPFThresh'])
            gp = newParams['AIinfo'][vendor]['groupName']
            dis = newParams['AIinfo'][vendor]['targetDisease']
            all_groupProbs.append (newParams['diseaseGroups'][newParams['AIinfo'][vendor]['groupName']]['groupProb'])
            all_groupNames.append (gp)
            # Find disease index in the list of all diseases in a gp.
            dis_idx = numpy.where (numpy.array (newParams['diseaseGroups'][gp]['diseaseNames'])==dis)[0][0]
            # Compute disease prevalence in population
            pop_prev = newParams['diseaseGroups'][gp]['diseaseProbs'][dis_idx] \
                        * newParams['diseaseGroups'][gp]['groupProb'] 
            all_pop_prevalence.append(pop_prev)
        all_se = numpy.array(all_se)
        all_sp = numpy.array(all_sp)
        all_pop_prevalence = numpy.array(all_pop_prevalence) 
        all_groupProbs = numpy.array (all_groupProbs)
        all_groupNames = numpy.array (all_groupNames)        

        return all_se, all_sp, all_pop_prevalence, all_groupProbs, all_groupNames

    def _update_newParams (self, newParams, groups_wAI, groups_noAI): 
        
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
        
        ## Compute equivalent group prob after combining all groups with AI
        ## i.e. groupProb for GroupEQ
        unique_gp_probs = [gpinfo['groupProb'] for gpname, gpinfo in newParams['diseaseGroups'].items()
                           if gpname in groups_wAI]
        gpEQ_prob = sum (unique_gp_probs) 
        
        ## Create an equivalent AI, group, and the corresponding diseased probabilities.
        all_se, all_sp, all_pop_prevalence, all_groupProbs, _ = self._get_info_for_equivalent_SeSp (newParams)

        ## Create an equivalent AI Se/Sp. Assuming ... 
        ##  1. Diseases seen by AIs are uncorrelated
        ##  2. Only 1 AI in each group
        EQSe, EQSp = self._calculate_equivalent_Se_Sp(all_se, all_sp, all_pop_prevalence, all_groupProbs)
        ## Disease probability for disease 'Z' in this GroupEQ
        disEQ_prob = numpy.sum(all_pop_prevalence) / gpEQ_prob 
        new_diseaseGroups = {'GroupEQ': {'diseaseNames': ['Z'], 'diseaseRanks':[1],
                                         'diseaseProbs': [disEQ_prob], 'groupProb': gpEQ_prob}}

        ## If there are any groups without AI, add the disease group info.
        if groups_noAI:
            for gp in groups_noAI:
                for groupname, agroup in newParams['diseaseGroups'].items():
                    if not groupname == gp: continue
                    new_diseaseGroups[groupname] = agroup 
        
        ## Update the corresponding mean service times. First, add in the old ones.
        new_meanServiceTimes = {'interrupting': newParams['meanServiceTimes']['interrupting']}
        for groupname in newParams['diseaseGroups'].keys():
            if not groupname in newParams['meanServiceTimes']: continue
            new_meanServiceTimes[groupname] = newParams['meanServiceTimes'][groupname] 
        ## Now add in the effective read time for GroupEQ with diseased and nondiseased         
        TDZ, TNDZ = self._calculate_equivalent_readtime (newParams['meanServiceTimes'],
                                                         newParams['diseaseGroups'],
                                                         newParams['AIinfo'], groups_wAI)
        new_meanServiceTimes['GroupEQ'] = {'Z': TDZ, 'non-diseased': TNDZ}

        ## Actually update the dictionary
        newParams['meanServiceTimes'] = new_meanServiceTimes
        newParams['diseaseGroups'] = new_diseaseGroups
        newParams['AIinfo'] = {'Vendor0': {'groupName': 'GroupEQ', 'targetDisease': 'Z',
                                           'TPFThresh': EQSe, 'FPFThresh': 1-EQSp, 'rocFile': None}}    
        
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

        ## Make a new copy to avoid contemination
        params_out = deepcopy (params) 

        ## Update AIinfo if given a list of AI names
        if keep_ai is not 'all':
            params_out['AIinfo'] = {ainame: params_out['AIinfo'][ainame] for ainame in keep_ai}

        ## For theory, this function may be called to create equivalent groups, either by
        ## combining AIs or as a dummy case with a single AI
        groups_wAI = [aiinfo['groupName'] for _, aiinfo in params_out['AIinfo'].items()]
        if len (params_out['AIinfo']) > 1: 
            groups_noAI = list(set(params_out['diseaseGroups'].keys()) - set(groups_wAI)) 
            params_out = self._update_newParams (params_out, groups_wAI, groups_noAI)
        else:
            params_out = self._update_disease_names(params_out, groups_wAI)
            
        # ## Update additional params
        params_out, _, _ = inputHandler.add_params (params_out, include_theory=False)

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
        
        idx = list (self.diseaseNamesWithAIs).index(diseaseDivide)
        hi_AIs = self.AINamesWithAIs[:idx]
        lo_AIs = self.AINamesWithAIs[:idx+1]
        return hi_AIs, lo_AIs

    def _get_positive_rate (self, params, groupname=None, AIname=None, diseasename=None):
        
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
        
        if 'GroupEQ' in params['diseaseGroups']:
            groupProb = params['diseaseGroups']['GroupEQ']['groupProb']
            diseaseProb = params['diseaseGroups']['GroupEQ']['diseaseProbs'][0]
            SeThresh = params['SeThresh']
            SpThresh = params['SpThresh']
        else:
            groupProb = params['diseaseGroups'][groupname]['groupProb']
            disIdx = params['diseaseGroups'][groupname]['diseaseNames'].index (diseasename)
            diseaseProb = params['diseaseGroups'][groupname]['diseaseProbs'][disIdx]
            SeThresh = params['SeThreshs'][AIname]
            SpThresh = params['SpThreshs'][AIname]

        tp_prev = diseaseProb * groupProb * SeThresh
        fp_prev = (1 - diseaseProb) * groupProb * (1 - SpThresh)
        return tp_prev + fp_prev

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

    def get_posNegWeights (self, params, disease, group):

        ''' Function to calculate weigths to convert positive/negative wait time to diseased
            and nondiseased. The mean wait-time of a disease of interests depends on 3 scenarios:
            I. the disease of interests has an AI to identify it
                (may have other AIs in the same group)
            II. the disease of interests does not have an AI to identify it, but there are other
                AIs in the same group i.e. any flagged cases are FP cases of other AIs.
            III. the disease of interests belong to a group that does not have any AIs.
            
            e.g. GroupCTA has 5 diseases (A, B, C, D, E) and 3 AIs for A, C, E. The wait-time
            I. For A is from both AI-A Se, AI-C Sp, and AI-E Sp.
                            N_TP_AIA + N_FP_A_AIC + N_FP_A_AID          N_FN_AIA + N_TN_A_AIC + N_TN_A_AID
                W_A = W+ x ------------------------------------ + W- x ------------------------------------ 
                                            N_A                                         N_A
                
                1)  N_TP_AIA              
                ---------- = AI-A-Se   
                    N_A                 

                2) N_FP_A_AIC     N_FP_A_AIC     N_FP_AIC     N_notC     N_G       pi_A                                     1
                ------------ = ------------ x ---------- x -------- x ----- = ---------- x (1 - AI-C-Sp) x (1 - pi_C) x ------ = 1 - AI-C-Sp
                    N_A         N_FP_AIC       N_notC       N_G       N_A     1 - pi_C                                  pi_A

                3)  N_FP_AID                    4)  N_FN_AIA
                ---------- = 1 - AI-D-Sp        ---------- = 1 - AI-A-Se
                    N_A                              N_A
                
                5)  N_TN_AIC                    6)  N_TN_AID
                ---------- = AI-C-Sp            ---------- = AI-D-Sp
                    N_A                              N_A  
                
                i.e. for the AI-trained for this disease A, use Se and 1-Se. For other AIs in this same group,
                    use (1-Sp) x (1-pi) / piA and Sp x (1-pi) / piA
            II. For diseased cases with no AIs in a group that have AIs, some cases would be
                accidentally prioritized. This calculation is the same as above except that 1)
                and 4) are replaced the same Sp-version of AI-A (like 2, 3, 5, and 6)
            III. For diseased cases with no AIs, the mean wait-time is simply the AI- mean wait-time.

        '''

        # Is there an AI reviewing this group?
        AIsInGroup = [ainame for ainame, aiinfo in params['AIinfo'].items() if aiinfo['groupName']==group]

        if disease in self.diseaseNamesWithAIs:
            # Scenario I.
            ainame_dis = [ainame for ainame, aiinfo in params['AIinfo'].items() if aiinfo['targetDisease']==disease][0]
            posWeight, negWeight = 0, 0
            for ainame in AIsInGroup:
                ## For the AI targeted for this disease, Se and 1- Se
                if ainame == ainame_dis:
                    posWeight += params['AIinfo'][ainame_dis]['TPFThresh']
                    negWeight += 1 - params['AIinfo'][ainame_dis]['TPFThresh']
                    continue
                ## For other AIs in the group, (1-Sp) and Sp 
                Sp = 1 - params['AIinfo'][ainame]['FPFThresh']
                posWeight += 1 - Sp
                negWeight += Sp 

            return posWeight, negWeight
        
        if len (AIsInGroup) > 0:
            # Scenario II.
            posWeight, negWeight = 0, 0
            for ainame in AIsInGroup:
                ## (1-Sp) and Sp 
                Sp = 1 - params['AIinfo'][ainame]['FPFThresh']
                posWeight += 1 - Sp 
                negWeight += Sp 

            return posWeight, negWeight
        
        else:
            ## Scenario III
            return 0, 1       

    def predict_mean_wait_time (self, params):

        ## For the lowst lowest priority class without an AI. This is the wait-time
        ## for all AI-negative patients and patients that has no AIs to review.
        _, theory_neg = self._predict (params, keep_ai='all', pclass='negative')

        ## Store the predicted wait-time for each disease/priority class
        theories = {}

        ## Now, loop through each disease condition to get the theorectical predictions
        for disease, group in zip(self.diseaseNames, self.groupNames):

            theories[disease] = {}

            if not disease in self.diseaseNamesWithAIs:
                theories[disease]['negative'] = theory_neg
                theories[disease]['diseased'] = theory_neg
                continue

            AIname = [ainame for ainame, aiinfo in params['AIinfo'].items()
                      if aiinfo['targetDisease']==disease][0]   

            HiAIs, LoAIs = self._define_high_low_classes (disease)
            ## If there is no HiAIs, i.e. working on the disease (with an AI) of
            ## the highest rank, just calculate the mean wait time of the low class.
            if len (HiAIs)==0:
                _, theory_pos = self._predict (params, keep_ai=LoAIs, pclass='positive')
                theories[disease]['positive'] = theory_pos
                theories[disease]['negative'] = theory_neg
                
            else:
                params_hi, theory_hi = self._predict (params, keep_ai=HiAIs, pclass='positive')
                params_lo, theory_lo = self._predict (params, keep_ai=LoAIs, pclass='positive')
            
                ## Get the rate of positive patients in each case: hi, lo, and chosen disease.
                loPosRate = self._get_positive_rate (params_lo)
                hiPosRate = self._get_positive_rate (params_hi)                 
                posRate = self._get_positive_rate (params, diseasename=disease, AIname=AIname, 
                                                    groupname=group)
                theory_pos = (theory_lo*loPosRate - theory_hi*hiPosRate) / posRate
                theories[disease]['positive'] = theory_pos
                theories[disease]['negative'] = theory_neg
            
            ## Combine the postive and negative wait times to obtain the diseased wait-time.
            posWeight, negWeight = self.get_posNegWeights (params, disease, group)
            theory_dis = theories[disease]['positive']*posWeight + \
                         theories[disease]['negative']*negWeight
            theories[disease]['diseased'] = theory_dis

        return theories
    