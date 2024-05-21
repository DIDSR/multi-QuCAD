
##
## By Elim Thompson (11/27/2020)
##
## This script defines classes for each disease, each group, and a tree of
## diseases. 
###########################################################################

################################
## Import packages
################################ 
import numpy

################################
## Define constants
################################

################################
## Define disease tree structure
################################
class disease (object):
    
    ''' An object to encapsulate properties of a disease condition
        * its name
        * which group does it belong to
        * disease prevalence within the group
        * rank of disease in a hierarchical queue
        * radiologist mean reading time when reading a diseased case
    '''

    def __init__ (self):
        
        self._diseaseName = None
        self._diseaseProb = None
        self._diseaseRank = None
        self._groupBelong = None
        self._meanReadTime = None

    def __str__ (self):
        summary =  '| * {0}:\n'.format (self.diseaseName)
        summary += '|    - Belong to: {0}\n'.format (self.groupBelong)
        summary += '|    - Rank: {0}\n'.format (self.diseaseRank)
        summary += '|    - Probability: {0}\n'.format (self.diseaseProb)        
        summary += '|    - Mean reading time: {0} min\n'.format (self.meanReadTime)
        return summary    

    @property
    def diseaseName (self):
        return self._diseaseName
    @diseaseName.setter
    def diseaseName (self, diseaseName):
        self._diseaseName = diseaseName
    
    @property
    def diseaseRank (self):
        return self._diseaseRank
    @diseaseRank.setter
    def diseaseRank (self, diseaseRank):
        self._diseaseRank = diseaseRank

    @property
    def diseaseProb (self):
        return self._diseaseProb
    @diseaseProb.setter
    def diseaseProb (self, diseaseProb):
        self._diseaseProb = round (diseaseProb, 6)

    @property
    def groupBelong (self):
        return self._groupBelong
    @groupBelong.setter
    def groupBelong (self, groupBelong):
        self._groupBelong = groupBelong

    @property
    def meanReadTime (self):
        return self._meanReadTime
    @meanReadTime.setter
    def meanReadTime (self, meanReadTime):
        self._meanReadTime = meanReadTime
    
class diseaseGroup (object):

    ''' An object to encapsulate properties of all disease conditions
        that belong to the same group
        * the group name
        * fraction of patients (w.r.t. all patients) that belong to this group
        * an array of all disease conditions within the group
        * an array of AIs that analyze images within this group
    '''

    def __init__ (self):
        
        self._groupName = None
        self._groupProb = None
        self._diseases = []
        self._AIs = []
    
    def __str__ (self):
        summary  = '+-------------------------------------\n'
        summary += '| Disease group - {0} ({1}) \n'.format (self.groupName, self.groupProb)
        for aDisease in self._diseases:
            summary += aDisease.__str__() 
        return summary
    
    @property
    def groupName (self): return self._groupName
    @groupName.setter
    def groupName (self, groupName): self._groupName = groupName
    
    @property
    def groupProb (self): return self._groupProb
    @groupProb.setter
    def groupProb (self, groupProb):
        self._groupProb = round (groupProb, 6)    
    
    @property
    def AIs (self): return self._AIs
    
    @property
    def diseases (self): return self._diseases

    def check_diseaseProbs (self):

        ''' Function to check if all disease prob (including non-diseased)
            adds up to 1.
        '''


        probs = [aDisease.diseaseProb for aDisease in self._diseases]   
        if not round (sum (probs), 5) == 1:
            raise IOError ('Disease probability in {0} do not add up.'.format (self.groupName))        
    
    def add_nondisease (self, nonDiseaseMeanReadTime):

        ''' Public function to add a non-disease into the array. Non-diseased
            cases are considered a "disease" object for coding purposes. Its
            "disease" name is always non-diseased, and its rank is always -1.
            Its disease prevalence is 1 minus the diseaseProb of all diseases
            in this group. Therefore, this function should always be called at
            the end after defining all disease conditions in the group.

            input
            -----
            nonDiseaseMeanReadTime (float): radiologist mean reading time in minutes
                                            to read a non-diseased case in this group.
        '''

        aDisease = disease()
        aDisease.diseaseName = 'non-diseased'
        aDisease.diseaseRank = -1
        aDisease.diseaseProb = 1 - sum ([aDisease.diseaseProb for aDisease in self._diseases]) 
        aDisease.meanReadTime = nonDiseaseMeanReadTime
        aDisease.groupBelong = self.groupName
        
        self._diseases.append (aDisease)
    
    def add_disease (self, diseaseName, diseaseRank, diseaseProb, meanReadTime):
        
        ''' Public function to add a disease condition into the array. Each
            disease condition is defined by its name, disease prevalence
            within the group, and radiologist reading time.

            input
            -----
            diseaseName (str): name of the disease condition
            diseaseRank (int): rank of disease in a hierarchical queue
            diseaseProb (float): disease prevalence within the group
            meanReadTime (float): radiologist mean reading time in minutes
                                  to read a diseased case.
        '''

        aDisease = disease()
        aDisease.diseaseName = diseaseName
        aDisease.diseaseRank = diseaseRank
        aDisease.diseaseProb = diseaseProb
        aDisease.meanReadTime = meanReadTime
        aDisease.groupBelong = self.groupName
        
        self._diseases.append (aDisease)

    def add_AI (self, anAI):

        ''' Public function to add an AI into the array i.e. these AI(s)
            analyze all images within this group.

            input
            -----
            anAI: list of AI object or one AI object that will be
                  appended to the AI list for this group.
        '''

        if isinstance (anAI, list):
            self._AIs.extend (anAI)
        else:
            self._AIs.append (anAI)

class diseaseTree (object):
    
    ''' A diseaseTree object that encapsulates all the groups of diseases
        defined by the users.
    '''

    def __init__ (self):       
        self._diseaseGroups = []
        self._diseaseRanked = None

    def __str__ (self):
        summary  = '===============================================\n'
        summary += '| Disease Tree\n'.format ()
        for aGroup in self._diseaseGroups:
            summary += aGroup.__str__() 
        summary += '==============================================='
        return summary        
        
    @property
    def diseaseGroups (self): return self._diseaseGroups

    @property
    def diseaseRanked (self): return self._diseaseRanked

    def get_groupNames (self):

        ''' Quick public function to get the names of all groups
        '''

        return [aGroup.groupName for aGroup in self.diseaseGroups]
        
    def check_groupProbs (self):

        ''' Check if all group probability adds up to 1.
        '''

        ## Check group and disease probabilities
        groupProbs = []
        for aGroup in self.diseaseGroups:
            aGroup.check_diseaseProbs()
            groupProbs.append (aGroup.groupProb)
    
        if not round (sum (groupProbs), 5) == 1:
            raise IOError ('Disease group probability do not add up.')    
    
    def add_aDiseaseGroup (self, diseaseGroupName, diseaseGroupProb,
                           diseaseNames, diseaseRanks, diseaseProbs,
                           meanReadTimes, AIs=[]):

        ''' Public function to add a disease group that may have multiple
            disease conditions. Note that diseaseNames, diseaseProbs, and
            meanReadTimes are in the same order as the cooresponding disease
            conditions. For example,
            diseaseNames  = [ 'A', 'B',  'C']
            diseaseRanks  = [   1,   2,    3]
            diseaseProbs  = [0.05, 0.1, 0.07]
            meanReadTimes = [  10, 7.5, 13.8] 
            i.e. Disease A has a probability of 0.05 within this group and 
                 mean reading time for A is 10 minutes.

            inputs
            ------
            diseaseGroupName (str): group name
            diseaseGroupProb (float): fraction that a patient belongs to this group
            diseaseNames (array str): array of disease names
            diseaseRanks (array int): ranking of disease conditions
            diseaseProbs (array float): array of disease prevalence within the group
            meanReadTimes (array float): radiologists' reading time in minutes 
            AIs (array AI): array of the AI objects that read every image in this group
        '''

        aGroup = diseaseGroup()
        aGroup.groupName = diseaseGroupName
        aGroup.groupProb = diseaseGroupProb
        
        ## Add the diseases first
        diseaseMeanReadTime = [meanReadTime for diseaseName, meanReadTime in meanReadTimes.items()
                               if not diseaseName=='non-diseased']
        for name, rank, prob, rTime in zip (diseaseNames, diseaseRanks, diseaseProbs, diseaseMeanReadTime):
            aGroup.add_disease (name, rank, prob, rTime)
        ## At the end, add non-disease group
        aGroup.add_nondisease (meanReadTimes['non-diseased'])
        
        ## Add AIs that belong to this group. This AI will read everything in this group.
        aGroup.add_AI (AIs)
        
        ## Store this group into this instance
        self._diseaseGroups.append (aGroup)
        
    def build_diseaseTree (self, diseaseDict, meanServiceTimes, AIs):
        
        ''' Public function to build the entire disease tree. 

            inputs
            ------
            diseaseDict (dict): group information from config file
                e.g. {'GroupCT':{'groupProb':0.4, 'diseaseNames':['A'],
                                 'diseaseRanks':[1], 'diseaseProbs':[0.3]},
                      'GroupUS':{'groupProb':0.6, 'diseaseNames':['F'],
                                 'diseaseRanks':[2], 'diseaseProbs':[0.6]}}
            meanServiceTimes (dict): radiologists' service time by groups and diseases
                                     e.g. {'GroupCT':{'A':10, 'non-diseased':7},
                                           'GroupUS':{'F':6, 'non-diseased':7}}
            AIs (dict): directary of all AIs involved in the queue
                        e.g. {AIname: an AI object}
        '''

        for groupname, aGroup in diseaseDict.items():
            ## Identify all the AIs in this group
            AIsInGroup = [anAI for _, anAI in AIs.items() if anAI.groupName==groupname]
            ## Add a new disease group
            self.add_aDiseaseGroup (groupname, aGroup['groupProb'],
                                    aGroup['diseaseNames'],
                                    aGroup['diseaseRanks'],
                                    aGroup['diseaseProbs'],
                                    meanServiceTimes[groupname],
                                    AIs=AIsInGroup)

        order = numpy.array ([agroup['diseaseRanks'][0] for _, agroup in diseaseDict.items()])
        diseaseNames = numpy.array ([agroup['diseaseNames'][0] for _, agroup in diseaseDict.items()])
        self._diseaseRanked = diseaseNames[order-1]

        self.check_groupProbs()
    
    def get_diseased_prevalence (self):

        ''' Quick public function to get the overall disease prevalence
            (considering all disease conditions) with respect to all
            patients. 
        '''

        ## Diseased prevalence = # diseased / # total
        ## Essentially, it is sum (groupProb * diseaseProb)
        prevalence = 0
        
        for aGroup in self.diseaseGroups:
            groupProb = aGroup.groupProb
            for aDisease in aGroup.diseases:
                if aDisease.diseaseName=='non-diseased': continue
                diseaseProb = aDisease.diseaseProb
                prevalence += groupProb * diseaseProb

        return prevalence
    
    def get_nondiseased_prevalence (self):

        ''' Quick public function to get the non-diseased prevalence
        '''

        return 1 - self.get_diseased_prevalence()
