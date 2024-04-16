
##
## By Elim Thompson (11/27/2020)
##
## This script encapsulates a disease class.
###########################################################################

################################
## Import packages
################################ 
import numpy, pandas, scipy
from copy import deepcopy

import AI
################################
## Define constants
################################

################################
## Define disease tree structure
################################
class disease (object):
    
    def __init__ (self):
        
        self._diseaseName = None
        self._diseaseProb = None
        self._groupBelong = None
        self._meanReadTime = None

    def __str__ (self):
        summary = '| * {0}:\n'.format (self.diseaseName)
        summary += '|    - Belong to: {0}\n'.format (self.groupBelong)
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
    def groupName (self):
        return self._groupName
    @groupName.setter
    def groupName (self, groupName):
        self._groupName = groupName
    
    @property
    def groupProb (self):
        return self._groupProb
    @groupProb.setter
    def groupProb (self, groupProb):
        self._groupProb = round (groupProb, 6)    
    
    @property
    def AIs (self):
        return self._AIs
    
    @property
    def diseases (self):
        return self._diseases

    def check_diseaseProbs (self):
        
        probs = [aDisease.diseaseProb for aDisease in self._diseases]   
        if not round (sum (probs), 5) == 1:
            raise IOError ('Disease probability in {0} do not add up.'.format (self.groupName))        
    
    def add_nondisease (self, nonDiseaseMeanReadTime):
        
        aDisease = disease()
        aDisease.diseaseName = 'non-diseased'
        aDisease.diseaseProb = 1 - sum ([aDisease.diseaseProb for aDisease in self._diseases]) 
        aDisease.meanReadTime = nonDiseaseMeanReadTime
        aDisease.groupBelong = self.groupName
        
        self._diseases.append (aDisease)
    
    def add_disease (self, diseaseName, diseaseProb, meanReadTime):
        
        aDisease = disease()
        aDisease.diseaseName = diseaseName
        aDisease.diseaseProb = diseaseProb
        aDisease.meanReadTime = meanReadTime
        aDisease.groupBelong = self.groupName
        
        self._diseases.append (aDisease)

    def add_AI (self, anAI):
        if isinstance (anAI, list):
            self._AIs.extend (anAI)
        else:
            self._AIs.append (anAI)

class diseaseTree (object):
    
    def __init__ (self):       
        self._diseaseGroups = []

    def __str__ (self):
        summary  = '===============================================\n'
        summary += '| Disease Tree\n'.format ()
        for aGroup in self._diseaseGroups:
            summary += aGroup.__str__() 
        summary += '==============================================='
        return summary        
        
    @property
    def diseaseGroups (self):
        return self._diseaseGroups

    def get_groupNames (self):
        return [aGroup.groupName for aGroup in self.diseaseGroups]
        
    def check_groupProbs (self):
        ## Check group and disease probabilities
        groupProbs = []
        for aGroup in self.diseaseGroups:
            aGroup.check_diseaseProbs()
            groupProbs.append (aGroup.groupProb)
    
        if not round (sum (groupProbs), 5) == 1:
            raise IOError ('Disease group probability do not add up.')    
    
    def add_aDiseaseGroup (self, diseaseGroupName, diseaseGroupProb,
                           diseaseNames, diseaseProbs, meanReadTimes, AIs=[]):
        
        aGroup = diseaseGroup()
        aGroup.groupName = diseaseGroupName
        aGroup.groupProb = diseaseGroupProb
        
        ## Add the diseases first
        diseaseMeanReadTime = [meanReadTime for diseaseName, meanReadTime in meanReadTimes.items()
                               if not diseaseName=='non-diseased']
        for name, prob, rTime in zip (diseaseNames, diseaseProbs, diseaseMeanReadTime):
            aGroup.add_disease (name, prob, rTime)
        ## At the end, add non-disease group
        aGroup.add_nondisease (meanReadTimes['non-diseased'])
        
        ## Add AIs that belong to this group. This AI will read everything in this group.
        aGroup.add_AI (AIs)
        
        ## Store this group into this instance
        self._diseaseGroups.append (aGroup)
        
    def build_diseaseTree (self, diseaseDict, meanServiceTimes, AIs):
        
        for groupname, aGroup in diseaseDict.items():
            ## Identify all the AIs in this group
            AIsInGroup = [anAI for subID, anAI in AIs.items() if anAI.groupName==groupname]
            ## Add a new disease group
            self.add_aDiseaseGroup (groupname, aGroup['groupProb'],
                                    aGroup['diseaseNames'],
                                    aGroup['diseaseProbs'],
                                    meanServiceTimes[groupname],
                                    AIs=AIsInGroup)

        self.check_groupProbs()
    
    def get_diseased_prevalence (self):
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
        return 1 - self.get_diseased_prevalence()
        
        