
##
## By Elim Thompson (05/21/2024)
##
## This script encapsulates all properties/calculations of a hierarchical
## queue based on Rucha and Michelle works.
###########################################################################

################################
## Import packages
################################ 
import numpy

################################
## Define constants
################################

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

    @property
    def diseaseNames (self): return self._diseaseNames

    @property
    def groupNames (self): return self._groupNames

    @property
    def AINames (self): return self._AINames

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
        '''

        order, diseaseNames, groupNames, AINames = [], [], [], []

        for groupname, aGroup in diseaseDict.items():

            order.extend(aGroup['diseaseRanks'])
            diseaseNames.extend(aGroup['diseaseNames'])
            groupNames.extend(groupname)
            for diseaseName in aGroup['diseaseNames']:
                thisAI = [ainame for ainame, aiinfo in AIinfo.items() if aiinfo['targetDisease']==diseaseName]
                aiName = None if len (thisAI) == 0 else thisAI[0]
                AINames.append(aiName)
        
        order = numpy.array (order) - 1
        diseaseNames = numpy.array (diseaseNames)
        groupNames = numpy.array (groupNames)
        AINames = numpy.array (AINames)

        self._diseaseNames = diseaseNames[order]       
        self._groupNames = groupNames[order]
        self._AINames = AINames[order]