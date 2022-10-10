#goal: initialization package
#author: Julian van Erk
from FedDifPrivModels.FedDifPrivPGM import FedDifPrivPGM
from FedDifPrivModels.FedPGMDataset import FedPGMDataset
from FedDifPrivModels.Utils import get_lipschitz_constant
from FedDifPrivModels.FactoredInferenceAdaption import FactoredInference
from networkx.algorithms import tree
from scipy.special import logsumexp
from disjoint_set import DisjointSet
import networkx as nx
from scipy.stats import norm
import itertools as iter
import numpy as np


class FedAllInOnePGM(FedDifPrivPGM):
    """
    Subclass of FedDifPrivPGM for PGM selection methods where all measurements after initialization are measured in a all-in-one fashion.
    """

    def __init__(self, dMu, iMaxDegree = 2, rng = None, iSeed = None, sGraphType = "tree", sScoreType = "standard", 
                 bVerbose = False, sOptMeth = 'IG', dMaxJTsize = None, iIterInit = 2500, iIterMeasure = 50000,
                 lQuadBudgetProp =  [0.1,0.1,0.8], dConv = 10**-1, bCombineLevels = False, dQuadPropCorrBinning = 0.4,
                 iMaxRoundsNumerical = 4, bLocalModel = False):
        """
        Goal: Constructor of FedDifPrivPGM object, where the measure step is all-in-one.
        Input:
            - dMu                   double, privacy budget used
            - iMaxDegree            integer, max clique degree 
            - rng                   random number generator
            - iSeed                 integer, seed for random number generator (only applicable if rng not supplied)
            - sGraphType            string, Graph type, i.e. tree or maxJTsize
            - sScoreType            string, Score type, i.e. standard, adjusted 
            - bVerbose              boolean, verbose (print intermediate output)
            - sOptMeth              string, optimization method, i.e. IG, RDA or MD
            - dMaxJTSize            double, maximum junction tree size (only applicable if sGraphType is maxJTsize)
            - iIterInit             integer, number of initial iterations
            - lQuadBudgetProp       list, size 3 with quadratic budget propotions for the initial, select and measure step respectively
            - dConv                 double, convergence parameter for termination condition (if L2 improvement after 100 iters < dConv, terminate)
            - bCombineLevels         boolean, use combineLevels approach
            - iMaxRoundsNumerical   integer, number of binary split to generate bins for numerical variables
            - bLocalModel           boolean, use a local model for determining the scores
        Set:
            - self
        """
        #set general attributes 
        super().__init__(dMu, iMaxDegree, rng = rng, iSeed = iSeed, sGraphType=sGraphType, sScoreType=sScoreType, 
                         bVerbose=bVerbose, sOptMeth=sOptMeth, dMaxJTsize= dMaxJTsize, iIterInit=iIterInit, dConv=dConv,
                         bCombineLevels = bCombineLevels, sSelectionMethod="all-in-one", dQuadPropCorrBinning = dQuadPropCorrBinning,
                         iMaxRoundsNumerical = iMaxRoundsNumerical)

        #set budget proportions
        self.lQuadBudgetProp = lQuadBudgetProp
        if lQuadBudgetProp is None:
            if bCombineLevels == False:
                self.lQuadBudgetProp = [0.1,0.1,0.8]
            else:
                self.lQuadBudgetProp = [0.25,0.1,0.65]

        #set all-in-one specific attributes
        vMuSplit = np.sqrt(dMu**2 * np.array(self.lQuadBudgetProp))
        self.dMuInit = vMuSplit[0]
        self.dMuSelection = vMuSplit[1]
        self.dMuMeasure = vMuSplit[2]
        self.iIterMeasure = iIterMeasure
        self.bLocalModel = bLocalModel


    def get_AllInOne_selection_all_clients(self, fedPGMDataset, rng):
        """
        Goal: Select step: get DP MST for each clients and combine these by applying another MST algorithm, to create a set of marginals to measure
        Input:
        - self              a FedAllInOnePGM object
        - fedPGMDataset     A FedPGMDataset object

        Output:
        - lCliques          final set of marginals to measure

        """

        if self.sGraphType == "tree":
            #initialize list of graphs
            lClientGraphs = []

            #get indivual graphs
            if self.bLocalModel:
                for c in range(fedPGMDataset.iC):
                    lClientGraphs.append(self.get_AllInOne_selection_client(fedPGMDataset.ldf[c], 
                        self.lClientEngines[c].model, self.dMuSelection, self.dMuMeasure, fedPGMDataset.vNoisyFrac[c], 
                        rng, self.sGraphType, 'local'))
            else:
                for c in range(fedPGMDataset.iC):
                    lClientGraphs.append(self.get_AllInOne_selection_client(fedPGMDataset.ldf[c], 
                        self.engine.model, self.dMuSelection, self.dMuMeasure, fedPGMDataset.vNoisyFrac[c], 
                        rng, self.sGraphType, self.sScoreType))

            #initialize combined maximum spanning tree
            globalMST = nx.Graph()

            #get list of all edges in the graphs
            lAllEdges = []
            for c in range(fedPGMDataset.iC):
                lAllEdges.append(list(lClientGraphs[c].edges))

            #add edges to global graph
            for c in range(fedPGMDataset.iC):
                for i in range(len(lAllEdges[c])):
                    #get edge and client
                    edge = lAllEdges[c][i]

                    #check if edge is already in graph, else add edge
                    if globalMST.has_edge(*edge):
                        globalMST[edge[0]][edge[1]]['weight'] = globalMST[edge[0]][edge[1]]['weight'] -fedPGMDataset.vNoisyFrac[c]
                    else:
                        globalMST.add_edge(u_of_edge = edge[0], v_of_edge = edge[1], weight = -fedPGMDataset.vNoisyFrac[c])


            #apply kruskal
            lFinalEdges = list(tree.minimum_spanning_edges(globalMST, algorithm='kruskal', data = False))
        
        else:
            raise NotImplementedError("all-in-one + maxJTSize not implemented yet")

        #set list of lists of cliques for every client
        lCliques = lFinalEdges

        return lCliques
    
    @staticmethod
    def get_AllInOne_selection_client(dataset, currentModel, dMu, dMuNext, dFrac, rng, sGraphType, score):
        """
        Goal: get the a differentially private (DP) maximum spanning tree (MST) of attribute combinations to measure in the next step
        input:
            - dataset       dataset of this client
            - currentModel  current (global) model
            - dMu           privacy budget for selection
            - dMuNext       privacy budget for measurements in next step
            - dFrac         fraction of total observations in this local dataset
            - rng           random number generator
        output:
            - client MST    DP maximum spanning tree of this client
        
        """
        if sGraphType != "tree":
            raise NotImplementedError(f"Graph: {sGraphType} not implemented yet for all-in-one")

        #get number of selections
        iSelections = len(dataset.domain) - 1 

        #get weights and normalize
        vWeights = np.ones(iSelections) 
        vWeights = vWeights / np.linalg.norm(vWeights)

        #get mu for all selections
        vMu = dMu * vWeights

        #transform mu to epsilon
        vEps = np.log((1.0/norm.cdf(-0.5*vMu)) - 1.0)

        #get all attribute combinations
        lCombs = list(iter.combinations(dataset.domain.attrs, 2))

        #get scores
        diScores = dict()
        for comb in lCombs:
            diScores[comb] = FedDifPrivPGM.get_score(dataset, currentModel, comb, dMuNext, dFrac, sScoreType = score)

        #define graph
        clientMST = nx.Graph()
        clientMST.add_nodes_from(dataset.domain.attrs)

        #get set of disjoint 
        disj = DisjointSet()

        #apply "kruskal" 
        for i in range(iSelections):

            #get candidates and corresponding scores
            lCandidates = [edge for edge in lCombs if not disj.connected(*edge)]
            vScoresCandidates = np.array([0.5*vEps[i]*diScores[edge] for edge in lCandidates]) #dp by adding 0.5*eps

            #apply exponential mechanims
            vProbs = np.exp(vScoresCandidates - logsumexp(vScoresCandidates))
            iChoice = rng.choice(len(vProbs), p = vProbs)
            selectedCandidate = lCandidates[iChoice]

            #add choice to graph
            clientMST.add_edge(*selectedCandidate)
            disj.union(*selectedCandidate)

        return clientMST

    def fit(self, ldf, diCatUniqueValuesInfo, diMinMaxInfo, lDataTypes = None):
        """
        Goal: Estimate model with all-in-one selection approach
        Input:
        - self                  A FedAllInOnePGM object.                
        - ldf                   A list of pandas dataframes.
        - diCatUniqueValuesInfo Dictionary with values of categorical variables
        - diMinMaxInfo          Dictionary with minimum and maximum values for ordinal and numerical variables
        - lDatatypes            List of datatypes of the dataframes (determines way of discritization). Options: ["categorical","numerical","ordinal"].

        Set
        - engine                The Engine (and therefore the model) of the FedAllInOnePGM object.       
        output:
        - modelFinal            Final model.
        - fedPGMDataset         A fedPGMDataset with all measuremenets.
        """
        
        #################################
        # Initialization
        #################################

        if self.bVerbose:
            print("start discritization and initial measurement")


        #check which initialization to use
        if self.bCombineLevels:
            #use a part of the privacy budget for learning which levels to combine
            fedPGMDataset = FedPGMDataset(ldf, diMinMaxInfo=diMinMaxInfo, diCatUniqueValuesInfo= diCatUniqueValuesInfo, rng = self.rng, lDataTypes=lDataTypes, dMu = self.dMuInit, dMuMeasure=self.dMuMeasure, dQuadPropCorrBinning = self.dQuadPropCorrBinning, iMaxRoundsNumerical=self.iMaxRoundsNumerical)
        else:
            #standard splitting
            fedPGMDataset = FedPGMDataset(ldf,diMinMaxInfo=diMinMaxInfo, diCatUniqueValuesInfo= diCatUniqueValuesInfo, rng = self.rng, lDataTypes=lDataTypes, dMu = 0.0, dQuadPropCorrBinning = self.dQuadPropCorrBinning, iMaxRoundsNumerical=self.iMaxRoundsNumerical)

            #use dMuInit instead for measurements
            lOneWayCliques = [(col,) for col in fedPGMDataset.domain] 
            fedPGMDataset.get_measurements_all_clients(lOneWayCliques, rng = self.rng, dMu = self.dMuInit)
       

        if self.bVerbose:
            print("start initial estimation")

        #set engine
        self.engine = FactoredInference(fedPGMDataset.domain, warm_start=True)
        self.engine.iters = self.iIterInit

        #estimation of initialization
        self.engine._setup(fedPGMDataset.lMeasurements, total = None)
        if (self.sOptMeth == "RDA" or self.sOptMeth == "IG"):
            dLipschitz = get_lipschitz_constant(self.engine, fedPGMDataset.lMeasurements)
            self.engine.estimate(fedPGMDataset.lMeasurements, engine=self.sOptMeth, options = { "lipschitz" : dLipschitz, "dConvergence" : self.dConv})
        else:
            self.engine.estimate(fedPGMDataset.lMeasurements, engine=self.sOptMeth)

        #fit local independent models 
        if self.bLocalModel:
            self.lClientEngines = [FactoredInference(fedPGMDataset.domain, warm_start=True) for c in range(fedPGMDataset.iC)]
            
            for c in range(fedPGMDataset.iC):
                #set number of iterations
                self.lClientEngines[c].iters = self.iIterInit

                #first, set client measurements
                lMeasurements = fedPGMDataset.llClientMeasurements[c][0]
                
                #now optimize to get a local model with the same SNR
                if (self.sOptMeth == "RDA" or self.sOptMeth == "IG"):
                    dLipschitz = get_lipschitz_constant(self.engine, lMeasurements)
                    self.lClientEngines[c].estimate(lMeasurements, engine=self.sOptMeth, options = { "lipschitz" : dLipschitz, "dConvergence" : self.dConv})
                else:
                    self.lClientEngines[c].estimate(lMeasurements, engine=self.sOptMeth)


        #################################
        # Selection + measure
        #################################

        if self.bVerbose:
            print("start second selection step")

        #get selection of 2-way marginals
        lSecondStageCliques = self.get_AllInOne_selection_all_clients(fedPGMDataset, rng = self.rng)

        #measure 2-way marginals
        fedPGMDataset.get_measurements_all_clients(lSecondStageCliques, self.rng, dMu = self.dMuMeasure)

        #################################
        # final estimation
        #################################

        if self.bVerbose:
            print("start final estimation")

        #set iters
        self.engine.iters = self.iIterMeasure
        self.engine.log = self.bVerbose
    
        if (self.sOptMeth == "RDA" or self.sOptMeth == "IG"): 
            #get lipschitz
            dLipschitz = get_lipschitz_constant(self.engine, fedPGMDataset.lMeasurements)
            self.engine.estimate(fedPGMDataset.lMeasurements, engine=self.sOptMeth, options = { "lipschitz" : dLipschitz, "dConvergence" : self.dConv})
        else:
            self.engine.estimate(fedPGMDataset.lMeasurements, engine=self.sOptMeth)

        return self.engine.model, fedPGMDataset