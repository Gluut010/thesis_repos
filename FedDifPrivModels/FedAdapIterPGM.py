#goal: initialization package
#author: Julian van Erk
from FedDifPrivModels.FedDifPrivPGM import FedDifPrivPGM
from FedDifPrivModels.FedPGMDataset import FedPGMDataset
from FedDifPrivModels.Utils import get_lipschitz_constant, powerset, downward_closure, get_counts
from FedDifPrivModels.FactoredInferenceAdaption import FactoredInference
from mbi import GraphicalModel
from mbi.junction_tree import JunctionTree
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from datetime import datetime
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 

class FedAdapIterPGM(FedDifPrivPGM):
    """
    A class that fits a PGM model using a adaptive and iterative selection procedure.
      
    """

    def __init__(self, dMu, iMaxDegree = 3, rng = None, iSeed = None, sGraphType = "maxJTsize", sScoreType = "standard", 
                 bVerbose = False, sOptMeth = 'IG', dMaxJTsize = None, iIterInit = 2500, dConv = 10**-1, bCombineLevels = False,
                 dAlpha = 8.0/9.0, dBeta = 0.5, iT = None, iIterPerMeasure = 2500, iIterFinal = 25000, dTheta = 6.0, lWorkload = None,
                 iMaxLevels = None, iMaxChecks = 500, dQuadInitBudgetProp = None, dQuadPropCorrBinning = (1.0/3.5),
                 iMaxRoundsNumerical = 4):
        """
        Goal: Contructor for FedDifPrivPGM model with iterative and adaptive measure method
        Input:
            - dMu                   double, privacy budget used
            - iMaxDegree            integer, max clique degree 
            - rng                   random number generator
            - iSeed                 integer, seed for random number generator (only applicable if rng not supplied)
            - sGraphType            string, Graph type, i.e. tree or maxJTsize
            - sScoreType            string, Score type, i.e. standard or adjusted
            - bVerbose              boolean, verbose (print intermediate output)
            - sOptMeth              string, optimization method, i.e. IG, RDA or MD
            - dMaxJTSize            double, maximum junction tree size (only applicable if sGraphType is maxJTsize)
            - iIterInit             integer, number of initial iterations
            - dConv                 double, convergence parameter for termination condition (if L2 improvement after 100 iters < dConv, terminate)
            - bCombineLevels         boolean, use binary hierarchical splitting in initial step for "better" discretization.
            - dAplha                double, fraction of (squared) privacy budget for measuring, 1-alpha for selection
            - dBeta                 double, fraction of (squared) privacy budget in selection for selecting a candidate for each cleint (first step).
                                             1-beta for second step where the a (noisy) best of those candidates is selected.
            - iT                    integer, number of selection rounds
            - iIterPerMeasure       integer, number of iterations of estimation after each measure
            - iIterFInal            integer, final number of iterations of estimation
            - dTheta                double, parameter for filtering candidates
            - lWorkload             list of possible cliques to measure. If None, all combinations of max iMaxDegree is used
            - iMaxLevels            integer, max number of levels a clique can have (only used if dTheta is None)
            - iMaxChecks            integer, max number of cliques to check per client (randomly select from all candidates)
            - dQuadInitBudgetProp   double, (quadratic) proportion of privacy budget used for initialization.
            - iMaxRoundsNumerical   ineger, number of binary split to generate bins for numerical variables
        Set:
            - self                  a FedApdapIterPGM object
        """

        #set general attributes
        super().__init__( dMu, iMaxDegree, rng = rng, iSeed = iSeed, sGraphType=sGraphType, sScoreType=sScoreType, 
                         bVerbose=bVerbose, sOptMeth=sOptMeth, dMaxJTsize= dMaxJTsize, iIterInit = iIterInit, dConv=dConv,
                         sSelectionMethod="adaptive-iterative", bCombineLevels=bCombineLevels, dQuadPropCorrBinning = dQuadPropCorrBinning,
                         iMaxRoundsNumerical = iMaxRoundsNumerical)

        #set budget proportions
        if dQuadInitBudgetProp is None:
            if bCombineLevels == False:
                self.dQuadInitBudgetProp= 0.2
            else:
                self.dQuadInitBudgetProp = 0.35
        else:
            self.dQuadInitBudgetProp = dQuadInitBudgetProp
     
        #set subclass specific attributes
        self.dAlpha = dAlpha
        self.dBeta = dBeta 
        self.iT = iT 
        self.iIterPerMeasure = iIterPerMeasure
        self.iIterFinal = iIterFinal 
        self.lMeasuredCliques = []
        self.dMuUsed = 0.0
        self.dTheta = dTheta
        self.lWorkload = lWorkload
        self.ldiAnswers = None
        self.iMaxLevels = iMaxLevels
        self.iMaxChecks = iMaxChecks

    
    def filter_candidates(self, fedPGMDataset, dMuMeasure, iMinCliqueSize = 2, iMaxLevels = None):
        """
        Goal: filter candidates of workload on size
        Input:
            - self              a FedAdapIterPGM object
            - fedPGMDataset     a fedPGMDataset object
            - dMuMeasure        double, next privacy budget used for measuring
        Output:
            - lCandidates       list of candidates
        """
        #get workload if None
        if self.lWorkload is None:
            self.lWorkload = powerset(fedPGMDataset.domain, iMaxLevel=self.iMaxDegree)[1:]

        #get current downward closure
        lDownwardClosure = downward_closure(self.lMeasuredCliques)
        graphicalModel = GraphicalModel(fedPGMDataset.domain, self.lMeasuredCliques)

        #initialize candidates list
        lCandidates = []

        #check for all candidates in workload
        for cl in self.lWorkload:
            if cl in lDownwardClosure:
                lCandidates.append(cl)
            else: 
                #get size
                juncTree = JunctionTree(fedPGMDataset.domain, self.lMeasuredCliques + [cl])
                maxCliques = juncTree.maximal_cliques()
                potentialSize = sum(fedPGMDataset.domain.size(cliq) for cliq in maxCliques)
                #potentialSize = GraphicalModel(fedPGMDataset.domain, self.lMeasuredCliques + [cl]).size

                #check if size small enough
                if potentialSize*8/2**20 <= self.dMaxJTsize * (self.dMuUsed/self.dMu)**2:
                    lCandidates.append(cl)

        #set maximum number of levels of a marginal
        if iMaxLevels is None:
            iMinExpectedCountsPerMargLevel = self.dTheta*np.sqrt(fedPGMDataset.iC)*np.sqrt(2/np.pi)/dMuMeasure#*np.sqrt(len(fedPGMDataset.domain))    10
            dSize = fedPGMDataset.dNoisyTotal
            iMaxLevels = int(np.ceil(dSize/iMinExpectedCountsPerMargLevel))

        #filter workload with maximum number of level
        lCandidates =  [cl for cl in lCandidates if fedPGMDataset.corrDomain.size(cl) <= iMaxLevels]

        #filter the "empty" candidate
        lCandidates = [cl for cl in lCandidates if len(cl) > (iMinCliqueSize - 1)]

        return lCandidates


    def get_promising_candidates(self, fedPGMDataset, lCandidates, dMuSel1, dMuMeasure, iMaxChecks = 500):
        """
        Goal: get for every client a promising candidate
        Input:
            - self              a FedAdapIterPGM object
            - fedPGMDataset     a fedPGMDataset object
            - lCandidates       list of candidates we can measure
            - dMuSel1           privacy budget for selection-1 step
            - dMuMeasure        privacy budget for measure step
        Output:
            - lPromisingCandidates  list of promising candidates (for every client one candidate)
        """
        #get number of clients
        iC = fedPGMDataset.iC

        #initialize promising candidates list
        lPromisingCandidates = []

        #transform mu to epsilon
        dEps = np.log((1.0/norm.cdf(-0.5*dMuSel1)) - 1.0)

        #loop over clients
        iCand = len(lCandidates)
        for c in range(iC):
            # apply exponential mechanism ro find promising candidates
            if iCand > iMaxChecks:
                #check 250 2-way marginals and 250 3-way marginals 
                lTwoWayCandidates = [el for el in lCandidates if len(el) ==2]
                lThreePlusWayCandidates = [el for el in lCandidates if len(el) > 2]
                lCandidatesTemp1 = self.rng.choice(lTwoWayCandidates, size = min(len(lTwoWayCandidates),int(round(iMaxChecks/2))), replace = False)
                lCandidatesTemp2 = self.rng.choice(lThreePlusWayCandidates, size = min(len(lThreePlusWayCandidates),int(round(iMaxChecks/2))), replace = False)
                lCandidatesTemp = [tuple(el) for el in lCandidatesTemp1] + [tuple(el) for el in lCandidatesTemp2]
       
            else:
                lCandidatesTemp = lCandidates
            vScores = FedDifPrivPGM.get_scores(self.ldiAnswers[c], self.engine.model, lCandidatesTemp, dMuMeasure, fedPGMDataset.vNoisyFrac[c], fedPGMDataset.diCorrBins, fedPGMDataset.corrDomain, sScoreType = self.sScoreType)

            #multiply scores with DP variable
            vScoresCandidates = 0.5*dEps*vScores

            #find noisy candidate
            vProbs = np.exp(vScoresCandidates - logsumexp(vScoresCandidates))
            iChoice = self.rng.choice(len(vProbs), p = vProbs)
            promisingCandidate = lCandidatesTemp[iChoice]
                
            #add candidate
            lPromisingCandidates += [promisingCandidate]

        #return promising candidates
        return lPromisingCandidates

    def get_final_candidate(self, fedPGMDataset, lPromisingCandidates, dMuSel2, dMuMeasure):
        """
        Goal: get final candidate from promising candidates
        Input:
            - self                  a FedAdapIterPGM object
            - fedPGMDataset         a FedPGMDataset object
            - lPromisingCandidates  list of promising candidate cliques to measure
            - dMuSel2               double privacy budget for second selection stage
            - dMuMeasure            double, privacy budget for measure stage
        Output:
            - finalCandidate        clique to be measured in next step
        """

        #get size
        iC = fedPGMDataset.iC

        #set noise
        dSigma = np.sqrt(iC) * 1/dMuSel2 #we measure iC scores, for each candidate one, so we need to multiply the noise with  sqrt(iC)

        #initialize dictionary with scores
        diScores = dict()

        #loop over candidates to get scores
        for candidate in lPromisingCandidates:
            dTempScore = 0.0
            #loop over clients
            for c in range(iC):
                dTempScore += FedDifPrivPGM.get_score(fedPGMDataset.ldf[c], self.engine.model, candidate, dMuMeasure, fedPGMDataset.vNoisyFrac[c], sScoreType = self.sScoreType) + self.rng.normal(loc = 0, scale = dSigma, size = 1)
            diScores[candidate] = dTempScore

        #return maximum 
        finalCandidate = max(diScores, key=diScores.get)
        
        #return
        return finalCandidate

    def fit(self, ldf, diMinMaxInfo, diCatUniqueValuesInfo, lDataTypes = None):
        """
        Goal: FL version of AIM
        Input:
        - self                  a FedAdapIterPGM object.
        - ldf                   a list of pandas dataframes.
        - diMinMaxInfo          dictionariy with minimum and maximum values of ordinal and numerical variables
        - diCatUniqueValuesInfo dictionary with unique values for categorical variables.
        - lDataTypes            list of datatypes of the variables.
        - iMaxRoundsNumeric     integer, number of binary splits to create bins 
        Set
        - engine                The Engine (and therefore the model) of the FedAllInOnePGM object.       
        output:
        - modelFinal            Final model.
        - fedPGMDataset         A fedPGMDataset with all measuremenets.
        """


        #################################
        # Initialization
        #################################

        #set max number of rounds
        iD = len(ldf[0].columns)
        if self.iT is None:
            self.iT = 10 * iD
        iC = len(ldf)

        #set step
        iStep = 0

        #get initial budget
        dMu0 = np.sqrt(self.dQuadInitBudgetProp)*self.dMu

        if self.bVerbose:
            print("start discritization and initial measurement")

        #check which initialization to use
        if self.bCombineLevels:
            #binairy hierarchical splitting
            dMuMeasure =  np.sqrt((1.0 - self.dQuadInitBudgetProp)*self.dAlpha)*self.dMu
            fedPGMDataset = FedPGMDataset(ldf, diMinMaxInfo=diMinMaxInfo, diCatUniqueValuesInfo= diCatUniqueValuesInfo,  rng = self.rng, lDataTypes=lDataTypes, dMu = dMu0, dMuMeasure=dMuMeasure, dQuadPropCorrBinning = self.dQuadPropCorrBinning, iMaxRoundsNumerical = self.iMaxRoundsNumerical)
        else:
            #standard splitting
            fedPGMDataset = FedPGMDataset(ldf, diMinMaxInfo=diMinMaxInfo, diCatUniqueValuesInfo= diCatUniqueValuesInfo, rng = self.rng, lDataTypes=lDataTypes, dMu = 0.0, dQuadPropCorrBinning = self.dQuadPropCorrBinning, iMaxRoundsNumerical = self.iMaxRoundsNumerical)

            #use dMuInit instead for measurements
            lOneWayCliques = [(col,) for col in fedPGMDataset.domain] 
            fedPGMDataset.get_measurements_all_clients(lOneWayCliques, rng = self.rng, dMu = dMu0)

        if self.bVerbose:
            print("start initial estimation")

        #set 1-way marginals as measured
        lCurrentCliques = [(col,) for col in fedPGMDataset.domain]

        if self.bVerbose:
            print("Start initial model estimation")
        
        #set engine
        self.engine = FactoredInference(fedPGMDataset.domain, warm_start=True)
        self.engine.iters = self.iIterInit

        #setup engine
        self.engine._setup(fedPGMDataset.lMeasurements, total = None)
    
        if (self.sOptMeth == "RDA" or self.sOptMeth == "IG"): 
            #get lipschitz
            dLipschitz = get_lipschitz_constant(self.engine, fedPGMDataset.lMeasurements)
            currentModel = self.engine.estimate(fedPGMDataset.lMeasurements, engine=self.sOptMeth, options = { "lipschitz" : dLipschitz, "dConvergence" : self.dConv})
        else:
            currentModel = self.engine.estimate(fedPGMDataset.lMeasurements, engine=self.sOptMeth)

        #update privacy budget
        self.dMuUsed = dMu0
        if self.bVerbose:
            print("End initial model estimation")
            print(f"Squared Privacy Budget used: {round(100*(self.dMuUsed/self.dMu)**2,3)}%")

        #################################
        # adaptive steps
        #################################        

        #get budgets
        dMuStep = np.sqrt(1.0/self.iT)*self.dMu
        dMuMeasure = np.sqrt(self.dAlpha)*dMuStep
        dMuSelection = np.sqrt(1.0-self.dAlpha)*dMuStep 
        dMuSel1 = np.sqrt(self.dBeta)*dMuSelection
        dMuSel2 = np.sqrt(1.0-self.dBeta)*dMuSelection

        #define workload
        self.lWorkload = powerset(fedPGMDataset.domain, iMaxLevel=self.iMaxDegree)[1:]

        #get answers workloads
        self.ldiAnswers = [{cl: get_counts(fedPGMDataset.ldf[c], cl, fedPGMDataset.diCorrBins) for cl in self.lWorkload} for c in range(iC)]

        #set intermediate number of iterations
        self.engine.iters = self.iIterPerMeasure

        #start loop
        while self.dMuUsed < self.dMu:

            #update step
            iStep = iStep + 1

            #get list of candidates
            lCandidates = self.filter_candidates(fedPGMDataset, dMuMeasure, iMaxLevels = self.iMaxLevels)

            if len(lCandidates) > 0:

                #get list of promising candidates (every client 1)
                lPromisingCandidates = self.get_promising_candidates(fedPGMDataset, lCandidates, dMuSel1, dMuMeasure, iMaxChecks = self.iMaxChecks)

                #get final candidate
                finalCandidate = self.get_final_candidate(fedPGMDataset, lPromisingCandidates, dMuSel2, dMuMeasure)

                #add final candidate to current cliques
                lCurrentCliques += [finalCandidate]

                #measure clique
                fedPGMDataset.get_measurements_all_clients([finalCandidate], self.rng)

                #fit new model
                oldModel = currentModel

                if (self.sOptMeth == "RDA" or self.sOptMeth == "IG"): 
                    #get lipschitz
                    dLipschitz = get_lipschitz_constant(self.engine, fedPGMDataset.lMeasurements)
                    currentModel = self.engine.estimate(fedPGMDataset.lMeasurements, engine=self.sOptMeth, options = { "lipschitz" : dLipschitz, "dConvergence" : self.dConv})
                else:
                    currentModel = self.engine.estimate(fedPGMDataset.lMeasurements, engine=self.sOptMeth)

                #check counts
                vCountsNew =  get_counts(currentModel, finalCandidate, fedPGMDataset.diCorrBins)#currentModel.project(finalCandidate).datavector()
                vCountsOld =  get_counts(oldModel, finalCandidate, fedPGMDataset.diCorrBins)#oldModel.project(finalCandidate).datavector()
                dChange = np.linalg.norm(vCountsNew - vCountsOld, ord = 1) -  np.sqrt(2.0/np.pi) * (np.sqrt(iC)/dMuMeasure) * fedPGMDataset.corrDomain.size(finalCandidate) #currentModel.domain.size(finalCandidate)
                
                #update used mu
                self.dMuUsed = np.sqrt(self.dMuUsed**2 + dMuStep**2)
                if self.bVerbose:
                    print(f"End iteration: {iStep}. Squared Privacy Budget used: {round(100*(self.dMuUsed/self.dMu)**2,3)}%. Measured marginal = {finalCandidate}")
                
                if dChange <= 0.0:
                    dMuStep = np.sqrt(2)*dMuStep
                    if self.bVerbose:
                        print("double (squared) privacy budget for next iteration")

                #check if this privacy is possible with remaining budget
                if self.dMu**2 - self.dMuUsed**2 <= dMuStep**2:
                    if self.bVerbose:
                        print("after this iteration only one iteration remaining")
                    #use remaining budget
                    dMuStep = np.sqrt(self.dMu**2 - self.dMuUsed**2)

                #update params
                dMuMeasure = np.sqrt(self.dAlpha)*dMuStep
                dMuSelection = np.sqrt((1-self.dAlpha))*dMuStep
                dMuSel1 = np.sqrt(self.dBeta)*dMuSelection
                dMuSel2 = np.sqrt((1-self.dBeta))*dMuSelection
            
            #no candidates left, terminate
            else:
                if self.bVerbose:
                    print("No suitable candidates (noise too high because privacy budget is nothing more than a rounding error), terminating...")
                    break
        
        #set final number of iterations
        self.engine.iters = self.iIterFinal

        #final estimation
        if self.bVerbose:
            self.engine.log = True
        if (self.sOptMeth == "RDA" or self.sOptMeth == "IG"): 
            #get lipschitz
            dLipschitz = get_lipschitz_constant(self.engine, fedPGMDataset.lMeasurements)
            currentModel = self.engine.estimate(fedPGMDataset.lMeasurements, engine=self.sOptMeth, options = { "lipschitz" : dLipschitz, "dConvergence" : self.dConv})
        else:
            currentModel = self.engine.estimate(fedPGMDataset.lMeasurements, engine=self.sOptMeth)


        #return model
        return currentModel, fedPGMDataset

        


