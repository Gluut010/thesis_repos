#goal: define federated differentially private probabilistic graphical model object
#author: Julian van Erk
from FedDifPrivModels.FactoredInferenceAdaption import FactoredInference
from FedDifPrivModels.Utils import get_counts
import numpy as np
import itertools as iter

class FedDifPrivPGM:
    """
    A Federated, Differentially private PGM object.
    """

    def __init__(self, dMu, iMaxDegree, rng = None, iSeed = None, sGraphType = "tree", sScoreType = "standard", bVerbose = False,
                 sOptMeth = 'IG', dMaxJTsize = None, sSelectionMethod = "all-in-one", iIterInit = 2500, dConv = 10**-1, bCombineLevels = False,
                 dQuadPropCorrBinning = 0.4, iMaxRoundsNumerical = 4):
        """
        Goal: constructor for generad federated differential private PGM model
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
            - sSelectionMethod      string, method of selection marginals, i.e. all-in-on or adaptive-iterative
            - iIterInit             integere, number of iterations for initialization
            - dConv                 double, convergence parameter for termination condition (if L2 improvement after 100 iters < dConv, terminate)
            - bCombineLevels         boolean, use combine levels approach 
            - iMaxRoundsNumerical   ineger, number of binary split to generate bins for numerical variables
        Set:
            - self                  a FedDifPrivPGM object
        """
        #set random number generator
        if rng is None:
            if iSeed == None:
                self.rng = np.random.default_rng(1234)
            else:
                self.rng = np.random.default_rng(iSeed)
        else:
            self.rng = rng
        
        #set privacy budget
        if dMu <= 0:
            raise ValueError("Privacy budget dMu should be larger than 0")
        self.dMu = dMu

        #set graph type
        lPossibleGraphTypes = ["tree", "maxJTsize"]
        if sGraphType in lPossibleGraphTypes:
            self.sGraphType = sGraphType
            if sGraphType == "maxJTsize":
                if dMaxJTsize is None:
                    self.dMaxJTsize = 20
                else:
                    self.dMaxJTsize = dMaxJTsize
        else:
            raise ValueError(f"graphType {sGraphType} not implemented, use one of {lPossibleGraphTypes}")

        #set scoreType
        lPossibleScoreTypes = ["standard", "adjusted", "random"]
        if sScoreType in lPossibleScoreTypes:
            self.sScoreType = sScoreType
        else:
            raise ValueError(f"scoretype {sScoreType} not implemented, use one of {lPossibleScoreTypes}")

        #set optimization method
        lPossibleOptMeths = ["MD", "RDA", "IG"]   
        if sOptMeth in lPossibleOptMeths:
            self.sOptMeth = sOptMeth
        else: 
            raise ValueError(f"opt. meth, {sOptMeth} not implemented, use one of {lPossibleOptMeths}")

        #set max clique size
        self.iMaxDegree = iMaxDegree  
        if self.iMaxDegree > 3 and self.sGraphType == "tree":
            raise ValueError("A tree can only have max clique size of 2")     

        #set verbose
        self.bVerbose = bVerbose

        #set domain and engine
        self.engine = None

        #set measure method
        lPossibleMeasureMethods = ["all-in-one", "adaptive-iterative"]
        if sSelectionMethod in lPossibleMeasureMethods:
            self.sSelectionMethod = sSelectionMethod

        #set initial iters
        self.iIterInit = iIterInit

        #set convergence param
        self.dConv = dConv

        #set max numer of binary splits
        self.iMaxRoundsNumerical = iMaxRoundsNumerical

        #set quad prop corr bin
        self.dQuadPropCorrBinning = dQuadPropCorrBinning

        #set binairy hierarchical splitting true/false
        self.bCombineLevels = bCombineLevels

    @staticmethod
    def get_score(dataset, model, clique, dMuNext, dFrac, sScoreType = "standard"):
        """
        Goal: get expected improvement score for this client if signal-to-noise ratio is same as total dataset
        Input:
            - dataset       dataset of this client
            - model         current graphical model
            - clique        clique/marginal we want the score of
            - dMuNext       mu that will be used if we select this clique to measure
            - dFrac         fraction of observations of this client dataset w.r.t. total dataset
            - sScoreType    score function to use

        Output
            - dScore        score (the expected imporvement in L1 error by measuring this marginal with same signal-to-noise ratio as total)
        """
        if dFrac <0:
            dFrac =0
        if sScoreType == "standard":
            #get counts according to model and real counts
            vCountsHat = dFrac * get_counts(model, clique, dataset.diCorrBins) #get_counts(dataset, clique, diCorrBins)
            vCounts = get_counts(dataset, clique, dataset.diCorrBins)#dataset.project(clique).datavector()

            #get score
            dScore = np.linalg.norm(vCounts - vCountsHat, ord = 1) - np.sqrt(2.0/np.pi) * np.sqrt(dFrac) * (1.0/dMuNext) * dataset.corrDomain.size(clique)
        elif sScoreType == "local":
            #get counts according to model and real counts
            vCountsHat = get_counts(model, clique, dataset.diCorrBins) #get_counts(dataset, clique, diCorrBins)
            vCounts = get_counts(dataset, clique, dataset.diCorrBins)#dataset.project(clique).datavector()

            #get score
            dScore = np.linalg.norm(vCounts - vCountsHat, ord = 1) - np.sqrt(2.0/np.pi) * np.sqrt(dFrac) * (1.0/dMuNext) * dataset.corrDomain.size(clique)
        elif sScoreType == "adjusted":
            #get counts according to model and real counts
            vCountsHat = dFrac * get_counts(model, clique, dataset.diCorrBins)
            vCounts = get_counts(dataset, clique, dataset.diCorrBins)

            #get norm of 1-way marg count differences
            iL = len(clique)
            vOneWayNorms = np.zeros(len(clique))
            for i in range(iL):
                vOneWayNorms[i] = np.linalg.norm(dataset.project((clique[i],)).datavector() - dFrac * model.project((clique[i],)).datavector(), ord = 1)

            #get max one way norm
            dMaxNorm = np.max(vOneWayNorms)

            #get score
            dScore = 0.5*(np.linalg.norm(vCounts - vCountsHat, ord = 1) - dMaxNorm - np.sqrt(2.0/np.pi) * np.sqrt(dFrac) * (1.0/dMuNext) * dataset.corrDomain.size(clique))

        elif sScoreType == "random":
            dScore = 0.0
        return dScore

    @staticmethod
    def get_scores(diAnswers, model, lCliques, dMuNext, dFrac, diCorrBins, corrDomain, sScoreType = "standard"):
        """
        Goal: get expected improvement score for this client if signal-to-noise ratio is same as total dataset, with precalculated answers
        Input:
            - diAnswers     precalculated answers of all queries (for efficiency) (for a specific client)
            - model         current model
            - lClique       cliques/marginals we want the score of
            - dMuNextClient mu that will be used if we select this clique to measure
            - dFrac         fraction of observations of this client dataset w.r.t. total dataset
            - sScoreType    score function to use

        Output
            - vScore        score (the expected imporvement in L1 error by measuring this marginal with same signal-to-noise ratio as total)
        """
        if dFrac <0:
            dFrac =0
        #get score
        if sScoreType == "standard":
            dPenalty= np.sqrt(2.0/np.pi) * np.sqrt(dFrac) * (1.0/dMuNext)
            vScore = np.array([np.linalg.norm( diAnswers[clique] - dFrac *  get_counts(model, clique, diCorrBins), ord = 1) - dPenalty * corrDomain.size(clique) for clique in lCliques])
        elif sScoreType == "adjusted":
            #get penalty
            dPenalty= np.sqrt(2.0/np.pi) * np.sqrt(dFrac) * (1.0/dMuNext)

            #get non-iid norm for all variables
            lUniqueVars = list(set(iter.chain.from_iterable(lCliques)))
            diOneWayModelNorms = {}
            for el in lUniqueVars:
                diOneWayModelNorms[el] = np.linalg.norm(diAnswers[(el,)] - dFrac * model.project((el,)).datavector(), ord = 1)

            #get max norm per clique
            diMaxOneWayModelNorms = {}
            for cl in lCliques:
                dMaxNorm = 0.0
                for el in cl:
                    dTempNorm = diOneWayModelNorms[el]
                    dMaxNorm = dMaxNorm + dTempNorm
                    if dTempNorm > dMaxNorm:
                        dMaxNorm = dTempNorm
                diMaxOneWayModelNorms[cl] = dMaxNorm

            #get and return scores
            vScore = np.array([0.5*(np.linalg.norm( diAnswers[clique] - dFrac *  get_counts(model, clique, diCorrBins), ord = 1) - diMaxOneWayModelNorms[clique] - dPenalty * corrDomain.size(clique)) for clique in lCliques])
        elif sScoreType == "random":
            vScore = np.array([0.0 for clique in lCliques])
        return vScore
