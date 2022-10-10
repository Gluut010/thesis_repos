#goal: define federated dataset object
#author: Julian van Erk
from mbi import Dataset, Domain
import numpy as np
import pandas as pd
from FedDifPrivModels.Utils import isNaN, bin_subset_bin, get_counts
from FedDifPrivModels import FedDataset
from scipy import sparse

class FedPGMDataset(FedDataset):
    """
    A federated dataset object, specially designed for the training part of PGM
    """

    def __init__(self, ldf, diMinMaxInfo, diCatUniqueValuesInfo, rng, lDataTypes = None, iMaxRoundsNumerical = 4, dMu = 0.0, dMuMeasure = 0.0, iMinCountsNumerical = None, iMinCountsCategorical = None, dQuadPropCorrBinning = 0.4):
        
        """
        Goal: constructor for federated dataset object
        Input:
            - ldf                   list of pandas dataframes
            - diMinMaxInfo          dictionary with minimum and maximum info
            - lDataTypes            list of datatypes of the variables (either "numerical", "ordinal" or "categorical")
            - iMaxRoundsNumerica    integer, determines number of bins (=2^(iMaxRoundsNumerical + 1))
            - dMu                   double, total privacy budget
            - dMuMeasure            double, privacy budget for measuring phase
            - iMinCountsCategorical 
            - iMinCountsNumerical   integer, minimum expected counts for a numerical variable
        Set:
            - 
        """    
        #initialize as fedDataset
        super().__init__(ldf = ldf, diMinMaxInfo=diMinMaxInfo, diCatUniqueValuesInfo= diCatUniqueValuesInfo, lDataTypes = lDataTypes)
        self.diBins = dict()
        self.llClientMeasurements = [[] for c in range(self.iC)]
        self.dMu = dMu
        if dMu > 0:
            self.diCorrBins = dict()
            self.corrDomain = None

        #apply preprocessing for PGM
        self.preprocess_PGM(rng = rng, iMaxRoundsNumerical = iMaxRoundsNumerical, dMu = dMu, dMuMeasure=dMuMeasure,
                            iMinCountsNumerical = iMinCountsNumerical, iMinCountsCategorical = iMinCountsCategorical,
                            dQuadPropCorrBinning = dQuadPropCorrBinning)




    def preprocess_PGM(self, rng, iMaxRoundsNumerical = 4, dMu = 0.0, dMuMeasure = 0.0, iMinCountsNumerical = None, iMinCountsCategorical = None, dQuadPropCorrBinning = None):
        """
        Goal: preprocess fedDataset object for PGM by discretizing all variables
        Input:
            - self                      federated dataset object
            - ldfClientTrainDataRaw     list of raw dataframes for all client
            - ldfCLientTestDataRaw      list of raw test set dataframes for all clients
            - dMu                       double privacy budget for preprocessing
            - dMuMeasure                double privacy budget for measuring phase (needed for determining signal to noise ratio, not needed if dMu = 0)
            - iMaxRoundsNumerical       integer maximum number of hierarchical binary splitting rounds of bins
        Set:
            - ldf                       list of Dataset objects for all clients
            - lMeasurements             list with measurements that can be used for estimation
            - Q                         matrices for combining levels

        """    

        #discretize
        self.lMeasurements = self.discretize_all_variables(rng= rng, iMaxRoundsNumerical = iMaxRoundsNumerical, dMu = dMu, dMuMeasure = dMuMeasure,
                                        iMinCountsNumerical = iMinCountsNumerical, iMinCountsCategorical = iMinCountsCategorical,
                                        dQuadPropCorrBinning = dQuadPropCorrBinning)

        #transform to dataset objects
        self.ldf = [Dataset(self.ldf[i], self.domain) for i in range(self.iC)]
        self.get_all_oneway_Q()


        #set some attributes also for the individual datasets
        for i in range(self.iC):
            self.ldf[i].diCorrBins = self.diCorrBins
            self.ldf[i].corrDomain = self.corrDomain
            self.ldf[i].diOneWayQ = self.diOneWayQ



    def get_measurements_all_clients(self, lCliques, rng, dMu = 0.1, vWeights=None):
        """
        Goal: get measurements of all clients for a list of cliques to be measured
        input:
            - self              A federatedPGM dataset
            - lCliques          list of cliques to be measured
            - rng               random number generator
            - dMu               default privacy budget
            - vWeights          np.array of weights for privacy budget per measurement, should be same size as lCliques
        set:
            - lMeasurements     updated list of measurements 
            - vNoisyCountsVar   noisy counts variance vector for all clients
            - vNoisyCounts      noisy counts vector for all clients
            - dNoisyTotal       noisy total number of observations over all clients
            - vNoisyFrac        noisy fraction of observations w.r.t. total for all clients
        output:
            - None
        """     

        #initialize list of measurements
        llTempMeasurements = []

        #check if weights are matrices or use default
        #default option
        if vWeights is None:
            for c in range(self.iC):
                lClientMeasurements = FedPGMDataset.get_measurements_client(self.ldf[c], lCliques, rng, dMu)
                llTempMeasurements.append(lClientMeasurements)
                self.llClientMeasurements[c].append(lClientMeasurements)

            
        #weight the mu's
        else:
            for c in range(self.iC):
                lClientMeasurements = FedPGMDataset.get_measurements_client(self.ldf[c], lCliques, rng, dMu, vWeights = vWeights)
                llTempMeasurements.append(lClientMeasurements)
                self.llClientMeasurements[c].append(lClientMeasurements)


        #get number of measurements
        iMeasurements = len(lCliques)

        #get sum of counts per client
        mTotalCounts = np.zeros((iMeasurements, self.iC))
        mTotalCountsVar = np.zeros((iMeasurements, self.iC))

        #loop over measurements
        for m in range(iMeasurements):

            #get individual variances and counts
            vVariances = np.zeros(self.iC)
            mNoisyCounts = np.zeros((self.iC, len(llTempMeasurements[0][m][1])))

            for c in range(self.iC):

                #get variances and noisy counts
                vVariances[c] = llTempMeasurements[c][m][2]**2
                mNoisyCounts[c,:] = llTempMeasurements[c][m][1]

                #fill total count matrix
                mTotalCounts[m,c] = np.sum(mNoisyCounts[c,:])
                mTotalCountsVar[m,c] = mNoisyCounts.shape[1] * vVariances[c]

            #get summation noisy counts
            dSumNoisyCount = np.sum(mNoisyCounts, axis = 0)

            #get variance summation noisy counts
            dSumSigma = np.sqrt(np.sum(vVariances))

            #add summed 
            lSumMeasurement = (llTempMeasurements[0][m][0], dSumNoisyCount, dSumSigma, llTempMeasurements[0][m][3]) #(mI, vNoisyCounts, sigma, clique)
            self.lMeasurements.append(lSumMeasurement)

        #set noisy fraction estimate
        vTotalInverseSumVar = np.sum(1.0 / mTotalCountsVar, axis = 0) 
        mTempWeights = (1.0 / mTotalCounts) / vTotalInverseSumVar[np.newaxis,:]
        vNoisyCountsPerClient = np.array([ np.average(mTotalCounts[:,c], weights = mTempWeights[:,c]) for c in range(self.iC)])
        vNoisyCountsPerClientVar = 1.0/vTotalInverseSumVar

        #set new count estimates
        self.update_total_counts(vNoisyCountsPerClient, vNoisyCountsPerClientVar)

    @staticmethod
    def get_measurements_client(dataset, lCliques, rng, dMu = 0.1, vWeights=None):
        """
        Goal: get measurements for one client
        input:
            - dataset    mbi dataset object
            - diOneWayQ  dictionary, correlated bins object
            - lCliques   list of cliques to be measured
            - dFrac      fraction of total observations used for this client
            - rng        random number generator
            - dMu        privacy budget for this step (defaults = 0.5/sqrt(3))
            - vWeight    vector of weights per measurement 

        output:
            - lMeasurements  list of measurements, each measuremnt is a tuple of (querymatrix, noisy_counts, sigma, measured_clique)
        """
        #get number of measurements
        iMeasurements = len(lCliques)

        #get weights and normalize
        if vWeights is None:
            vWeights = np.ones(iMeasurements) 
        vWeights = vWeights / np.linalg.norm(vWeights)

        #get mu for all measurements
        vMu = dMu * vWeights
        vSigma =  (1.0 / vMu)
        
        #initialize measurements list
        lMeasurements =[]

        #fill measurements list
        for sigma, clique in zip(vSigma, lCliques):

            #get counts and multiply to original size
            vCounts = get_counts(dataset, clique, dataset.diCorrBins)#dataset.project(clique).datavector()
            mQ = FedPGMDataset.get_Q(dataset.diOneWayQ, clique)#sparse.eye(vCounts.size)
            if mQ.shape[0] == mQ.shape[1]:
                if np.allclose(mQ, np.eye(mQ.shape[0])):
                    mQ = sparse.eye(mQ.shape[0])

            #add noise to counts
            vNoisyCounts = vCounts +  rng.normal(loc = 0, scale = sigma, size = vCounts.size)

            #add noisy counts to measurements list
            lMeasurements += [(mQ, vNoisyCounts, sigma, clique)]

        return lMeasurements

    #remove
    def get_var_name(self, d):
        """
        Goal: Get var name of a federated PGM dataset object
        Input: 
            - self   A federated PGM Dataset object
            - d      column of the var name
        output:
            - sVarName  string with variable name
        """
    
        return self.ldf[0].df.columns[d]

    #remove
    def map_to_default_discritization(self, dfData, rng):
        """
        Goal: map discritezed dataset to other (hierarchical) discritization
        Input:
            - self          a federated PGM dataset object
            - dfData        discretized pandas dataframe that needs to have a different discritization
            - rng           a random number generator
        Output
            - dfDataNew     dfData with the new discretization
        """

        #get number of columns
        iD = self.iD

        #predefine new data dict
        diNewData = dict()

        #loop over columns
        for d in range(iD):
            #get variable name
            sVarName = dfData.columns[d]

            #get bins
            lBinsSelf = self.diFinalBins[sVarName]
            lBinsOther = self.diDefaultBins[sVarName]

            #initialize map dictionary
            diMap = dict()

            #get number of different values original and other
            iNumValuesSelf = len(lBinsSelf)
            iNumValuesOther = len(lBinsOther)

            #perform transformation
            if (self.lDataTypes[d] == "numerical") or (self.lDataTypes[d] == "ordinal"):   
                for i in range(iNumValuesSelf):
                    diMap[i] = []
                    for j in range(iNumValuesOther):
                        #get bins
                        binSelf = lBinsSelf[i]
                        binOther = lBinsOther[j]
                        if bin_subset_bin(binSelf, binOther):
                            diMap[i] += [j]
                            break
                        elif bin_subset_bin(binOther, binSelf):
                            diMap[i] += [j]
                
            else: 
                #get index other bin if it exists
                if "other" in lBinsSelf:
                    iOtherIndex = lBinsSelf.index("other")
                for i in range(iNumValuesSelf):
                    diMap[i] = []
                for j in range(iNumValuesOther):
                    binOther = lBinsOther[j]
                    if binOther in lBinsSelf:
                        i = lBinsSelf.index(binOther)
                        diMap[i] += [j]
                    else:
                        #binother has value "other" in binself
                        diMap[iOtherIndex] += [j]
                        

            lNewData = [ rng.choice(diMap[x]) for x in dfData[sVarName].values]
            diNewData[sVarName] = lNewData

        dfNewData = pd.DataFrame.from_dict(diNewData)

        return dfNewData


    def get_all_oneway_Q(self):
        """
        Goal: get all measurement matrices Q for one-way marginals
        Input:
            - self  A fedPGMDataseta object
        Set:
            - diOneWayQ dictionary with one way marginal Q matrices for each variable.
        Output:
            - self
        
        """
        diOneWayQ = dict()
        for sVarName in self.ldf[0].df.columns:
            #get correlation bins
            llCorrBins = self.diCorrBins[sVarName]

            #get number of measurements
            iCorrSize = self.corrDomain.config[sVarName]

            #get original space size
            iOrigSize = self.domain.config[sVarName]

            #set/fill Q
            mQ = np.zeros((iCorrSize, iOrigSize))
            for i in range(mQ.shape[0]):
                mQ[i, llCorrBins[i]] = 1

            #set in dictionariy
            diOneWayQ[sVarName] = mQ

        #set as attribute
        self.diOneWayQ = diOneWayQ

    @staticmethod
    def get_Q(diOneWayQ, clique):
        """
        Goal: Get Q matrix
        Input:
            - self      a fedPGKDataset object
            - clique    clique we would like the Q matrix of
        Output:
            - mQ        the measurement matrix
        """

        mQ = np.array([1])
        
        for i in range(len(clique)):
            mQ = np.kron(mQ, diOneWayQ[clique[i]])
        return mQ


    