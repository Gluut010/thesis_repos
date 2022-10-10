#goal: define federated dataset object
#author: Julian van Erk
from mbi import Dataset, Domain
import numpy as np
import pandas as pd
from FedDifPrivModels.Utils import isNaN, bin_subset_bin, delim_ceil, delim_floor
from collections import Counter

class FedDataset:
    """
    A federated dataset object
    """
    
    def __init__(self, ldf, diMinMaxInfo, diCatUniqueValuesInfo, lDataTypes = None):
        
        """
        Goal: constructor for federated dataset object
        Input:
            - ldf                list of pandas dataframes
            - diMinMaxInfo       dictionary with min and max values for numerical and ordianl variables
            - diCatUnqieuValues  Info dictionary with unique categorical values info
            - lDataTypes         list of datatypes of the variables (either "numerical", "ordinal" or "categorical")
        Set:
            - self
        """
        #set list of dataframes
        self.diMinMaxInfo = diMinMaxInfo
        self.diCatUniqueValuesInfo = diCatUniqueValuesInfo
        self.ldf = ldf
        self.lDataTypes = lDataTypes
        self.vNoisyFrac = None
        self.vNoisyCounts = None
        self.vNoisyCountsVar = None
        self.dNoisyTotal = None
        self.domain = None

        #get attributes from dataset
        self.iC = len(self.ldf)
        self.iD = self.ldf[0].shape[1]

        #set datatypes, if not provided use get_data_types function
        if lDataTypes is None:
            self.get_data_types()
        else:
            self.lDataTypes = lDataTypes

    def get_data_types(self):
        """
        Goal: Get datatypes of pandas dataframe
        Input:
            - self          fedDataset object
        Set:
            - lDataTypes    list of dataypes (int64, float64 --> numerical. object, category --> categorical)
        Output:
            None
        """

        #initialize datatype list
        lDataTypes = []

        #loop over attributes
        for d in range(self.iD):
            sDataType = self.ldfRaw.dtypes[d] 
            if (sDataType == "float64") or (sDataType == "int64"):
                lDataTypes.append("numerical")
            elif (sDataType == "object") or (sDataType == "category"):
                lDataTypes.append("categorical")
            else:
                raise ValueError(f"datatype {sDataType} not supported")

        #set datatype attribute
        self.lDataTypes = lDataTypes


    def discretize_all_variables(self, rng, iMaxRoundsNumerical = 4, dMu = 0.0, dMuMeasure = 0.0, iMinCountsNumerical = None, iMinCountsCategorical = None, dQuadPropCorrBinning =0.4):
        """
        Goal: Discretize all variables
        Input:
            - self                      federated dataset object
            - ldfClientTrainDataRaw     list of raw dataframes for all client
            - ldfCLientTestDataRaw      list of raw test set dataframes for all clients
            - dMu                       double privacy budget for preprocessing
            - rng                       random number generator
            - iMaxRoundsNumerical       integer maximum number of hierarchical binary splitting rounds of bins
            - dQuadPropCorrBinning      double, quadratic proportion for measuring correlation bins.
        Set:
            - ldf                       list of ordinal encoded dataframes for all clients
            - diFinalBIns               dictionary with split info
            - domain                    discrete domain object
        Output:
            %- diSplitInfo               dictionary with split info (REMOVE)
            - lMeasurements             list with measurements that can be used for estimation
        """                             
        #set diBins
        self.diBins = dict()
        self.diCorrBins = dict()

        #get number of clients
        iC = self.iC

        #get number of variables
        iD = self.iD

        #get privacy budget single variable
        dMuSingle = dMu / np.sqrt(iD)

        #initialize encoded datasets dictionaries
        ldiEncoded = [dict() for i in range(iC)]

        #initialize measurements and dictionairy diSplitInfo
        lMeasurements = []

        #special case: mu is 0 (no learning about distribution):
        if dMu == 0.0:
            iMinCountsCategorical = -np.inf
            iMinCountsNumerical = -np.inf

        #loop over variables
        for i in range(iD):
            #get name
            sVarName = self.ldf[0].columns[i]

            #get data type and discretize
            if self.lDataTypes[i] == "numerical":
                #discretize numerical
                llDiscretizedTrainSingleVar, lMeasurementsTemp = self.discretize_numeric_variable(sVarName=sVarName, dMu = dMuSingle, dMuMeasure= dMuMeasure, iMaxRounds = iMaxRoundsNumerical, rng = rng, iMinCounts=iMinCountsNumerical, dQuadPropCorrBinning = dQuadPropCorrBinning)
            elif self.lDataTypes[i] == "ordinal":
                #discretize ordinal as numerical
                llDiscretizedTrainSingleVar, lMeasurementsTemp = self.discretize_numeric_variable(sVarName=sVarName, dMu = dMuSingle,dMuMeasure= dMuMeasure, iMaxRounds = iMaxRoundsNumerical, rng = rng, iMinCounts=iMinCountsNumerical, bAllUniqueValuesKnown = True, dQuadPropCorrBinning = dQuadPropCorrBinning)
            elif self.lDataTypes[i] == "mixed":
                llDiscretizedTrainSingleVar, lMeasurementsTemp = self.discretize_numeric_variable(sVarName=sVarName, dMu = dMuSingle,dMuMeasure= dMuMeasure, iMaxRounds = iMaxRoundsNumerical, rng = rng, iMinCounts=iMinCountsNumerical, dQuadPropCorrBinning = dQuadPropCorrBinning, bMixed=True)
            else: 
                #discretize categorical
                llDiscretizedTrainSingleVar, lMeasurementsTemp = self.discretize_categorical_variable(sVarName = sVarName, dMu = dMuSingle,dMuMeasure= dMuMeasure, rng = rng, iMinCounts=iMinCountsCategorical, dQuadPropCorrBinning = dQuadPropCorrBinning)

            #add measuremetns
            lMeasurements += lMeasurementsTemp

            #add discretized variables to dictionaries
            for c in range(iC):
                ldiEncoded[c][sVarName] = llDiscretizedTrainSingleVar[c]
        
        #tranform dictionairy to pandas datafre
        self.ldf = [pd.DataFrame.from_dict(ldiEncoded[c]) for c in range(iC)]

        #set domains
        self.domain = self.get_domain()
        self.corrDomain = self.get_domain(diBins = self.diCorrBins)

        #return diSplitInfo, lMeasurements
        return lMeasurements


    def discretize_numeric_variable(self, sVarName, dMu, dMuMeasure, rng, iMaxRounds = 4,
                                     iMinCounts = None, bAllUniqueValuesKnown = False, 
                                     dQuadPropCorrBinning = 0.4, bMixed = False):
        """
        Goal: create bins [a,b) by means of hierarchical binary splitting. If there are enough observations in a bin [c,d), 
        split bin in 2 new bins [c, (c+d)/2), [(c+d)/2, d)
        Input:
            - llNumerical        List of pd dataframes with numerical train values. Every client one dataframe.
            - sVarName           String variable name. Needed for defining measurments
            - dMu                Double, privacy budget we can use for discretizing this variable
            - dMuMeasure         Double, total privay budget used for measure step
            - rng                random number generator
            - iMaxRounds         max number of splitting times
            - iMinCounts         min number of counts before splitting is possible
            - dQuadPropCorrBinning      double, quadratic proportion for measuring correlation bins.
            - bMixed             boolean, is this a mixed variable which should have a "zero" bin?
        Output
            - lFinalBins            List of final bins in form [ [a,b], [b,c],...]
            - llDiscretizedTrain    list of discretized train sets of the clients.
            - lUniqueValuesPerBin   list with all unique values per bin (for transforming back)
            - lMeasurements         list with measurements that can be used for the model.
        """

        iC = self.iC

        #change to numpy]
        llNumerical = [np.array(self.ldf[c][sVarName]) for c in range(iC)]


        #identify min, max and middlepoint
        if bMixed:
            dMin = self.diMinMaxInfo[sVarName][0] 
            if dMin == 0.0:
                dMin = dMin + 10e-8
        else:
            dMin = self.diMinMaxInfo[sVarName][0] #np.nanmin([np.nanmin(llNumerical[c]) for c in range(iC)])
        dMax = self.diMinMaxInfo[sVarName][1] + 10e-8 #np.nanmax([np.nanmax(llNumerical[c]) for c in range(iC)]) + 10e-8
        dMiddle = (dMax + dMin) / 2

        #check if missing values
        bMissingValues = False
        for c in range(iC):
            if np.isnan(llNumerical[c]).any():
                bMissingValues = True

        #set correlation bins
        llCorrBins = []

        #get bins
        if not bMixed:
            lCurrentBins = [[dMin, dMiddle], [dMiddle, dMax]]
            if bMissingValues:
                lCurrentBins.append(["NaN"])
        if bMixed:
            lCurrentBins = [['zero'],[dMin, dMiddle], [dMiddle, dMax]]
            if bMissingValues:
                lCurrentBins.append(["NaN"])
        llBins = []
        llBins.append(lCurrentBins)
        iCurrentBins = len(lCurrentBins)


        #list whether splitting is possible
        lCurrentPossibleSplits = iCurrentBins*[True]
        for i in range(len(lCurrentPossibleSplits)):
            if (lCurrentBins[i] == ["NaN"]) | (lCurrentBins[i] == ["zero"]):
                lCurrentPossibleSplits[i] = False       

        #set noise
        dMuInit = dMu*np.sqrt(1- dQuadPropCorrBinning)
        dMuCorrBinning = dMu*np.sqrt(dQuadPropCorrBinning)
        if dMu != 0:
            dSigma = np.sqrt(iMaxRounds)*(1.0/(dMuCorrBinning))

        #set counts list of list
        llCounts = []

        #get unique values
        lUniqueValues = list({el for list in llNumerical for el in list})
        
        #remove missing value
        if bMissingValues:
            lUniqueValues = [el for el in lUniqueValues if isNaN(el) == False]
            #lUniqueValues += ['NaN']

        #get number of unique values
        iUniqueValues = len(lUniqueValues)


        #set iMinCount
        if iMinCounts is None: 
            dTheta = 6.0
            iMinCounts = 2*dTheta*np.sqrt(2/np.pi)*(np.sqrt(self.iC)/(dMuMeasure/np.sqrt(self.iD)))#
        #loop
        for splitRound in range(0, iMaxRounds):

            #initialize current counts
            lCurrentCounts = iCurrentBins*[0.0]
            
            #get current splits
            lCurrentSplits = [el[1] for el in lCurrentBins if len(el)>1]
            if bMixed:
                lCurrentSplits = [10e-8] + lCurrentSplits

            #get unique values per bin
            lUniqueValuesDiscretized = np.digitize(x = lUniqueValues, bins = lCurrentSplits)
            iBins = len(lCurrentBins)
            lUniqueValuesPerBin = [[] for _ in range(iBins)]#iBins*[[]]
            for i in range(len(lUniqueValues)):
                lUniqueValuesPerBin[lUniqueValuesDiscretized[i]] = lUniqueValuesPerBin[lUniqueValuesDiscretized[i]] + [lUniqueValues[i]]

            #if the privacy budget is not zero, record the counts.
            if dMu !=0:
                lCurrentCounts = self.get_all_numeric_counts_this_round(llNumerical, lCurrentCounts, lCurrentBins, dSigma, rng)

            #add current noisy counts to overall counts
            llCounts.append(lCurrentCounts)

            #splitting
            lCurrentBinsNew = []
            lCurrentPossibleSplitsNew = []
            for i in range(iCurrentBins):

                #get current bin
                bin = lCurrentBins[i]

                if bAllUniqueValuesKnown:
                    #check if bin has more than 1 unique value
                    if len(lUniqueValuesPerBin[i]) <= 1:
                        lCurrentPossibleSplits[i] = False

                #splitting
                if lCurrentPossibleSplits[i]:

                    #get new bin 
                    dTempMin = bin[0]
                    dTempMax = bin[1]
                    dMiddle = (dTempMax + dTempMin) / 2
                    #split bins
                    lCurrentBinsNew.append([dTempMin, dMiddle])
                    lCurrentBinsNew.append([dMiddle, dTempMax])
                    #possible splits
                    lCurrentPossibleSplitsNew.append(True)
                    lCurrentPossibleSplitsNew.append(True)
                else:
                    #add NaN bin
                    lCurrentBinsNew.append(bin)
                    lCurrentPossibleSplitsNew.append(False)

                #splitting

                if ((lCurrentCounts[i] < iMinCounts) & (lCurrentPossibleSplits[i])): 
                    if bAllUniqueValuesKnown:
                        lTempCorrList = [j for j in range( int(np.ceil(bin[0])), int(np.ceil(bin[1] - 10e-8)))]
                    elif bMixed:
                        lTempCorrList = [j for j in range(2**(iMaxRounds - splitRound)*(i-1) + 1 , 2**(iMaxRounds - splitRound)*(i) + 1)]
                    else:
                        lTempCorrList = [j for j in range(2**(iMaxRounds - splitRound)*i , 2**(iMaxRounds - splitRound)*(i+1))]
                    #check if super list is not already in there
                    if not any( [set(lTempCorrList).issubset(sublist) for sublist in llCorrBins]):
                        llCorrBins.append(lTempCorrList)
                    

            #update bins
            lCurrentBins = lCurrentBinsNew.copy()
            lCurrentPossibleSplits = lCurrentPossibleSplitsNew.copy()
            llBins.append(lCurrentBins)

            #update
            iCurrentBins = len(lCurrentBins)

        #update llCorrBins
        if bMissingValues:
            iUniqueValues = iUniqueValues + 1
        if bAllUniqueValuesKnown:
            for i in range(iUniqueValues):
                if not any( i in sublist for sublist in llCorrBins):
                    llCorrBins.append([i])
        elif bMixed:
            for i in range(2**(iMaxRounds + 1) + 1 + bMissingValues):
                if not any( i in sublist for sublist in llCorrBins):
                    llCorrBins.append([i])
        else:
            for i in range(2**(iMaxRounds + 1) + bMissingValues):
                if not any( i in sublist for sublist in llCorrBins):
                    llCorrBins.append([i])
        
        #final splits
        lFinalBins = lCurrentBins.copy()
        lFinalSplits = [el[1] for el in lCurrentBins if len(el) >1]
        if bMixed: 
            lFinalSplits = [10e-8] + lFinalSplits

        #get final number of bins
        iTotalBins = len(lFinalBins)

        #final counts
        lCurrentCounts = iTotalBins*[0.0]
        if dMu !=0:
            dSigmaFinal = (1.0/dMuInit)
            lCurrentCounts = self.get_all_numeric_counts_this_round(llNumerical, lCurrentCounts, lFinalBins, dSigmaFinal, rng)

        #add current noisy counts to overall counts
        llCounts.append(lCurrentCounts)

        #get discretized train data for all clients
        llDiscretizedTrain = iC*[None]
        for c in range(iC):
            llDiscretizedTrain[c] = np.digitize(x = llNumerical[c], bins = lFinalSplits)

        #get tranformation (back) object
        lUniqueValuesDiscretized = np.digitize(x = lUniqueValues, bins = lFinalSplits)
        iBins = len(lFinalSplits)
        llUniqueValuesPerBin = [[] for _ in range(iBins)]#iBins*[[]]
        for i in range(len(lUniqueValues)):
            llUniqueValuesPerBin[lUniqueValuesDiscretized[i]] = llUniqueValuesPerBin[lUniqueValuesDiscretized[i]] + [lUniqueValues[i]]

        # update final bins
        self.diBins[sVarName] = lFinalBins

        #order llCorrBins and to diCorrBins
        llCorrBins.sort()
        self.diCorrBins[sVarName] = llCorrBins

        #predefine measurements
        lMeasurements = []
        if dMu == 0:
            return llDiscretizedTrain, lMeasurements

        #fill measurements
        for i in range(len(llCounts)):
            vTempCounts = np.array([el[0] for el in llCounts[i]])
            iBinsThisRound = len(llCounts[i])
            mQ = np.zeros((iBinsThisRound,iTotalBins))
            for j in range(iBinsThisRound):
                for k in range(iTotalBins):
                    mQ[j,k] = bin_subset_bin(lFinalBins[k], llBins[i][j])
            #if i == iMaxRounds:
            #    lMeasurements.append((mQ, vTempCounts, np.sqrt(iC)*dSigma, (sVarName,)))
            if i == iMaxRounds:
                lMeasurements.append((mQ, vTempCounts, np.sqrt(iC)*dSigmaFinal, (sVarName,)))    
            else:
                lMeasurements.append((mQ, vTempCounts, np.sqrt(iC)*dSigma, (sVarName,)))

        return llDiscretizedTrain, lMeasurements

    def discretize_categorical_variable(self, sVarName, dMu, dMuMeasure, rng, iMinCounts = None, dQuadPropCorrBinning = 0.4):
        """
        Goal: discretize categorical variables by combining bins that have very little counts. (only if there are more than 2 bins)
        Input:
            - llCategorical             list of list with train set of every client
            - sVarName                  string with variable name
            - dMu                       double/float privacy budget
            - dMuMeasure                double total privacy budget used for measure step
            - rng                       random number generator
            - iMinCounts                integer min count to not be put in the "other" (leftover) bin
            - dQuadPropCorrBinning      double, quadratic proportion for measuring correlation bins.
        Output:
            - lFinalBins                list of the final bins
            - llCategoricalEncoded      list of list with encoded train set for every client
            - llUniqueValuesPerBin      list of list of unique values per bin (needed for transforming back)
            - lMeasurements             list of measurements that can be used for estimation
        """
        #get number of clients
        iC = self.iC

        llCategorical = [self.ldf[c][sVarName] for c in range(iC)]

        #get unique values
        lUniqueValues = self.diCatUniqueValuesInfo[sVarName]#list({el for list in llCategorical for el in list})
        #check if missing values
        bMissingValues = False
        for el in lUniqueValues:
            if isNaN(el):
                bMissingValues = True
                break
        
        if bMissingValues:
            #replace all np.nans by NaN (easier)
            lUniqueValues = [el for el in lUniqueValues if isNaN(el) == False]
            llCategorical = [[el if not isNaN(el) else "NaN" for el in lCategorical] for lCategorical in llCategorical]
            lUniqueValues.append("NaN")

        #get noise
        dMuInit = dMu*np.sqrt(1- dQuadPropCorrBinning)
        dMuCorrBinning = dMu*np.sqrt(dQuadPropCorrBinning)
        if dMu != 0.0:
            dSigma1 = (1.0/dMuInit)

            #set iMinCount
            if iMinCounts is None: 
                dTheta = 6.0
                iMinCounts = 2*dTheta*np.sqrt(2/np.pi)*(np.sqrt(self.iC)/(dMuMeasure/np.sqrt(self.iD)))

        #set counts list of list
        llCounts = []

        #set current values
        lOriginalBins = lUniqueValues.copy()
        iUniqueValues = len(lUniqueValues)
        iCurrentBins = iUniqueValues
        lCurrentCounts = iCurrentBins*[0.0]
        lEnoughObs = iCurrentBins*[True]
        lOtherBins = []
        lCombinedBins = []
        llCorrBins = []

        #if we use some privacy budget, get noisy counts per client
        if dMu != 0:
            vNoisyCountsPerClient = np.zeros(iC)
            vNoisyCountsPerClientVar = np.zeros(iC)
            #loop over clients
            for c in range(iC):
                #loop over different values
                for j in range(iCurrentBins):
                    bin = lOriginalBins[j]
                
                    #get noisy count
                    dNoisyCount = np.count_nonzero( np.array(llCategorical[c]).astype("str") == str(bin)) +  rng.normal(loc = 0, scale = dSigma1, size = 1)
                
                    #update bin and client count
                    lCurrentCounts[j] = lCurrentCounts[j] + dNoisyCount
                    vNoisyCountsPerClient[c] = vNoisyCountsPerClient[c] + dNoisyCount
                    vNoisyCountsPerClientVar[c] = vNoisyCountsPerClientVar[c] + dSigma1**2
    
            #update client counts (if applicable)
            self.update_total_counts(vNoisyCountsPerClient, vNoisyCountsPerClientVar)

        #add counts to list
        llCounts.append(lCurrentCounts)
        llCategoricalCombined = llCategorical

        if (iUniqueValues > 2) & (dMu != 0.0):
            #check bin counts
            for j in range(iCurrentBins):
                if lCurrentCounts[j] < iMinCounts:
                    lEnoughObs[j] = False
                    lOtherBins.append(lOriginalBins[j])


            #define new Bins
            lOtherIndices = []
            for j in range(iCurrentBins):
                if lEnoughObs[j]:
                    lCombinedBins.append(lOriginalBins[j])
                    llCorrBins.append([j])
                else:
                    lOtherIndices += [j]
            if len(lOtherIndices) > 0:
                llCorrBins.append(lOtherIndices)

            #add other bin
            iOtherBins = len(lOtherBins)
            if iOtherBins > 1:
                #set all values to "other"
                llCategoricalCombined = [["other" if x in lOtherBins else x for x in llCategorical[c]] for c in range(iC)]
                #add other to final bins
                lCombinedBins.append("other")
            elif iOtherBins == 1:
                lCombinedBins.append(lOtherBins[0])
        else:
            #if we do not combine, the final bins are the same as the original
            lCombinedBins = lOriginalBins.copy()
            llCorrBins = [[i] for i in range(iUniqueValues)]

        #encode
        llCategoricalEncoded = [np.array([ lOriginalBins.index(x) for x in llCategorical[c]]) for c in range(iC)]

        #################################################
        #second stage
        #################################################

        #define number of bins in the second stage and predefine the currentcounts per bin
        iCurrentBins = len(lCombinedBins)
        lCurrentCounts = iCurrentBins*[0.0]
                
        #predefine noisy counts per client
        vNoisyCountsPerClient = np.zeros(iC)
        vNoisyCountsPerClientVar = np.zeros(iC)

        #again loop over clients
        for c in range(iC):
            #get counts
            diCount = Counter(llCategoricalCombined[c])
            #loop over different bins
            for j in range(iCurrentBins): 
                if dMu != 0:     
                    dSigma2 = (1.0/dMuCorrBinning)
                    #get noisy count
                    bin = lCombinedBins[j]
                    dNoisyCount = diCount[bin] +  rng.normal(loc = 0, scale = dSigma2, size = 1)

                    #update bin and client count
                    lCurrentCounts[j] = lCurrentCounts[j] + dNoisyCount
                    vNoisyCountsPerClient[c] = vNoisyCountsPerClient[c] + dNoisyCount
                    vNoisyCountsPerClientVar[c] = vNoisyCountsPerClientVar[c] + dSigma2**2

        #update client counts (if applicable)
        if dMu != 0:
            self.update_total_counts(vNoisyCountsPerClient, vNoisyCountsPerClientVar)

        #add counts to list
        llCounts.append(lCurrentCounts)

        #get measurements
        lMeasurements = []

        #if mu = 0, we now know everything
        if dMu == 0:

            #set bins
            self.diBins[sVarName] = lOriginalBins
            self.diCorrBins[sVarName] = llCorrBins

            #return
            return llCategoricalEncoded, lMeasurements

        #if mu != 0, we have measurements we can use.

        #step one measurement
        iOriginalBins = len(lOriginalBins)
        mQ1 = np.identity(iOriginalBins)
        dSigma1 = np.sqrt(iC)*dSigma1
        vNoisyCounts1 = np.array([ llCounts[0][lOriginalBins.index(bin)][0] for bin in lOriginalBins])
        lMeasurements.append((mQ1, vNoisyCounts1, dSigma1, (sVarName,)))

        #step two measurement
        iCombinedBins = len(lCombinedBins)
        mQ2 = np.zeros((iCombinedBins,iOriginalBins))
        for i in range(iCombinedBins):
            mQ2[i, llCorrBins[i]] = 1
        vNoisyCounts2 = np.array([ llCounts[1][lCombinedBins.index(bin)][0] for bin in lCombinedBins] )
        dSigma2 = np.sqrt(iC)*dSigma2
        lMeasurements.append((mQ2, vNoisyCounts2, dSigma2, (sVarName,)))

        #set bins
        self.diBins[sVarName] = lOriginalBins
        self.diCorrBins[sVarName] = llCorrBins

        return llCategoricalEncoded,  lMeasurements

    def get_all_numeric_counts_this_round(self, llNumerical, lCurrentCounts, lCurrentBins, dSigma, rng):
        """
        GOal: get all counts for this round and update noisy total count per client
        Input: 
            - self           a federated dataset object
            - llNumerical    list of list of numerical values for each client
            - lCurrentCounts list of current counts
            - lCurrentBins   list of the bins we want to measure
            - dSigma         double sigma used for noise
            - rng            random number generator
        Set:
            - see update_total_counts
        Output: 
            -lCurrentCounts   updated counts
        """
        #set length
        iCurrentBins = len(lCurrentBins)

        #get noisy counts
        vNoisyCountsPerClient = np.zeros(self.iC)
        vNoisyCountsPerClientVar = np.zeros(self.iC)
        for c in range(self.iC):
            for i in range(iCurrentBins):

                #get current bin
                bin = lCurrentBins[i]

                #get noisy count
                dNoisyCount = FedDataset.get_noisy_counts(llNumerical[c], bin, dSigma, rng)
                        
                #update counts
                lCurrentCounts[i] += dNoisyCount
                vNoisyCountsPerClient[c] = vNoisyCountsPerClient[c] + dNoisyCount
                vNoisyCountsPerClientVar[c] = vNoisyCountsPerClientVar[c] + dSigma**2

        #use counts per client to update general count estimates
        self.update_total_counts(vNoisyCountsPerClient, vNoisyCountsPerClientVar)
        return lCurrentCounts

    def update_total_counts(self, vNoisyCountsPerClient, vNoisyCountsPerClientVar):
        """
        Goal: update the noisy count en fraction estimates for this federated datset with a new noisy count
        Input:
            - self                      A federated dataset object
            - vNoisyCountsPerClient     Vector with new noisy counts estimate
            - vNoisyCOuntsPerClientVar  Vector with variances of the noisy count estimate
        Set:
            - vNoisyCounts              estimate for the number of obervations per client
            - vNoisyCountsVar           variance of the current estimate for the noisy counts per client
            - dNoisyTotal               estimate for total number of obervations among all clients
            - vNoisyFrac                noisy fraction of observation estimates
        
        """

        
        #set new estimates
        if self.vNoisyCounts is None:
            self.vNoisyCountsVar = vNoisyCountsPerClientVar
            self.vNoisyCounts = vNoisyCountsPerClient
        else:
            vOptVar = 1.0 / (( 1.0/ self.vNoisyCountsVar) + (1.0/ vNoisyCountsPerClientVar))
            self.vNoisyCounts = vOptVar*(( self.vNoisyCounts/ self.vNoisyCountsVar) + (vNoisyCountsPerClient / vNoisyCountsPerClientVar))           
            self.vNoisyCountsVar = vOptVar
        self.dNoisyTotal = np.sum(self.vNoisyCounts)
        self.vNoisyFrac = self.vNoisyCounts / self.dNoisyTotal

    @staticmethod
    def get_noisy_counts(lNumerical, bin, dSigma, rng):
        """
        Goal: get the noisy counts of 1 bin for 1 client
        Input:
            - lNumerical: list, a list of numerical values (or NaN)
            - bin          het bin, either ["NaN"] or [min, max]
            - dSigma       double, noise to use
            - rng          random number generator
        Output:
            - dNoisyCount   double noisy count estimate
        """
        #check if it is the NaN bin
        if bin == ["NaN"]:
            #get noisy NaN count
            dNoisyCount = np.count_nonzero(np.isnan(lNumerical)) +  rng.normal(loc = 0, scale = dSigma, size = 1)  
        elif bin == ["zero"]:
            dNoisyCount = np.count_nonzero((lNumerical == 0.0)) + rng.normal(loc = 0, scale = dSigma, size = 1)
        else:
            #get bin min and max
            dTempMin = bin[0]
            dTempMax = bin[1]

            #get actual counts
            iRealCount = np.count_nonzero( (lNumerical >= dTempMin) & (lNumerical < dTempMax))

            #get noisy count
            dNoisyCount = iRealCount + rng.normal(loc = 0, scale = dSigma, size = 1)      

        return dNoisyCount

    def discretize_out_of_sample(self, dfOutOfSample, diFinalBins = None):
        """
        Goal: discretize an out of sample (test) dataset
        Input:
            - self              An federated dataset object
            - dfOutOfSample     The out of sample dataframe that needs to be discretized
            - diFinalBins       dictionary of final bins
        Return:
            - dfTestDiscrete    The discrete out of sample pandas dataframe
            - domain            The domain of the dataset
        """
        #initialize encoded dataset
        iD = self.iD
        diEncoded = dict()

        if diFinalBins is None:
            diFinalBins = self.diBins

        for d in range(iD):
            #get name
            sVarName = self.get_var_name(d)#self.ldf[0].columns[d]

            #get variable dataset
            lData =  dfOutOfSample[sVarName] 

            
            #get data type and discretize
            lDiscretizedVar = [None]
            if (self.lDataTypes[d] == "numerical") | (self.lDataTypes[d] == "ordinal"):
                lFinalSplits = [el[1] for el in diFinalBins[sVarName] if len(el) >=2]
                lDiscretizedVar = np.digitize(x = lData, bins = lFinalSplits)
            elif (self.lDataTypes[d] == "mixed"):
                lFinalSplits = [10e-8] + [el[1] for el in diFinalBins[sVarName] if len(el) >=2]
                lDiscretizedVar = np.digitize(x = lData, bins = lFinalSplits)
            else:
                lFinalBins = diFinalBins[sVarName]
                lData = [el if not isNaN(el) else "NaN" for el in lData]
                #lData = ["other" if (x not in lFinalBins and isNaN(x) == False) else x for x in lData]
                lDiscretizedVar = np.array([lFinalBins.index(x) for x in lData])

            #add discretized variables to dictionaries
            diEncoded[sVarName] = lDiscretizedVar

        #tranform dictionairy to pandas datafre
        dfDiscrete = pd.DataFrame.from_dict(diEncoded) 
        domain = self.get_domain()
        self.domain = domain
        return dfDiscrete, domain

    @staticmethod
    def transform_to_original_space(dfDiscrete, lDataTypes, diBins, diDefaultBins, delimiterInfo, rng):
        """
        Goal: transform a discrete pandas dataframe to original space
        Input: 
            - dfDiscrete    dataframe, the discrete dataframe
            - lDataTypes    list of original datatypes
            - diBins        dictionary of bins used for this discrete dataset
            - rng           random number generator
        Output:
            - dfNewData     dataframe, the dataframe transformed to the original space
        """

        #data
        
        #get number of columns
        iD = len(dfDiscrete.columns)
        iN = dfDiscrete.shape[0]

        #predefine new data dict
        diNewData = dict()

        #loop over columns
        for d in range(iD):
            #get variable name
            sVarName = dfDiscrete.columns[d]
        
            #get bins
            lBins = diBins[sVarName]
            lDefaultBins = diDefaultBins[sVarName]

            #initialize map dictionary
            diMap = dict()

            #get number of different values original and other
            iNumValues = len(lBins)
            iNumValuesDefault = len(lDefaultBins)

            #perform transformation
            if (lDataTypes[d] == "numerical"):   
                #get values
                lData = list(dfDiscrete[sVarName])
                #get in which bin
                iDelim = delimiterInfo[sVarName]
                #get in which bin
                lNewData = iN*[None]
                dCorrection2 = 10**iDelim
                dCorrection1 = 10**(-iDelim)
                for i in range(iN):
                    if diBins[sVarName][lData[i]] == ["NaN"]:
                        lNewData[i] = np.nan
                    else:
                        low = int(np.ceil((dCorrection2)*lBins[lData[i]][0]))
                        high = int(np.ceil((dCorrection2)*lBins[lData[i]][1]))
                        if low == high:
                            lNewData[i] = dCorrection1 * int(np.round((dCorrection2)*lBins[lData[i]][1]))
                        else:
                            lNewData[i] = dCorrection1 * rng.integers(low = low, high = high, size = 1)[0]
            elif (lDataTypes[d] == "ordinal"):   
                #get values
                lData = list(dfDiscrete[sVarName])
                #get in which bin
                lNewData = iN*[None]
                for i in range(iN):
                    if diBins[sVarName][lData[i]] == ["NaN"]:
                        lNewData[i] = np.nan #####
                    else:
                        lNewData[i] = rng.choice([j for j in range(int(np.ceil(diBins[sVarName][lData[i]][0])), int(np.ceil(diBins[sVarName][lData[i]][1])))])
            elif (lDataTypes[d] == "mixed"):
                #get values
                lData = list(dfDiscrete[sVarName])
                iDelim = delimiterInfo[sVarName]
                #get in which bin
                lNewData = iN*[None]
                dCorrection2 = 10**iDelim
                dCorrection1 = 10**(-iDelim)
                for i in range(iN):
                    if diBins[sVarName][lData[i]] == ["zero"]:
                        lNewData[i] = 0.0
                    elif diBins[sVarName][lData[i]] == ["NaN"]:
                        lNewData[i] = np.nan
                    else:
                        low = int(np.ceil((dCorrection2)*lBins[lData[i]][0]))
                        high = int(np.ceil((dCorrection2)*lBins[lData[i]][1]))
                        if low == high:
                            lNewData[i] = dCorrection1 * int(np.round((dCorrection2)*lBins[lData[i]][1]))
                        else:
                            lNewData[i] = dCorrection1 * rng.integers(low = low, high = high, size = 1)[0] 
            else: 
                #get index other bin if it exists
                if "other" in lBins:
                    iDefaultIndex = lBins.index("other")
                for i in range(iNumValues):
                    diMap[i] = []
                for j in range(iNumValuesDefault):
                    binDefault = lDefaultBins[j]
                    if binDefault in lBins:
                        i = lBins.index(binDefault)
                        diMap[i] += [lDefaultBins[j]]
                    else:
                        #binother has value "other" in binself
                        diMap[iDefaultIndex] += [lDefaultBins[j]]
                        
                #lNewData = [ rng.choice(diMap[x]) for x in dfDiscrete[sVarName].values]
                lNewData = [diMap[x][0] for x in dfDiscrete[sVarName].values]
                lNewData = [x if x!="NaN" else np.nan for x in lNewData]

            #set variable
            diNewData[sVarName] = lNewData

        dfNewData = pd.DataFrame.from_dict(diNewData)

        return dfNewData


    
    def get_var_name(self, d):
        """
        Goal: Get var name of a general FedDataset object
        Input: 
            - self   A federated Dataset object
            - d      column of the var name
        output:
            - sVarName  string with variable name
        """
    
        return self.ldf[0].columns[d]


    def get_domain(self, diBins = None):
        """
        Goal: Get the domain object of the data after discretization
        Input:
            - self          federated dataset object with diFInalBins attribute
        Output:
            - domain        mbi domain object.
        
        """
        if diBins is None:
            if self.diBins is None:
                raise ValueError(f"No final bins diBins, discretize before retrieving the domain.")
            else:
                diBins = self.diBins

        #get domain from final split info
        diDomain = dict()

        #get domain size per value
        for key, value in diBins.items():
            diDomain[key] = len(value)

        #create domain object
        return Domain(diDomain.keys(), diDomain.values())
