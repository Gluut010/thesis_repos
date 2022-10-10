#goal: define federated differentially private tabular GAN model object
#author: Adaptation of Datatransformer by zhao-zilong
from sklearn.mixture import BayesianGaussianMixture
from sklearn import preprocessing
from FedDifPrivModels import FedDataset, FedAdapIterPGM, Utils
import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F


class FedGANDataset(FedDataset):
    """
    A federated dataset object, specially designed for the training part of PGM
    """

    def __init__(self, ldf =[], lDataTypes = [], iClusters=10, dEps=0.005, diMinMaxInfo = {}, diCatUniqueValuesInfo = {},
     dMuPreprocess = 0.1, rng = None, diMixedModes = {}, bDiscretize = False, lLogColumn = [], delimiterInfo = {}):
        """
        Goal: define a federated gan dataset
        Input
            - ldf                   list of pandas dataframes with data of the clients
            - lDataTypes            list of datatypes of the variables
            - iClusters             integer, number of clusters for mode specific normalization using variational gaussian mixture model
            - dEps                  double, minimum proportion of observations that should belong to a cluster
            - diMinMaxInfo          dictionary, minimum and maximum info of this variable
            - diCatUniqueValuesInfo dictionary, unique values of al categorical variables
            - dMuPreprocess         double, privacy budget for preprocessing 
            - rng                   random number generator object 
            - diMixedModes          dictionary, modes of mixed variables (defaults to zero)
            - bDiscretize           boolean, should we use the standard discretization for mixed and numerical variables?
            - lLogColumn            list, list of columns by name that have long tails that need to be log-transformed
            - delimiterInfo         dictionary with number of decimals of original data info
        Set:
            - self
        """
        #construct super
        super().__init__(ldf = ldf, diMinMaxInfo=diMinMaxInfo, diCatUniqueValuesInfo= diCatUniqueValuesInfo, lDataTypes = lDataTypes)

        #get subclass specific attributes
        self.dMuPreprocess = dMuPreprocess
        self.delimiterInfo = delimiterInfo
        self.lLogColumn = lLogColumn
        self.diMixedModes = diMixedModes
        self.lMeta = None
        self.iC = len(ldf)
        self.iD = len(lDataTypes)
        self.lDataTypes = lDataTypes
        self.iClusters = iClusters
        self.dEps = dEps
        self.lOrdering = []
        self.lOutputInfo = []
        self.iOutputDim = 0
        self.iCondDim = 0
        self.lCondDimPerVariable = []
        self.ldfCond = [np.zeros((self.ldf[c].shape[0], self.iD)) for c in range(self.iC)]
        self.lComponents = []
        self.llModeFilterArr = []
        self.labelEncoderList = []
        self.rng = rng
        self.bDiscretized = False
        self.llOneWayMargProbs = []
        self.dtypes = self.ldf[0].dtypes

        #get meta data for (pre)processing
        self.lMeta = self.get_DP_metadata()

        #preprocess
        self.preprocess_all_client_data(bDiscretize=bDiscretize)

        #fit vgm models
        self.fit_VGM()

    def get_DP_metadata(self):
        """
        Goal: Get differentially private metadata needed.
        Input:
            -self           A FedGANDataset object
        Set:
            - indepModel    A independent DP-PGM model for DP- norm specic normalization and the 
            - fedPGMDataset The federated discretized dataset belonging to the indepModel
        Return:
            - lMeta     list of meta data 
        
        
        """
        lMeta = []

        ##treat ordinal as categorical
        #for i in range(len(self.lDataTypes)):
        #    if self.lDataTypes[i] == "ordinal":
        #        self.lDataTypes[i] = "categorical"

        #fit independent PGM model
        fedIndepPGM = FedAdapIterPGM(dMu = self.dMuPreprocess, dQuadInitBudgetProp = 1.0, rng = self.rng, bVerbose=False, sGraphType = "maxJTsize", sScoreType = "standard", bBinHierSplit = False, iMaxRoundsNumerical = 5)
        indepModel, fedPGMDataset = fedIndepPGM.fit(self.ldf, diMinMaxInfo = self.diMinMaxInfo, diCatUniqueValuesInfo = self.diCatUniqueValuesInfo, lDataTypes = self.lDataTypes)

        #set model
        self.indepModel = indepModel
        self.diBins = fedPGMDataset.diBins
        self.vNoisyFrac = fedPGMDataset.vNoisyFrac
        self.vNoisyCounts = fedPGMDataset.vNoisyCounts
        self.dNoisyTotal = fedPGMDataset.dNoisyTotal
        self.diBins = fedPGMDataset.diBins
        fedPGMDataset = None

        #set metadata
        for i in range(self.iD):
            
            sVarName = self.ldf[0].columns[i]
            if self.lDataTypes[i] == "categorical":
                diDict = {}
                index = self.diBins[sVarName]
                diDict["value"] = indepModel.project((sVarName,)).datavector()
                dfTemp = pd.DataFrame(diDict, index = index)#.sort_values(by = "value", ascending = False) -- why sorting?
                diMapper = dfTemp.index.to_list()
                lMeta.append({
                        "name": sVarName,
                        "type": "categorical",
                        "size": len(diMapper),
                        "i2s": diMapper
                        })
            elif self.lDataTypes[i] == "ordinal":
                index = [i for i in range(len(self.diBins[sVarName]))]
                diDict["value"] = indepModel.project((sVarName,)).datavector()
                dfTemp = pd.DataFrame(diDict, index = index)#.sort_values(by = "value", ascending = False) -- why sorting?
                diMapper = dfTemp.index.to_list()
                lMeta.append({
                        "name": sVarName,
                        "type": "categorical",
                        "size": len(diMapper),
                        "i2s": diMapper
                        })
            elif self.lDataTypes[i] == "mixed":
                if sVarName not in self.diMixedModes:
                #    #default mode is zero
                    self.diMixedModes[sVarName]=[0.0]
                lMeta.append({
                    "name": sVarName,
                    "type": "mixed",
                    "min": self.diMinMaxInfo[sVarName][0],
                    "max": self.diMinMaxInfo[sVarName][1],
                    "modal": self.diMixedModes[sVarName]
                })
            else:
                lMeta.append({
                    "name": sVarName,
                    "type": "continuous",
                    "min": self.diMinMaxInfo[sVarName][0],
                    "max": self.diMinMaxInfo[sVarName][1]
                })
        return lMeta

    def preprocess_all_client_data(self, bDiscretize = False, dEps = 1):
        """
        Goal: preprocess a single dataset
        Input:  
            - self         A FedGANDataset object
            - dfData       A pandas dataframe that needs to be preprocessed
            - bDiscretize  Boolean, shoul we discretize the data (treat all variables as categorical)
            - dEps         double, value for logarithmic transformation of variables in lLogColumns
        Return  
            - dfData       The preprocessed pandas dataframe
        """

        #log transformation (optional)
        iCount = 0
        for dfData in self.ldf:
            for sVarName in self.lLogColumn:
                dEps = dEps
                dLow = self.diMinMaxInfo[sVarName][0]
                if dLow >0:
                    dfData[sVarName] = dfData[sVarName].apply(lambda x: np.log(x) if x!=-9999999 else -9999999)
                else:
                    dfData[sVarName] = dfData[sVarName].apply(lambda x: np.log(x - dLow + dEps) if x!=-9999999 else -9999999)
            self.ldf[iCount] = dfData
            iCount +=1
        
        iCount = 0
        for dfData in self.ldf:
            if bDiscretize:
                #discritize
                dfData = self.discretize_out_of_sample(dfData)
                self.bDiscretized = True
            else:
                #replace empty strings
                dfData = dfData.replace(r" ", np.nan)
                dfData = dfData.fillna("empty")
                
                #deal with empty values
                lColumns = list(dfData.columns)
                for i in range(len(lColumns)):
                    column = lColumns[i]
                    if self.lDataTypes[i] == "mixed":
                        if "empty" in list(dfData[column].values):
                            #change modes
                            if (iCount == 0) and (-9999999 not in self.lMeta[i]['modal']): self.lMeta[i]['modal'].append(-9999999)

                            #change values
                            dfData[column] = dfData[column].apply(lambda x: -9999999 if x =="empty" else x)

                    elif self.lDataTypes[i] == "numerical":
                        if "empty" in list(dfData[column].values):
                            #change to mixed and add mode
                            if iCount == 0:
                                self.lMeta[i] = {
                                        "name": column,
                                        "type": "mixed",
                                        "min": self.diMinMaxInfo[column][0],
                                        "max": self.diMinMaxInfo[column][1],
                                        "modal": [-9999999]
                                    }
                            #change values
                            dfData[column] = dfData[column].apply(lambda x: -9999999 if x =="empty" else x)
            self.ldf[iCount] = dfData
            iCount += 1

        #process labelencoder
        dfTotal = pd.concat(self.ldf, ignore_index=True)
        for i, sVarName in enumerate(dfTotal.columns):
            if self.lDataTypes[i] == "categorical" or self.lDataTypes[i] == "ordinal":
                #define label encoder
                labelEncoder = preprocessing.LabelEncoder()
                dfTotal[sVarName] = dfTotal[sVarName].astype(str)
                labelEncoder.fit(dfTotal[sVarName])
                if self.lDataTypes[i] == "categorical":
                    lClasses = [int(el) if isinstance(el, float) else el for el in self.diBins[sVarName]]
                    lClasses = [str(el) if el != "NaN" else "empty" for el in lClasses]
                    labelEncoder.classes_ = np.array(lClasses)
                if self.lDataTypes[i] == "ordinal":
                    labelEncoder.classes_ = np.array([ str(int(np.ceil(el[0]))) if el[0] != "NaN" else "empty" for el in self.diBins[sVarName]]  ) 
                currentLabelEncoder = dict()
                currentLabelEncoder['column'] = sVarName
                currentLabelEncoder['labelEncoder'] = labelEncoder
                self.labelEncoderList.append(currentLabelEncoder)
            else:
                self.labelEncoderList.append(None)
        
        #fit individual dataset with this encoder
        iCount = 0
        for dfData in self.ldf:
            for i, sVarName in enumerate(dfTotal.columns): 
                if self.lDataTypes[i] == "categorical" or self.lDataTypes[i] == "ordinal":    
                    labEnc = self.labelEncoderList[i]['labelEncoder']
                    dfData[sVarName] = [int(el) if isinstance(el, float) else el for el in dfData[sVarName]]
                    dfData[sVarName] = dfData[sVarName].astype("str")
                    dfData[sVarName] = labEnc.transform(dfData[sVarName])
            self.ldf[iCount] = dfData
            iCount +=1


    def inverse_preprocess(self, ganSynthData, bDiscretized = False, dEps = 1):
        """
        Goal: inverse the preprocessing to get data in the original space
        Input:
            - self              A FedGANDataset object
            - ganSynthData      Synthetic data generated by (a version of) CTABGAN
            - bDiscretize       Boolean, shoul we discretize the data (treat all variables as categorical)
            - dEps              double, value for logarithmic transformation of variables in lLogColumns
        Output:
            - dfData            pandas dataframe in original space
        """
        #convert to pandas dataframe
        lColumnNames = []
        for id, diInfo in enumerate(self.lMeta):
            lColumnNames.append(diInfo['name'])

        dfData = pd.DataFrame(ganSynthData, columns = lColumnNames)

        #reverse label encoding
        if bDiscretized:
            dfData = FedDataset.transform_to_original_space(dfData, self.lDataTypes, diBins = self.diBins, diDefaultBins = self.diBins, delimiterInfo = self.delimiterInfo, rng = self.rng)
        else:
            for id, diInfo in enumerate(self.lMeta):
    
                if diInfo['type']  == "categorical":
                    labelEncoder = self.labelEncoderList[id]['labelEncoder']
                    labelEncoder.classes_  = np.array(labelEncoder.classes_ )
                    dfData[self.labelEncoderList[id]['column']] = dfData[self.labelEncoderList[id]['column']].astype(int)
                    dfData[self.labelEncoderList[id]['column']] = labelEncoder.inverse_transform(dfData[self.labelEncoderList[id]['column']])

        #reverse log encoding
        for i in range(len(dfData.columns)):
            sVarName = dfData.columns[i]
            if sVarName in self.lLogColumn:
                dLow = self.diMinMaxInfo[sVarName][0]
                if dLow >0:
                    dfData[sVarName] = dfData[sVarName].apply(lambda x: np.ceil(np.exp(x)-dEps) if ((x!=-9999999) & ((np.exp(x)-dEps) < 0)) else (np.exp(x)-dEps if x!=-9999999 else -9999999))
                else:
                    dfData[sVarName] = dfData[sVarName].apply(lambda x: np.exp(x)-dEps+dLow if x!=-9999999 else -9999999)
        
        #convert -9999999 to np.nan
        dfData.replace('empty', np.nan, inplace=True)
        dfData.replace(-9999999, np.nan, inplace = True)

        #round numeric columns 
        for i in range(len(dfData.columns)):
            if self.lDataTypes[i] == "numerical" or self.lDataTypes[i] == "mixed":
                sVarName = dfData.columns[i]
                iDelim = self.delimiterInfo[sVarName]
                dfData[sVarName] = np.round(dfData[sVarName], decimals = iDelim)

        #ensure correct datatypes
        dfData = dfData.astype(self.dtypes)

        return dfData

    def fit_VGM(self, iSamples = 10000):
        """
        Goal: fit a variational Gaussian mixture model on the data
        Input:
            - self          A FedGANDataset object
            - iSamples      integer, number of samples created from the independent model for fitting VGM
        Set:
            - lVgmModels    list of variational gaussian models fitted for the numerical variables 
        """

        #first, create synthetic data for the model to fit on.
        dfDiscreteIndep = self.indepModel.synthetic_data(rows = iSamples).df
        dfGaussTrain = FedDataset.transform_to_original_space(dfDiscreteIndep, self.lDataTypes, diBins = self.diBins, diDefaultBins = self.diBins, delimiterInfo = self.delimiterInfo, rng = self.rng)
        mGaussTrain = dfGaussTrain.values

        #loop over variables
        lVgmModels = []
        for id, diInfo in enumerate(self.lMeta):      
            
            if diInfo['type'] == "continuous":
                #fit vgm model
                vgmModel = BayesianGaussianMixture(n_components = self.iClusters, weight_concentration_prior_type = "dirichlet_process", weight_concentration_prior=0.001, max_iter = 10000, n_init = 1, random_state=1234, init_params = "kmeans")
                mGaussTrainTemp = np.array(mGaussTrain[:,id],float)
                vgmModel.fit(mGaussTrainTemp[np.isfinite(mGaussTrainTemp)].reshape([-1,1]))
                lVgmModels.append(vgmModel)
                #now keep only relevant modes
                lOldComp = vgmModel.weights_ > self.dEps
                vModeFreq = (pd.Series(vgmModel.predict(mGaussTrainTemp[np.isfinite(mGaussTrainTemp)].reshape(-1,1))).value_counts().keys())
                lNewComp = []
                for i in range(self.iClusters):
                    if (i in (vModeFreq)) & lOldComp[i]:
                        lNewComp.append(True)
                    else:
                        lNewComp.append(False)
                self.lComponents.append(lNewComp) 
                self.lOutputInfo += [(1, 'tanh'), (np.sum(lNewComp), 'softmax')]
                self.iOutputDim += 1 + np.sum(lNewComp)
                self.iCondDim += np.sum(lNewComp)
                self.lCondDimPerVariable.append(np.sum(lNewComp))

                #set probabilities each mode'
                self.llOneWayMargProbs.append( vgmModel.weights_[lNewComp] / np.sum(vgmModel.weights_[lNewComp]) )


            elif diInfo['type'] == "mixed":
                #fit two vgm models
                vgmModelCat =  BayesianGaussianMixture(n_components = self.iClusters, weight_concentration_prior_type = "dirichlet_process", weight_concentration_prior=0.001, max_iter = 10000, n_init = 1, random_state=1234, init_params = "kmeans") #k-means++
                vgmModelNum =  BayesianGaussianMixture(n_components = self.iClusters, weight_concentration_prior_type = "dirichlet_process", weight_concentration_prior=0.001, max_iter = 10000, n_init = 1, random_state=1234, init_params = "kmeans")
                
                #fit entire data for normalized valu of categorical mode
                mGaussTrainTemp = np.array(mGaussTrain[:,id],float)
                vgmModelCat.fit(mGaussTrainTemp[np.isfinite(mGaussTrainTemp)].reshape([-1,1]))

                #other model is for continuous component 
                lFilter = []
                for el in mGaussTrainTemp[np.isfinite(mGaussTrainTemp)]:
                    if el not in diInfo['modal']:
                        lFilter.append(True)
                    else:
                        lFilter.append(False)
                #self.lFilterArr.append(lFilter)

                #fit second model
                vgmModelNum.fit(mGaussTrainTemp[np.isfinite(mGaussTrainTemp)][lFilter].reshape([-1,1]))

                #add models
                lVgmModels.append((vgmModelCat,vgmModelNum))

                #again, keep only relevant modes for continuous data
                lOldComp = vgmModelNum.weights_ > self.dEps
                vModeFreq = (pd.Series(vgmModelNum.predict(mGaussTrainTemp[np.isfinite(mGaussTrainTemp)][lFilter].reshape([-1,1]))).value_counts().keys())
                lNewComp = []
                for i in range(self.iClusters):
                    if (i in (vModeFreq)) & lOldComp[i]:
                        lNewComp.append(True)
                    else:
                        lNewComp.append(False)
                self.lComponents.append(lNewComp) 
                self.lOutputInfo += [(1, 'tanh'), (np.sum(lNewComp) + len(diInfo['modal']), 'softmax')]
                self.iOutputDim += 1 + np.sum(lNewComp) + len(diInfo['modal'])
                self.iCondDim += np.sum(lNewComp) + len(diInfo['modal'])
                self.lCondDimPerVariable.append(np.sum(lNewComp) + len(diInfo['modal']))

                #get probability categorical nodes
                vIndepResults = self.indepModel.project((diInfo['name'],)).datavector()
                vIndepProbs  = vIndepResults / np.sum(vIndepResults)
                if -9999999 in diInfo['modal']:
                    if 0.0 in diInfo['modal']:
                        vCatProb = np.array([vIndepProbs[0],vIndepProbs[-1]])
                    else: 
                        vCatProb = np.array([vIndepProbs[-1]])
                else: 
                    vCatProb = np.array([vIndepProbs[0]])
                
                #get prob continuous part
                vContProb = 1 - np.sum(vCatProb)

                #get weights continuous nodes
                vContModeProb = vContProb * (vgmModel.weights_[lNewComp] / np.sum(vgmModel.weights_[lNewComp]))

                #set probabilisties
                vProbs = np.concatenate([vCatProb, vContModeProb])
                self.llOneWayMargProbs.append(vProbs)  
        
            else:
                # in case of categorical columns, bgm model is ignored
                lVgmModels.append(None)
                self.lComponents.append(None)
                self.lOutputInfo += [(diInfo['size'], 'softmax')]
                self.iOutputDim += diInfo['size']
                self.iCondDim += diInfo['size']
                self.lCondDimPerVariable.append(diInfo['size'])

                #set probabilities
                vIndepResults = self.indepModel.project((diInfo['name'],)).datavector()
                vIndepProbs = vIndepResults / np.sum(vIndepResults)
                self.llOneWayMargProbs.append(vIndepProbs)

        #set list of models
        self.lVgmModels = lVgmModels

    def transform_single_dataset(self, dfData, c = 0):
        """
        Goal: Transform a single dataset using the CTABGAN encoding
        Input:
            - self          A FedGANDataset object
            - dfData        pandas dataframe with data to transform
        Output:
            - mTransformed  A numpy array with the transformed dataset
        
        """
        
        #list of transformed values
        llTransformedValues = []

        #initialize counter mixed varibles
        iMixedCount = 0

        #get number of observations
        iN = dfData.shape[0]

        #to numpy 
        dfData = dfData.to_numpy() 

        #loop over columns
        for id, diInfo in enumerate(self.lMeta):
            #get data
            vData = dfData[:,id].reshape([-1,1])
                
            if (diInfo['type'] == "continuous") and (self.bDiscretized == False):
                                
                #normalize variables
                vMeans = self.lVgmModels[id].means_.reshape((1, self.iClusters)) 
                vStds = np.sqrt(self.lVgmModels[id].covariances_).reshape((1, self.iClusters))
                mFeatures = np.empty(shape=(iN, self.iClusters))
                mFeatures = (vData - vMeans) / (4.0 * vStds)

                #get number of distinct modes
                iModes = sum(self.lComponents[id])
         
                #get mode each data point
                vOptMode = np.zeros(iN, dtype = "int")    
                mProbs = self.lVgmModels[id].predict_proba(vData.reshape([-1,1]))               
                mProbs = mProbs[:, self.lComponents[id]]  
                vModes = np.arange(iModes, dtype = np.int32)
                for i in range(iN):
                    dTempProb = mProbs[i] + 1e-6
                    dTempProb = dTempProb/ np.sum(dTempProb)
                    vOptMode[i] = self.rng.choice(vModes, p = dTempProb)

                #create one hot encodings for selected modes
                mProbsOneHot = np.zeros_like(mProbs)
                idx = np.arange((iN))
                mProbsOneHot[idx, vOptMode] = 1

                # obtain normalized values and clip to -1, 1
                mFeatures = mFeatures[:, self.lComponents[id]]
                mFeatures = mFeatures[idx, vOptMode].reshape([-1,1])
                mFeatures = np.clip(mFeatures, -.99,.99) 

                #set conditional value
                self.ldfCond[c][:,id] = np.argmax(mProbsOneHot, axis = 1)#np.argmax(mReOrder, axis = 1)                 

                #store values
                llTransformedValues += [mFeatures, mProbsOneHot]# mReOrder]
          
            elif (diInfo['type'] == "mixed") and (self.bDiscretized == False):
                   
                #get means, stds of first model
                vMeans0 = self.lVgmModels[id][0].means_.reshape([-1])
                vStds0 = np.sqrt(self.lVgmModels[id][0].covariances_).reshape([-1])

                #define list for categorical components
                lZerosStdList = []
                lMeansCat = []
                lStdsCat = []

                #get mode closest to categorical components
                for dMode in diInfo['modal']:
                    #skip missing values
                    if dMode != -9999999:
                        lDist = []
                        for index, dValue in enumerate(list(vMeans0.flatten())):
                            lDist.append(abs(dMode - dValue))
                        iIndexMin = np.argmin(np.array(lDist))
                        lZerosStdList.append(iIndexMin)
                    else: continue

                #list of normalized categorical modes
                lModeValues = []
                for index in lZerosStdList:
                    lMeansCat.append(vMeans0[index])
                    lStdsCat.append(vStds0[index])

                for i,j,k in zip(diInfo['modal'], lMeansCat, lStdsCat):
                    dVal = np.clip(((i - j) / (4*k)), -.99, .99) 
                    lModeValues.append(dVal)
                    
                #add normalized value missing value
                if -9999999 in diInfo['modal']:
                    lModeValues.append(0)
                #tranform numeric comp mixed column using scond vgm
                lFilter = []
                for j in range(len(vData)):
                    if vData[j] in diInfo['modal']:
                        lFilter.append(False)
                    else:
                        lFilter.append(True)
                vData = vData[lFilter]

                #normalize variables
                vMeans = self.lVgmModels[id][1].means_.reshape((1, self.iClusters)) 
                vStds = np.sqrt(self.lVgmModels[id][1].covariances_).reshape((1, self.iClusters))
                mFeatures = np.empty(shape=(len(vData), self.iClusters))
                mFeatures = (vData - vMeans) / (4.0 * vStds)

                #get number of distinct modes
                iModes = sum(self.lComponents[id])
                #get mode each data point
                vOptMode = np.zeros(len(vData), dtype = "int")
                mProbs = self.lVgmModels[id][1].predict_proba(vData.reshape([-1,1]))
                mProbs = mProbs[:, self.lComponents[id]]
                for i in range(len(vData)):
                    dTempProb = mProbs[i] + 1e-6
                    dTempProb = dTempProb/ np.sum(dTempProb)
                    vOptMode[i] = self.rng.choice(np.arange(iModes), p = dTempProb)
                    
                #create one hot encodings for selected modes
                mProbsOneHot = np.zeros_like(mProbs)
                mProbsOneHot[np.arange(len(mProbs)), vOptMode] = 1

                # obtain normalized values and clip to -1, 1
                idx = np.arange((mFeatures.shape[0]))
                mFeatures = mFeatures[:, self.lComponents[id]]
                mFeatures = mFeatures[idx, vOptMode].reshape([-1,1])
                mFeatures = np.clip(mFeatures, -.99,.99)

                #additonal modes categorical comp
                mExtraBits = np.zeros((len(vData), len(diInfo['modal'])))
                mTempProbsOneHot = np.concatenate([mExtraBits, mProbsOneHot], axis = 1)

                #stor final normalized value
                mFinal = np.zeros([iN, 1 + mProbsOneHot.shape[1] + len(diInfo['modal'])])

                #iterate of continuous component
                iFeaturesCurser = 0

                for idx, value in enumerate(dfData[:,id]):

                    if value in diInfo['modal']:
                        #deal with categorical component mode
                        lCategory_ = list(map(diInfo['modal'].index, [value]))[0]
                        mFinal[idx,0] = lModeValues[lCategory_]
                        mFinal[idx, (lCategory_ + 1)] = 1
                    else:
                        #continious modes
                        mFinal[idx,0] = mFeatures[iFeaturesCurser]
                        mFinal[idx, (1+len(diInfo['modal'])):] = mTempProbsOneHot[iFeaturesCurser][len(diInfo['modal']):]
                        iFeaturesCurser = iFeaturesCurser + 1

                ##reorder
                mJustOneHot = mFinal[:,1:]         
                mFinalFeatures = mFinal[:,0].reshape([-1,1])

                #set conditional value
                self.ldfCond[c][:,id] = np.argmax(mJustOneHot, axis = 1)#np.argmax(mReOrder, axis = 1)
                  
                #update values
                llTransformedValues += [mFinalFeatures, mJustOneHot]# mReOrderedJustOneHot]
                    
                #update counter
                iMixedCount = iMixedCount + 1
            else:
                #categorical, standard one hot encoding
                if c == 0: self.lOrdering.append(None)
                mColT = np.zeros([iN, diInfo['size']])
                #idx = list(map(diInfo['i2s'].index, vData.reshape([-1,1]))) #?????
                idx = vData.reshape([1,-1]).astype(np.int32)           
                mColT[np.arange(iN), idx] = 1
                #set conditional value
                self.ldfCond[c][:,id] = np.argmax(mColT, axis = 1) #idx?
                #get transofmered values
                llTransformedValues.append(mColT)
  

        #return everything concatenated
        dfTransformed = np.concatenate(llTransformedValues, axis = 1)
        return dfTransformed

    def transform_all_client_data(self):
        """
        Goal: Transform all datasets using the CTABGAN encoding
        Input:
            - self          A FedGANDataset object
        Output:
            - dfTransformed A pandas dataframe with the transformed dataset
        
        """
        for c in range(self.iC):
            dfData = self.ldf[c]      
            self.ldf[c] = self.transform_single_dataset(dfData, c = c)


    def inverse_transform(self, dfData):
        """
        Goal: Inverse transform a dataset from the CTABGAN encoding to original encoding
        Input: 
            - self          A FedGANDataset object
            - dfData        A pandas dataframe with the data that needs to be transformed
        Output:
            - dfTransformedBack  A pandas dataframe with the transformed-back dataset
        """
        #store in dfTransformedBack dataframe
        iN = len(dfData)
        dfTransformedBack = np.zeros([iN, self.iD])

        #iterator
        iSt = 0

        #loop over original columns
        for id, diInfo in enumerate(self.lMeta):
            if diInfo["type"] == "continuous":

                #get normalized values
                vU = dfData[:,iSt]
                vU = np.clip(vU, -1, 1)

                #get one hot encoding modes
                mV = dfData[:, iSt+1:iSt +1 +np.sum(self.lComponents[id])]

                #set unused mode to small value to ignore
                mVt = np.ones((iN, self.iClusters)) *-100
                mVt[:, self.lComponents[id]] = mV
                mV = mVt

                #obtain appropriate means and standard deviatons
                vMeans = self.lVgmModels[id].means_.reshape([-1])
                vStds = np.sqrt(self.lVgmModels[id].covariances_).reshape([-1])
                vPargmax = np.argmax(mV, axis = 1)
                vStdst = vStds[vPargmax]
                vMeant = vMeans[vPargmax]

                #inverse transformation
                vTemp = vU * 4 * vStdst + vMeant

                #set data
                dfTransformedBack[:,id] = vTemp

                #move to next columns
                iSt += 1 + np.sum(self.lComponents[id])

            elif diInfo['type'] == "mixed":
                #get normalized values with modes
                vU = dfData[:, iSt]
                vU = np.clip(vU, -1, 1)
                mFullV = dfData[:, (iSt+1):(iSt+1)+len(diInfo["modal"])+np.sum(self.lComponents[id])]

                #get modes catgorical compontnet
                mMixedV = mFullV[:,:len(diInfo['modal'])]

                #get modes continuous components
                mV = mFullV[:, -np.sum(self.lComponents[id]):]

                # set unused modes to -100
                mVt = np.ones((iN, self.iClusters))*-100
                mVt[:, self.lComponents[id]] = mV
                mV = np.concatenate([mMixedV, mVt], axis = 1)
                vPargmax = np.argmax(mV, axis = 1)

                #get means, stds of the continuous components
                vMeans = self.lVgmModels[id][1].means_.reshape([-1])
                vStds = np.sqrt(self.lVgmModels[id][1].covariances_).reshape([-1])

                #predefine inverse transform data
                vResults = np.zeros_like(vU)

                for idx in range(iN):
                    #check if categorical mode selected, and assign this value
                    if vPargmax[idx] < len(diInfo['modal']):
                        dMax = vPargmax[idx]
                        vResults[idx] = float(list(map(diInfo['modal'].__getitem__, [dMax]))[0])
                    else:
                        #continuous mode
                        dStdst = vStds[(vPargmax[idx] - len(diInfo['modal']))]
                        dMeant = vMeans[(vPargmax[idx] - len(diInfo['modal']))]
                        vResults[idx] = vU[idx] *4 * dStdst + dMeant
                
                #set transformed data
                dfTransformedBack[:, id] = vResults

                #set counter
                iSt += 1 + np.sum(self.lComponents[id]) + len(diInfo["modal"])
            else:
                #use reversed one hot encoding
                dfTemp = dfData[:, iSt:iSt + diInfo["size"]]
                idx = np.argmax(dfTemp, axis = 1)
                dfTransformedBack[:, id] = idx#list(map(diInfo['i2s'].__getitem__,idx))
                iSt += diInfo['size']

        return dfTransformedBack

    def apply_activate(self, dfData):
        """
        Goal: This function applies the final activation corresponding to the column information associated with transformer
        Inputs:
            - data -> input data generated by the model in the same format as the transformed input data
            - output_info -> column information associated with the transformed input data
        Outputs:
            - act_data -> resulting data after applying the respective activations 
        """
        
        data_t = []
        # used to iterate through columns
        st = 0
        # used to iterate through column information
        for item in self.lOutputInfo:
            # for numeric columns a final tanh activation is applied
            if item[1] == 'tanh':
                ed = st + item[0]
                data_t.append(torch.tanh(dfData[:, st:ed]))
                st = ed
            # for one-hot-encoded columns, a final gumbel softmax (https://arxiv.org/pdf/1611.01144.pdf) is used 
            # to sample discrete categories while still allowing for back propagation 
            elif item[1] == 'softmax':
                ed = st + item[0]
                # note that as tau approaches 0, a completely discrete one-hot-vector is obtained
                data_t.append(F.gumbel_softmax(dfData[:, st:ed], tau=0.2))
                st = ed
        
        act_data = torch.cat(data_t, dim=1) 

        return act_data


class ImageTransformer():

    """
    Transformer responsible for translating data rows to images and vice versa
    Variables:
    1) side -> height/width of the image
    Methods:
    1) __init__() -> initializes image transformer object with given input
    2) transform() -> converts tabular data records into square image format
    3) inverse_transform() -> converts square images into tabular format
    """
    
    def __init__(self, side):
    
        self.height = side
            
    def transform(self, dfData):
        
        if self.height * self.height > len(dfData[0]):
            # tabular data records are padded with 0 to conform to square shaped images
            padding = torch.zeros((len(dfData), self.height * self.height - len(dfData[0]))).to(dfData.device)
            dfData = torch.cat([dfData, padding], axis=1)

        return dfData.view(-1, 1, self.height, self.height)

    def inverse_transform(self, dfData):
        
        dfData = dfData.view(-1, self.height * self.height)

        return dfData
