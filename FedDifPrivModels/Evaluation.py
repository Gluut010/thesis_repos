#goal: evaluation of synthetic datasets
#author: Julian van Erk
from multiprocessing.sharedctypes import Value
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from FedDifPrivModels import FedDataset
from FedDifPrivModels.Utils import powerset, isNaN
from mbi import Dataset, Domain
import pandas as pd
import numpy as np

class Evaluation():

    def __init__(self, ldfSynth, dfTrain, dfTest, lDataTypes, fedDataset= None, rng = None,
                delimiterInfo = None, iBins = 32, sSynthType = "discrete", sEvalType = "discrete"):
        """
        Goal: constructor for evaluation object
        Input:
            - ldfSynth              list of synthetic datasets (pandas dataframes) 
            - dfTrain               Pandas dataframe with train dataset used to create test dataset in original space
            - dfTest                Pandas dataframe with out-of-sample real data in original space
            - lDatatypes            List of datatypes of the variables
            - fedDataset            federated dstaset used for discretization
            - rng                   random number generator
            - delimiterInfo         info on # decimals
            - iBins                 integer number of bins for discretization if quantile discretization needed
            - sSynthType            string, synthetic data type, original or discrete.
            - sEvalType             Type of evaluation: use discrete or original space
        Set:
            - self                  a Evaluation object
        """
        #get lengths datasets
        self.iNTest = dfTest.shape[0]
        self.iNTrain = dfTrain.shape[0]
        self.lNSynth = [dfSynth.shape[0] for dfSynth in ldfSynth]
        #set typdes
        self.rng = rng
        self.lDataTypes = lDataTypes
        self.iBins = iBins
        self.iS = len(ldfSynth)
        if fedDataset is None and sEvalType == "discrete":
            raise ValueError("for discrete evaluation, give (discretized) fedDataset object")
        else:
            self.fedDataset = fedDataset
        #self.sSynthType = sSynthType
        self.sEvalType = sEvalType
        if sSynthType == "discrete" and sEvalType =="discrete":
            #synthetic data is okay
            self.ldfSynth = ldfSynth
            #train, test need discretization
            self.dfTrain, _ = fedDataset.discretize_out_of_sample(dfTrain, diFinalBins = fedDataset.diBins)
            self.dfTest, _ = fedDataset.discretize_out_of_sample(dfTest, diFinalBins = fedDataset.diBins)
        elif sSynthType == "discrete" and sEvalType == "original":
            #check if delimiter info
            if delimiterInfo is None:
                ValueError("add delimiterInfo needed to transform to original space")
            #synthetic datasets have to be transformed to original
            self.ldfSynth = [FedDataset.transform_to_original_space(dfSynth, self.lDataTypes, diBins = fedDataset.diBins, diDefaultBins = fedDataset.diBins, delimiterInfo = delimiterInfo, rng = rng) for dfSynth in ldfSynth]
            #train, test already okay
            self.dfTrain = dfTrain
            self.dfTest = dfTest
        elif sSynthType == "original" and sEvalType == "discrete":
            #synthetic data needs to be discretized
            self.ldfSynth = [fedDataset.discretize_out_of_sample(dfSynth, diFinalBins = fedDataset.diBins)[0] for dfSynth in ldfSynth]
            #train and test need to be discretized
            self.dfTrain, _ = fedDataset.discretize_out_of_sample(dfTrain, diFinalBins = fedDataset.diBins)
            self.dfTest, _ = fedDataset.discretize_out_of_sample(dfTest, diFinalBins = fedDataset.diBins)
        elif sSynthType == "original" and sEvalType == "original":
            #everything already correct
            self.dfTest = dfTest
            self.dfTrain = dfTrain
            self.ldfSynth = ldfSynth
        else:
            raise ValueError("sSynthtype or sEvalType not in [discrete, original]")


    def quantile_discretization(self):
        """
        Goal: discretize datasets using quantiles
        Input:
            - self      An evaluation object
            - ldfSynth  A list of dataframes with synthetic data
            - dfTrain   Pandas dataframe with data used to create the synthetic data
            - dfTest    Pandas dataframe with out of sample dataset
        return
            - dfTransformedTrain    pandas dataframe with ordinal encoding, quantiles as borders
            - dfTransformedTest     pandas dataframe with ordinal encoding, quantiles as border
            - ldfTransformedSynth   list of pandas dataframe with oridnal encoding, quantiles as borders
        """
        
        #get number of columns
        iD = len(self.dfTrain.columns)

        #get number of synthetic datasets
        iS = len(self.ldfSynth)

        #predefine new data dict
        diTransformedTrain = dict()
        diTransformedTest = dict()
        ldiTransformedSynth = [{} for _ in range(iS)]

        #predefine domain
        diDomain = dict()

        #loop over variables
        for d in range(iD):
            #get variable name
            sVarName = self.dfTrain.columns[d]            

            #get data
            lDataTrain = list(self.dfTrain[sVarName])
            lDataTest = list(self.dfTest[sVarName])
            llDataSynth = []
            for df in self.ldfSynth:
                llDataSynth.append(list(df[sVarName]))

            #transform
            if (self.lDataTypes[d] == "numerical") :  
                #get quantiles 
                quantiles = np.unique(np.nanquantile(np.array(lDataTrain), np.linspace(0,1,self.iBins + 1)[1:]))
                quantiles = (quantiles[1:] + quantiles[:-1]) / 2
                
                #transform
                diTransformedTrain[sVarName] = list(np.digitize(np.array(lDataTrain), bins = quantiles, right = True))
                diTransformedTest[sVarName] = list(np.digitize(np.array(lDataTest), bins = quantiles, right = True))
                for i in range(iS):
                    ldiTransformedSynth[i][sVarName] = list(np.digitize(np.array(llDataSynth[i]), bins = quantiles, right = True))
                
                #get domain
                diDomain[sVarName] = len(quantiles)
            elif (self.lDataTypes[d] == "ordinal"):
                #get unique values
                lUniqueValuesSorted = list(set(lDataTrain).union(set(lDataTest)))
                lUniqueValuesSorted.sort()
                if any([isNaN(val) for val in lUniqueValuesSorted]):
                    lUniqueValuesSorted = [el for el in lUniqueValuesSorted if not isNaN(el)]
                    lUniqueValuesSorted =  lUniqueValuesSorted + [-99999]

                #transform
                diTransformedTrain[sVarName] = [0 if isNaN(value) else lUniqueValuesSorted.index(value) for value in lDataTrain]
                diTransformedTest[sVarName] = [0 if isNaN(value) else lUniqueValuesSorted.index(value) for value in lDataTest]
                for i in range(iS):
                    ldiTransformedSynth[i][sVarName] = [0 if isNaN(value) else lUniqueValuesSorted.index(value) for value in llDataSynth[i]]
                
                #get domain
                diDomain[sVarName] = len(lUniqueValuesSorted)
            elif (self.lDataTypes[d] == "mixed"):
                #get quantiles 
                quantiles = np.unique(np.concatenate((np.array([10e-8]), np.nanquantile(np.array([el for el in lDataTrain if el > 0]), np.linspace(0,1,self.iBins + 1)[1:]))))
                quantiles = (quantiles[1:] + quantiles[:-1]) / 2

                #transform
                diTransformedTrain[sVarName] = list(np.digitize(np.array(lDataTrain), bins = quantiles, right = True))
                diTransformedTest[sVarName] = list(np.digitize(np.array(lDataTest), bins = quantiles, right = True))
                for i in range(iS):
                    ldiTransformedSynth[i][sVarName] = list(np.digitize(np.array(llDataSynth[i]), bins = quantiles, right = True))   

                #get domain
                diDomain[sVarName] = len(quantiles)
            else:
                #get unique values
                lUniqueValuesSorted = list(set(lDataTrain).union(set(lDataTest)))
                if any([isNaN(val) for val in lUniqueValuesSorted]):
                    lUniqueValuesSorted = [el for el in lUniqueValuesSorted if not isNaN(el)]
                    lUniqueValuesSorted =  lUniqueValuesSorted + [-99999]

                #transform
                diTransformedTrain[sVarName] = [0 if isNaN(value) else lUniqueValuesSorted.index(value) for value in lDataTrain]
                diTransformedTest[sVarName] = [0 if isNaN(value) else lUniqueValuesSorted.index(value) for value in lDataTest]
                for i in range(iS):
                    ldiTransformedSynth[i][sVarName] = [0 if isNaN(value) else lUniqueValuesSorted.index(value) for value in llDataSynth[i]]
                
                #get domain
                diDomain[sVarName] = len(lUniqueValuesSorted)

        #to dataframe
        dfTransformedTrain = pd.DataFrame.from_dict(diTransformedTrain)
        dfTransformedTest = pd.DataFrame.from_dict(diTransformedTest)
        ldfTransformedSynth = [None for i in range(iS)]
        for i in range(iS):
            ldfTransformedSynth[i] = pd.DataFrame.from_dict(ldiTransformedSynth[i])
        domain = Domain.fromdict(diDomain)

        #return
        return dfTransformedTrain, dfTransformedTest, ldfTransformedSynth, domain


    def marginalError(self, iMargOrder = 1, iMaxMarginals = None):
        """
        Goal: find the (nomralized) marginal error score
        Input:
            - self              An evalutation ojbect
            - iMargOrder        order of marginals (k of k-way marginal)
            - iMaxMarginals     Maximum number of marginals to evaluate
        Output
            - vMargAvgError        np array double, average marginal error score
        """

        #check if we need to discretize because we look at original space
        if self.sEvalType =="original":
            #get quantiles
            dfTransformedTrain, dfTransformedTest, ldfTransformedSynth, domain = self.quantile_discretization()
        else:
            dfTransformedTrain, dfTransformedTest, ldfTransformedSynth = self.dfTrain, self.dfTest, self.ldfSynth
            domain =  self.fedDataset.domain

        #get test and synth transform to dataset mbi object
        testDataset = Dataset(dfTransformedTest, domain)
        lSynthDataset = [Dataset(dfTransformedSynth, domain) for dfTransformedSynth in ldfTransformedSynth]

        #get marginals to measure
        lMarginals = powerset(domain, iMaxLevel=iMargOrder, iMinLevel = iMargOrder)[1:]
        if iMaxMarginals is not None:
            if self.rng is None:
                ValueError("add RNG for random selection of marginals")
            else:
                lMarginals = self.rng.choice(lMarginals, size = iMaxMarginals, replace = False)
        
        #preallocate results
        vMargAvgError = np.zeros((self.iS,))
        
        #loop over all synthetic datasets
        for i in range(self.iS):
            #set dataset
            synthDataset = lSynthDataset[i]
            #loop over marginals
            dFracObs = (self.iNTest/self.lNSynth[i])
            vIndMargError = np.array([np.linalg.norm( testDataset.project(marginal).datavector(flatten=True)- dFracObs * synthDataset.project(marginal).datavector(flatten=True), ord = 1) for marginal in lMarginals])
            #get value
            vMargAvgError[i] = np.mean(vIndMargError) / (2.0 * self.iNTest)

        return vMargAvgError

    def discriminator(self, sModel = "logistic", iInnerSplits = 3, iOuterSplits = 3, bNestedCross = False, diGrid = None, seed = 1234, diScoring = None):
        """
        Goal: get generalziation error of a discrimantor between real and fake data using (nested) CV
        input:
            - self          An evaluation object
            - sModel        Model to fit, either logistic (regression) or randomforest
            - iInnerSplits  Integer, number of fold for the inner loop (parameter selection)
            - iOuterSplits  Integer, number of folds for the outer loop (validation score)
            - bNestedCross  boolean, use nested cross validation to determine "optimal" parameters instead of standard parameters
            - diGrid        dictionary with parameter grid for model.
        Output:
            - mResults     vector with Average Accuracy  and average (negative) brier score over outer loop for all synthetic datasets
        """

        #check if we need to discretize because we look at original space
        if self.sEvalType =="original":
            #get quantiles
            dfTransformedTrain, dfTransformedTest, ldfTransformedSynth, _ = self.quantile_discretization()
        else:
            dfTransformedTrain, dfTransformedTest, ldfTransformedSynth = self.dfTrain, self.dfTest, self.ldfSynth

        #set seed
        np.random.seed(seed)

        #we need dummies for categorical variables
        iD = len(self.dfTest.columns)
        lCatColumns = self.dfTest.columns[[i for i in range(iD) if self.lDataTypes[i] == "categorical"]]

        #get scores
        if diScoring is None:
            diScoring = {'acc': 'accuracy', 'neg_brier': 'neg_brier_score'}
        iScores = len(diScoring)

        #set grid
        if diGrid is None:
            if sModel == "logistic":
                diGrid = {"Cs": list(np.exp(np.array([np.log(10**-(np.linspace(-4,4,5)[i])) for i in range(5)])))}
            elif sModel == "randomforest":
                diGrid = {'max_depth': [15],
                'min_samples_split': [10],
                "max_features": [2, 5, 10, 20],
                'n_estimators': [500]}
            else:
                raise ValueError(f"Model type {sModel} not possible")

        #predfine results
        mResults = np.zeros((self.iS,iScores))

        #define estimator
        if sModel == "logistic":
            estimator =  make_pipeline(StandardScaler(),LogisticRegression(random_state=1234, max_iter=500, solver = "saga"))
        elif sModel == "randomforest":
            estimator = RandomForestClassifier(n_estimators=500, random_state = 1234)
        else:
            raise ValueError(f"Model type {sModel} not possible")

        #loop over dataset
        for i in range(self.iS):
            #print(f"cv of dataset {i}")

            #get data
            dfSupervisedSynth = ldfTransformedSynth[i]#self.ldfSynth[i]
            dfSupervisedTest = dfTransformedTest#self.dfTest
            dfSupervisedSynth['real'] = 0
            dfSupervisedTest['real'] = 1
            dfSupervisedTotal =  pd.concat([dfSupervisedSynth, dfSupervisedTest], ignore_index=True)      

            #one hot encode categorical variables
            dfSupervisedTotal = pd.get_dummies(data=dfSupervisedTotal , columns=lCatColumns, drop_first = True)

            #define cross validations
            outerCrossVal = StratifiedKFold(n_splits = iOuterSplits, shuffle=True)

            #get inner classifiers
            if bNestedCross:
                innerCrossVal = StratifiedKFold(n_splits = iInnerSplits, shuffle=True)
                clf = GridSearchCV(estimator=estimator, param_grid=diGrid, cv=innerCrossVal)
            else:
                clf = estimator

            #get scores
            diResults = cross_validate(clf, dfSupervisedTotal.drop("real", axis = 1), dfSupervisedTotal['real'], scoring=diScoring, cv = outerCrossVal, n_jobs = 1)
            vScores = np.zeros((iScores,))
            for j in range(iScores):
                vScores[j] = np.mean(diResults[f"test_{list(diScoring)[j]}"])
            mResults[i,:] = vScores

        return mResults

    def utility(self, yVarName, diBinaryMapping = None, sModel = "logistic", iInnerSplits = 5, bParSelection = False, diGrid = None, seed = 1234):
        """
        Goal: Learn a classification model using the traindata and the synthetic datasets, and predict the test set.
        Input:
            - self          An evaluation object
            - yVarName      string, name of variable of interest (dependent variable)
            - diMapper      if y is categorical, map to binary
            - sModel        Model to fit, either logistic (regression) or randomforest
            - iInnerSplits  Integer, number of fold for the inner loop (parameter selection)
            - bParSelection boolean, use cross validation over train data to determine "optimal" parameters instead of standard parameters
            - diGrid        dictionary with parameter grid for model.
        Output
        """
        #set seed
        np.random.seed(seed)

        #we need dummies for categorical variables
        iD = len(self.dfTest.columns)
        lCatColumns = self.dfTest.columns[[i for i in range(iD) if (self.lDataTypes[i] == "categorical" and self.dfTest.columns[i] != yVarName)]]
        iScores = 6

        #set data
        ldfSynthTrainUtility = self.ldfSynth
        dfTrainUtility = self.dfTrain
        dfTestUtility = self.dfTest
        #print(dfTrainUtility)
        #map to 1 - 0. 0 is the majority class
        if self.dfTrain.dtypes[yVarName] == np.object:
            diMap = { self.dfTrain[yVarName].value_counts().idxmax(): 0,
                        self.dfTrain[yVarName].value_counts().idxmin(): 1}
            dfTestUtility[yVarName] =  dfTestUtility[yVarName].map(diMap)
            dfTrainUtility[yVarName] =  dfTrainUtility[yVarName].map(diMap)
            for dfSynthTrainUtility in ldfSynthTrainUtility:
                dfSynthTrainUtility[yVarName] = dfSynthTrainUtility[yVarName].map(diMap)

        #set grid
        if diGrid is None:
            if sModel == "logistic":
                diGrid = {"Cs": list(np.exp(np.array([np.log(10**-(np.linspace(-4,4,5)[i])) for i in range(5)])))}
            elif sModel == "randomforest":
                diGrid = {'max_depth': [15],
                'min_samples_split': [10],
                #"max_features": [2, 5, 10, 20],
                'n_estimators': [500]}
            else:
                raise ValueError(f"Model type {sModel} not possible")

        #predfine results
        mResultsSynth = np.zeros((self.iS, iScores))
        mResultsReal = np.zeros((1, iScores))

        #define estimator
        if sModel == "logistic":
            estimatorSynth =  make_pipeline(StandardScaler(),LogisticRegression(random_state=1234, max_iter=1000, solver = "saga"))
            estimatorReal =  make_pipeline(StandardScaler(),LogisticRegression(random_state=1234, max_iter=1000, solver = "saga"))
        elif sModel == "randomforest":
            estimatorSynth = RandomForestClassifier(n_estimators=250, random_state = 1234)
            estimatorReal = RandomForestClassifier(n_estimators=250, random_state = 1234)
        else:
            raise ValueError(f"Model type {sModel} not possible")

        #train evaluation
        
        #set data
        dfSynthTrainUtility = ldfSynthTrainUtility[0] 
        dfSynthTrainUtility['temp'] = 0
        dfTrainUtility['temp'] = 1
        dfTestUtility['temp'] = 2

        #create dummies for dataset
        dfTotal = pd.concat([dfSynthTrainUtility, dfTrainUtility, dfTestUtility], ignore_index= False)
        dfTotal = pd.get_dummies(data = dfTotal, columns = lCatColumns, drop_first = True)

        #split in train, synth, test
        dfSynthTrainUtility = dfTotal[dfTotal['temp']==0]
        dfTrainUtility = dfTotal[dfTotal['temp']==1]
        dfTestUtility = dfTotal[dfTotal['temp']==2]

        #drop temp variable
        dfSynthTrainUtility = dfSynthTrainUtility.drop("temp", axis = 1)
        dfTrainUtility = dfTrainUtility.drop("temp", axis = 1)
        dfTestUtility = dfTestUtility.drop("temp", axis = 1)

        #fit model 
        #return estimatorReal, yVarName, dfTrainUtility
        estimatorReal.fit(dfTrainUtility.drop(yVarName, axis = 1), dfTrainUtility[yVarName])

        #get estimates
        vYPredReal = estimatorReal.predict(dfTestUtility.drop(yVarName, axis = 1))
        vYProbPredReal = estimatorReal.predict_proba(dfTestUtility.drop(yVarName, axis = 1))[:,1]

        #get real scores
        meanPred = (dfTestUtility[yVarName].value_counts()/dfTestUtility[yVarName].size)[1]
        BSS = 1.0/(np.mean((dfTestUtility[yVarName] - meanPred)**2))
        mResultsReal[0,0] = accuracy_score(dfTestUtility[yVarName], vYPredReal)
        mResultsReal[0,1] = 1.0 - BSS * brier_score_loss(dfTestUtility[yVarName], vYProbPredReal)
        mResultsReal[0,2] = f1_score(dfTestUtility[yVarName], vYPredReal)
        mResultsReal[0,3] = precision_score(dfTestUtility[yVarName], vYPredReal)
        mResultsReal[0,4] = recall_score(dfTestUtility[yVarName], vYPredReal)
        mResultsReal[0,5] = roc_auc_score(dfTestUtility[yVarName], vYProbPredReal)
        
        #loop over dataset
        for i in range(self.iS):
            #print(f"cv of dataset {i}")

            #set data
            dfSynthTrainUtility = ldfSynthTrainUtility[i]
            dfTrainUtility = self.dfTrain
            dfTestUtility = self.dfTest
            dfSynthTrainUtility['temp'] = 0
            dfTrainUtility['temp'] = 1
            dfTestUtility['temp'] = 2

            #create dummies for dataset
            dfTotal = pd.concat([dfSynthTrainUtility, dfTrainUtility, dfTestUtility], ignore_index=True)
            dfTotal = pd.get_dummies(data = dfTotal, columns = lCatColumns, drop_first = True)

            #split in train, synth, test
            dfSynthTrainUtility = dfTotal[dfTotal['temp']==0]
            dfTrainUtility = dfTotal[dfTotal['temp']==1]
            dfTestUtility = dfTotal[dfTotal['temp']==2]

            #drop temp variable
            dfSynthTrainUtility = dfSynthTrainUtility.drop("temp", axis = 1)
            dfTrainUtility = dfTrainUtility.drop("temp", axis = 1)
            dfTestUtility = dfTestUtility.drop("temp", axis = 1)

            #fit model 
            estimatorSynth.fit(dfSynthTrainUtility.drop(yVarName, axis = 1), dfSynthTrainUtility[yVarName])

            #get estimates
            vYPredSynth = estimatorSynth.predict(dfTestUtility.drop(yVarName, axis = 1))
            vYProbPredSynth = estimatorSynth.predict_proba(dfTestUtility.drop(yVarName, axis = 1))[:,1]

            #get synthetic scores
            mResultsSynth[i,0] = accuracy_score(dfTestUtility[yVarName], vYPredSynth)
            mResultsSynth[i,1] = 1.0 - BSS * brier_score_loss(dfTestUtility[yVarName], vYProbPredSynth)
            mResultsSynth[i,2] = f1_score(dfTestUtility[yVarName], vYPredSynth)
            mResultsSynth[i,3] = precision_score(dfTestUtility[yVarName], vYPredSynth)
            mResultsSynth[i,4] = recall_score(dfTestUtility[yVarName], vYPredSynth)
            mResultsSynth[i,5] = roc_auc_score(dfTestUtility[yVarName], vYProbPredSynth)

        return mResultsSynth, mResultsReal