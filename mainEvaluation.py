def warn(*args, **kwargs):
    pass
import warnings
from FedDifPrivModels.FedDPCTGAN import FedDPCTGAN
warnings.warn = warn
from FedDifPrivModels.FedAllInOnePGM import FedAllInOnePGM
from FedDifPrivModels.FedAdapIterPGM import FedAdapIterPGM
from FedDifPrivModels.FedDifPrivPGM import FedDifPrivPGM
from FedDifPrivModels.FedDPCTGAN import FedDPCTGAN
from FedDifPrivModels.FedDataset import FedDataset
from FedDifPrivModels.Evaluation import Evaluation
from FedDifPrivModels.Utils import FL_split_data, get_FL_datasets, map_to_other_discritization
from mbi import domain, Dataset, GraphicalModel
import multiprocessing as mp
import numpy as np
import pandas as pd
import pickle as pk
import itertools as iter
import tqdm
import time


def get_eval_object(tInfo):
    """
    Goal: helper function to get correct evaluation object
    Input:
        - tInfo,     see first line
    Output:
        - evalObject Evaluation object
    """

    #retrieve information
    model = tInfo[0]  
    fedDataset = tInfo[1] 
    dfTotalTrainData = tInfo[2]
    dfTotalTestData = tInfo[3]
    iDatasetsPerModel = tInfo[4]
    delimiterInfo = tInfo[5]
    i = tInfo[7]
    dMu = tInfo[8]

    #generate datasets
    ldfSynthData = [None for _ in range(iDatasetsPerModel)]
    for j in range(iDatasetsPerModel):

        #generate dataset
        if isinstance(model, FedDPCTGAN):
            #data already transformed to original space
            ldfSynthData[j] = model.synthetic_data(iRows = dfTotalTestData.shape[0], device = "cpu")           
        elif isinstance(model, GraphicalModel):
            ldfSynthData[j] = model.synthetic_data(rows = dfTotalTestData.shape[0]).df
            #transform dataset to original space
            ldfSynthData[j] = FedDataset.transform_to_original_space(ldfSynthData[j], fedDataset.lDataTypes, diBins = fedDataset.diBins, diDefaultBins = fedDataset.diBins, delimiterInfo = delimiterInfo, rng = np.random.default_rng(12344))
    if model is None:
        #training data evaluation
        ldfSynthData = [dfTotalTrainData]

        

    #get eval object
    evalObject = Evaluation(ldfSynthData, dfTotalTrainData, dfTotalTestData, lDataTypes=fedDataset.lDataTypes, fedDataset=fedDataset, sSynthType = "original", sEvalType = "original")
    return evalObject


def get_marginal_errors1(tInfo):
    """
    Goal: Get the first order marginal errors for a single model
    Input: 
        - tInfo,    see first line
    Output
        - result    a lost of two elements, a tuple with dataset i, privacy budget mu, and as second element the marginal error.

    """
    #get eval object
    evalObject = get_eval_object(tInfo)
    i = tInfo[7]
    dMu = tInfo[8]

    #get results
    vRes = evalObject.marginalError(iMargOrder=1)
    dValue = np.mean(vRes)
    if dValue == 0.0:
        dValue = np.nan

    #return value
    return [(i, dMu), dValue]

def get_marginal_errors2(tInfo):
    """
    Goal: Get the second order marginal errors for a single model
    Input: 
        - tInfo,    see first line
    Output
        - result    a lost of two elements, a tuple with dataset i, privacy budget mu, and as second element the marginal error.

    """
    #get eval object
    evalObject = get_eval_object(tInfo)
    i = tInfo[7]
    dMu = tInfo[8]

    #get results
    vRes = evalObject.marginalError(iMargOrder=2)
    dValue = np.mean(vRes)
    if dValue == 0.0:
        dValue = np.nan

    #return value
    return [(i, dMu), dValue]

def get_marginal_errors3(tInfo):
    """
    Goal: Get the third order marginal errors for a single model
    Input: 
        - tInfo,    see first line
    Output
        - result    a lost of two elements, a tuple with dataset i, privacy budget mu, and as second element the marginal error.

    """
    #get eval object
    evalObject = get_eval_object(tInfo)
    i = tInfo[7]
    dMu = tInfo[8]

    #get results
    vRes = evalObject.marginalError(iMargOrder=3)
    dValue = np.mean(vRes)
    if dValue == 0.0:
        dValue = np.nan

    #return value
    return [(i, dMu), dValue]


def get_discriminator_LR_scores(tInfo):
    """
    Goal: Get the brier skill scores for the LR discriminator
    Input: 
        - tInfo,    see first line
    Output
        - result    a lost of two elements, a tuple with dataset i, privacy budget mu, and as second element the marginal error.
    """

    #get eval object
    evalObject = get_eval_object(tInfo)
    i = tInfo[7]
    dMu = tInfo[8]

    #get results
    mRes = evalObject.discriminator(sModel = "logistic")
    vResAcc = mRes[:,0]
    vResBrier = 1.0 + 4.0 * mRes[:,1]

    #get average results
    dMeanAcc = np.mean(vResAcc)
    dMeanBrier = np.mean(vResBrier)
    if dMeanBrier == 0.0:
        dMeanBrier = np.nan
    if dMeanAcc == 0.0:
        dMeanAcc = np.nan

    #return value
    return [(i, dMu), dMeanBrier]

def get_discriminator_RF_scores(tInfo):
    """
    Goal: Get the brier skill scores for the RF discriminator
    Input: 
        - tInfo,    see first line
    Output
        - result    a lost of two elements, a tuple with dataset i, privacy budget mu, and as second element the marginal error.
    """

    #get eval object
    evalObject = get_eval_object(tInfo)
    i = tInfo[7]
    dMu = tInfo[8]

    #get results
    mRes = evalObject.discriminator(sModel = "randomforest")
    vResAcc = mRes[:,0]
    vResBrier = 1.0 + 4.0 * mRes[:,1]

    #get average results
    dMeanAcc = np.mean(vResAcc)
    dMeanBrier = np.mean(vResBrier)
    if dMeanBrier == 0.0:
        dMeanBrier = np.nan
    if dMeanAcc == 0.0:
        dMeanAcc = np.nan

    #return value
    return [(i, dMu), dMeanBrier]

def get_utility_LR_scores(tInfo):
    """
    Goal: Get the utility scores for the RF discriminator
    Input: 
        - tInfo,    see first line
    Output
        - result    a lost of two elements, a tuple with dataset i, privacy budget mu, and as second element the marginal error.
    """

    #get eval object
    evalObject = get_eval_object(tInfo)
    sVarName = tInfo[6]
    i = tInfo[7]
    dMu = tInfo[8]

    #get results
    tUtil = evalObject.utility(sModel = "logistic", yVarName = sVarName)
    mRes = tUtil[0]
    dMeanBSS = np.mean(mRes[:,1])
    dMeanBSSTrain = np.mean(tUtil[1][:,1])
    return [(i, dMu), dMeanBSS, dMeanBSSTrain]

def get_utility_RF_scores(tInfo):
    """
    Goal: Get the brier skill scores for the RF discriminator
    Input: 
        - tInfo,    see first line
    Output
        - result    a lost of two elements, a tuple with dataset i, privacy budget mu, and as second element the marginal error.
    """

    #get eval object
    evalObject = get_eval_object(tInfo)
    sVarName = tInfo[6]
    i = tInfo[7]
    dMu = tInfo[8]

    #get results
    tUtil = evalObject.utility(sModel = "logistic", yVarName = sVarName)
    mRes = tUtil[0]
    dMeanBSS = np.mean(mRes[:,1])
    dMeanBSSTrain = np.mean(tUtil[1][:,1])
    return [(i, dMu), dMeanBSS, dMeanBSSTrain]

def main():
    #set seed
    iSeed = 1234

    #path
    #desktop: "D:/master_thesis/thesis_repos"
    #laptop: "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian"
    computerPath = "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian" #laptop: D:/master_thesis/thesis_repos

    #set model settings

    #llModelSettings = [
        #["PGM","FIST","noCorrBins","defaultScore","maxDegree3"],
        #["GAN","noMode","NA","noScaffold","NA"],
        #["GAN","withMode","NA","noScaffold","NA"],
        #["PGM", "Independent", "noCorrBins", "defaultScore", "maxDegree2"],
        #["PGM", "FTST", "noCorrBins", "defaultScore", "maxDegree2"],
        #["PGM","FTST","noCorrBins","randomScore","maxDegree2"],
        #["PGM","FTST","corrBins","defaultScore","maxDegree2"],
        #["PGM","FIST","noCorrBins","randomScore","maxDegree2"],
        #["PGM","FIST","corrBins","defaultScore","maxDegree3"],   
        #["TrainData","NA","NA","NA","NA"]
    #    ]   

    
    llModelSettings = [
        ["GAN","withMode","NA","noScaffold","NA"],
        #["GAN","withMode","NA","withScaffold","NA"],
        ["PGM", "Independent", "noCorrBins", "defaultScore", "maxDegree2"],
        ["PGM", "FTST", "noCorrBins", "defaultScore", "maxDegree2"],
        #["PGM", "FTST", "noCorrBins", "adjustedScore", "maxDegree2"],
        #["PGM", "FTST", "noCorrBins", "localModel", "maxDegree2"],
        #["PGM","FIST","noCorrBins","defaultScore","maxDegree3"],
        #["PGM","FIST","noCorrBins","adjustedScore","maxDegree3"],
        ["TrainData","NA","NA","NA","NA"]
        ] 
         
    """
    llModelSettings = [
        #["GAN","noMode","NA","noScaffold","NA"],
        #["GAN","withMode","NA","noScaffold","NA"],
        #["PGM", "Independent", "noCorrBins", "defaultScore", "maxDegree2"],
        #["PGM","FTST", "noCorrBins", "defaultScore", "maxDegree2"],
        #["PGM","FTST","noCorrBins","randomScore","maxDegree2"],
        #["PGM","FTST","corrBins","defaultScore","maxDegree2"],
        #["TrainData","NA","NA","NA","NA"]
        ] 
    """
    """
    llModelSettings = [
        ["GAN","withMode","NA","noScaffold","NA"],
        ["GAN","withMode","NA","withScaffold","NA"],
        ["PGM", "Independent", "noCorrBins", "defaultScore", "maxDegree2"],
        ["PGM", "FTST", "noCorrBins", "defaultScore", "maxDegree2"],
        ["PGM", "FTST", "noCorrBins", "adjustedScore", "maxDegree2"],
        ["PGM", "FTST", "noCorrBins", "localModel", "maxDegree2"],
        #["PGM","FIST","noCorrBins","defaultScore","maxDegree3"],
        #["PGM","FIST","noCorrBins","adjustedScore","maxDegree3"],
        ["TrainData","NA","NA","NA","NA"]
        ]   
    """

    #set evaluation measures
    lEvalTypes = ['utility_RF','utility_LR']#['marginals2', "marginals3", "discriminator_LR", "discriminator_RF"]#['marginals1', 'marginals2', "marginals3"] ['utility_LR','utility_RF']#

    #set datasplit setting
    sSetting = "homogeneous_equalsize" #homogeneous_equalsize, heterogeneous_equalsize, homogeneous_diffsize 
    #sDataset = "adult"
    lsDataset = ["loan"]

    for sDataset in lsDataset:

        #load data
        sRef =  f"{computerPath}/pickle_data/{sDataset}.pickle"
        iRep = 3#5,3
        lMu = [0.01,0.025,0.05,0.10,0.25,0.50,1.00,2.50]
        llClientTrainDataRaw, llClientTestDataRaw, lTotalTrainData, lTotalTestData,  diMapper, lDataTypes, sYname, diMinMaxInfo, delimiterInfo, diCatUniqueValuesInfo = get_FL_datasets(sRef=sRef, sSetting = sSetting, dTrainFrac = 0.75, iRep = iRep, iSeed = iSeed)
     
        #load fedDatasets
        with open(f'{computerPath}/models/lFedDatasets_{sSetting}_{sDataset}.pickle', 'rb') as handle:
            lFedDatasets = pk.load(handle)


        #start loop over models
        for lModelSetting in llModelSettings:
            #set settings
            sModelType = lModelSetting[0]
            sModelVersion = lModelSetting[1]
            sCorrBins = lModelSetting[2]
            sScoreType = lModelSetting[3]
            sMaxDegree = lModelSetting[4]

            #loop overevaluation settings
            for sEvalType in lEvalTypes:
                print(f"Starting {sEvalType} for {lModelSetting}")

                #load models
                lInputs = []
                if sModelType != "TrainData":
                    sRef = f"{computerPath}/models/{sSetting}_{sModelType}_{sModelVersion}_{sCorrBins}_{sScoreType}_{sMaxDegree}_{sDataset}.pickle"
                    with open(sRef, 'rb') as handle:
                        diModels = pk.load(handle)
            
                    #set seed for synthetic data generation
                    np.random.seed(iSeed)

                    #set number of generated datasets per model
                    iDatasetsPerModel = 3#5#10

                    for key, value in diModels.items():
                        model = value
                        i = key[0]
                        if i > 2:
                           continue
                        dMu = key[1]
                        if sModelType == "PGM":
                            fedDataset = lFedDatasets[i]
                        elif sModelType == "GAN":
                            model.fedGANDataset.dtypes = lTotalTrainData[0].dtypes
                            fedDataset = model.fedGANDataset

                        dfTotalTrainData = lTotalTrainData[i]
                        dfTotalTestData = lTotalTestData[i]
                        lInputs.append((model, fedDataset, dfTotalTrainData, dfTotalTestData, iDatasetsPerModel, delimiterInfo, sYname, i, dMu))
                else:
                    for i in range(iRep):
                        for j in lMu:
                            lInputs.append((None, lFedDatasets[i], lTotalTrainData[i], lTotalTestData[i], 0, delimiterInfo, sYname, i, j))

                iComputations = len(lInputs)

                #multiprocessing
                t1 = time.time()

                #check evaluation type
                if sEvalType == "marginals1":
                    with mp.Pool(processes = 6) as pl:
                        lResults = list(tqdm.tqdm(pl.imap_unordered(get_marginal_errors1, lInputs), total = iComputations))
                        pl.close()
                        pl.join()
                elif sEvalType == "marginals2":
                    with mp.Pool(processes = 6) as pl:
                        lResults = list(tqdm.tqdm(pl.imap_unordered(get_marginal_errors2, lInputs), total = iComputations))
                        pl.close()
                        pl.join()
                elif sEvalType == "marginals3":
                    with mp.Pool(processes = 6) as pl:
                        lResults = list(tqdm.tqdm(pl.imap_unordered(get_marginal_errors3, lInputs), total = iComputations))
                        pl.close()
                        pl.join()
                elif sEvalType == "discriminator_LR":
                    with mp.Pool(processes = 6) as pl:
                        lResults = list(tqdm.tqdm(pl.imap_unordered(get_discriminator_LR_scores, lInputs), total = iComputations))
                        pl.close()
                        pl.join()
                elif sEvalType == "discriminator_RF":
                    with mp.Pool(processes = 6) as pl:
                        lResults = list(tqdm.tqdm(pl.imap_unordered(get_discriminator_RF_scores, lInputs), total = iComputations))
                        pl.close()
                        pl.join()
                elif sEvalType == "utility_LR":
                    with mp.Pool(processes = 3) as pl:
                        lResults = list(tqdm.tqdm(pl.imap_unordered(get_utility_LR_scores, lInputs), total = iComputations))
                        pl.close()
                        pl.join()
                elif sEvalType == "utility_RF":
                    with mp.Pool(processes = 3) as pl:
                        lResults = list(tqdm.tqdm(pl.imap_unordered(get_utility_RF_scores, lInputs), total = iComputations))
                        pl.close()
                        pl.join()
                else:
                    print("supply a correct evaluation type ")
                    return None
                    
                t2 = time.time()
                print(f"Took {round(t2-t1,3)} seconds")
                
                mAggResults = np.zeros((len(lMu),4))
                mAggResults[:] = np.nan
                for i in range(len(lMu)):
                    #get mu
                    dMu = lMu[i]
                    #get correct values
                    vTempResults = np.zeros((iRep,))
                    for el in lResults:
                        if el[0][1] == dMu:
                            vTempResults[el[0][0]] = el[1]
                    #get mean, median, min, max
                    mAggResults[i,0] = np.nanmean(vTempResults)   #mean
                    mAggResults[i,1] = np.nanmedian(vTempResults) #medain
                    mAggResults[i,2] = np.nanmin(vTempResults)    #min
                    mAggResults[i,3] = np.nanmax(vTempResults)    #max

                print(mAggResults)

                #save results
                path = f"{computerPath}/results/{sEvalType}_{sSetting}_{sModelType}_{sModelVersion}_{sCorrBins}_{sScoreType}_{sMaxDegree}_{sDataset}_adj.pickle"
                with open(path, 'wb') as handle:
                    pk.dump(lResults, handle)

                #save aggregated results
                path = f"{computerPath}/results_agg/{sEvalType}_{sSetting}_{sModelType}_{sModelVersion}_{sCorrBins}_{sScoreType}_{sMaxDegree}_{sDataset}_adj.pickle"
                with open(path, 'wb') as handle:
                    pk.dump(mAggResults, handle)




if __name__ == '__main__':
    main()