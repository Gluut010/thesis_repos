
from FedDifPrivModels.FedAllInOnePGM import FedAllInOnePGM
from FedDifPrivModels.FedAdapIterPGM import FedAdapIterPGM
from FedDifPrivModels.FedDPCTGAN import FedDPCTGAN
from FedDifPrivModels.Utils import FL_split_data, get_FL_datasets, map_to_other_discritization
from mbi import domain, Dataset
import numpy as np
import pandas as pd
import pickle as pk
import multiprocessing as mp
import itertools as iter
import time
import tqdm
import torch

def get_fedDatasets_objects(ltData):
    """
    Goal: Get the federated dataset objects for all datasets.
    Input:
        - ltData      list with tuple of dataset information
    
    """

    #retrieve inputs
    tDataInfo = ltData
    #llClientTrainDataRaw = tDataInfo[0]
    #diMinMaxInfo = tDataInfo[1]
    #diCatUniqueValuesInfo = tDataInfo[2]
    #lDataTypes = tDataInfo[3]
    #i = tDataInfo[4]

    #get fedDataset
    fedDPPGM = FedAdapIterPGM(dMu = 0.1, dQuadInitBudgetProp = 1.0, iSeed = 1234)
    lFedDatasets = [None for _ in range(len(ltData))]
    for j in range(len(ltData)):
        print(f"getting fedDataset {tDataInfo[j][5]}")
        lFedDatasets[tDataInfo[j][5]] =  fedDPPGM.fit( tDataInfo[j][0], diMinMaxInfo = tDataInfo[j][1], diCatUniqueValuesInfo = tDataInfo[j][2], lDataTypes = tDataInfo[j][3])[1]
    return lFedDatasets

def run_parallel_PGM(tInputs):
    """
    Goal: function which allows PGM models to be run in parallel
    Input:
        - tInputs       tuple of inputs (see first lines of code for details)
    Output
        - model         Fitted FedDPPGM model (without the federated dataset)
    
    """

    #retrieve inputs
    sModelVersion = tInputs[0]
    sCorrBins = tInputs[1]
    sScoreType= tInputs[2]
    sMaxDegree = tInputs[3]
    dMu = tInputs[4]
    tDataInfo = tInputs[5]
    lClientTrainDataRaw = tDataInfo[0]
    diMinMaxInfo = tDataInfo[1]
    diCatUniqueValuesInfo = tDataInfo[2]
    lDataTypes = tDataInfo[3]
    i = tDataInfo[5]

    #set function parameters
    if sCorrBins == "noCorrBins":
        bCombineLevels = False
    elif sCorrBins == "corrBins":
        bCombineLevels = True
    if sMaxDegree == "maxDegree2":
        iMaxDegree = 2
    elif sMaxDegree == "maxDegree3":
        iMaxDegree = 3

    # Get correct funciton 
    if sModelVersion == "FTST" or sModelVersion == "FTSTSingleClient":
        if sScoreType == "defaultScore":
            fedDPPGM = FedAllInOnePGM(dMu = dMu, bVerbose=False, sGraphType = "tree", sScoreType = "standard", bCombineLevels = bCombineLevels, iSeed = 1234)
        elif sScoreType == "adjustedScore":
            fedDPPGM = FedAllInOnePGM(dMu = dMu, bVerbose=False, sGraphType = "tree", sScoreType = "adjusted", bCombineLevels = bCombineLevels, iSeed = 1234)
        elif sScoreType == "localModel":
            fedDPPGM = FedAllInOnePGM(dMu = dMu, bVerbose=False, sGraphType = "tree", sScoreType = "standard", bLocalModel=True, bCombineLevels = bCombineLevels, iSeed = 1234)
        elif sScoreType == "randomScore":
            fedDPPGM = FedAllInOnePGM(dMu = dMu, bVerbose=False, sGraphType = "tree", sScoreType = "random", bCombineLevels = bCombineLevels, iSeed = 1234)
    elif sModelVersion == "FIST" or sModelVersion == "FISTSingleClient":
        if sScoreType == "defaultScore":
            fedDPPGM = FedAdapIterPGM(dMu = dMu, iMaxDegree=iMaxDegree, bVerbose=False, sGraphType = "maxJTsize", sScoreType = "standard", dMaxJTsize = 5, bCombineLevels = bCombineLevels, iSeed = 1234)
        elif sScoreType == "adjustedScore":
            fedDPPGM = FedAdapIterPGM(dMu = dMu, iMaxDegree=iMaxDegree, bVerbose=False, sGraphType = "maxJTsize", sScoreType = "adjusted", dMaxJTsize = 5, bCombineLevels = bCombineLevels, iSeed = 1234)
        elif sScoreType == "randomScore":
            fedDPPGM = FedAdapIterPGM(dMu = dMu, iMaxDegree=iMaxDegree, bVerbose=False, sGraphType = "maxJTsize", sScoreType = "random", dMaxJTsize = 5, bCombineLevels = bCombineLevels, iSeed = 1234)
    elif sModelVersion == "Independent":
        fedDPPGM = FedAdapIterPGM(dMu = dMu, dQuadInitBudgetProp = 1.0, iSeed = 1234, iMaxDegree=2)

        
    return [(i,dMu), fedDPPGM.fit(lClientTrainDataRaw, diMinMaxInfo = diMinMaxInfo, diCatUniqueValuesInfo = diCatUniqueValuesInfo, lDataTypes = lDataTypes)[0]]


def run_GAN(tInputs):
    """
    Goal: function which allows PGM models to be run in parallel
    Input:
        - tInputs       tuple of inputs (see first lines of code for details)
    Output
        - model         Fitted FedDPCTGAN model (without the federated dataset)
    
    """
        
    #retrieve inputs
    sModelVersion = tInputs[0]
    dMu = tInputs[4]
    tDataInfo = tInputs[5]
    lClientTrainDataRaw = tDataInfo[0].copy()
    diMinMaxInfo = tDataInfo[1]
    diCatUniqueValuesInfo = tDataInfo[2]
    lDataTypes = tDataInfo[3]
    delimiterInfo = tDataInfo[4]
    i = tDataInfo[5]
    
    #check version
    if sModelVersion == "noMode" or sModelVersion == 'noModeSingleClient':
        fedDPCTGAN = FedDPCTGAN(dMu = dMu, iSeed = 1234, iSteps = min(1000, int(round(1000*np.sqrt(0.75)*dMu))), bVerbose=False, sLoss="cross-entropy", iDiscrUpdatesPerStep = 5, iGenUpdatesPerStep = 1, sDevice = "cuda", sClientOpt = "Adam", bModeLoss = False)
       
    elif sModelVersion == "withMode" or sModelVersion == 'withModeSingleClient':
        fedDPCTGAN = FedDPCTGAN(dMu = dMu, iSeed = 1234, iSteps = min(1000,  int(round(1000*np.sqrt(0.75)*dMu))), bVerbose=False, sLoss="cross-entropy", iDiscrUpdatesPerStep = 5, iGenUpdatesPerStep = 1, sDevice = "cuda", sClientOpt = "Adam", bModeLoss = True)
    
    fedDPCTGAN.fit(lClientTrainDataRaw, lDataTypes, diMinMaxInfo, diCatUniqueValuesInfo, delimiterInfo)[0]
    fedDPCTGAN.globalDiscriminator.to("cpu")
    fedDPCTGAN.globalGenerator.to("cpu")
    return [(i,dMu), fedDPCTGAN]

def main():
    #set seed
    iSeed = 1234

    #path
    #desktop: "D:/master_thesis/thesis_repos"
    #laptop: "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian"
    computerPath = "C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian" #laptop: D:/master_thesis/thesis_repos

    #set setting
    sSetting = "homogeneous_equalsize" #homogeneous_equalsize, heterogeneous_equalsize, homogeneous_diffsize 
    sModelType = "PGM" #PGM, GAN
    sModelVersion = "FIST" #"FTST" #Independent, FIST, noMode, withMode, FTSTSingleClient, FISTSingleClient, withModeSingleClient
    sCorrBins = "noCorrBins" #corrBins, noCorrBins, NA
    sScoreType = "defaultScore" #defaultScore, adjustedScore, randomScore, localModel, noScaffold, withScaffold
    sMaxDegree = "maxDegree3" #maxDegree2, maxDegree3, NA
    sDataset = "loan"

    #load data
    sRef = f"{computerPath}/pickle_data/{sDataset}.pickle"
    llClientTrainDataRaw, llClientTestDataRaw, lTotalTrainData, lTotalTestData,  diMapper, lDataTypes, sYname, diMinMaxInfo, delimiterInfo, diCatUniqueValuesInfo = get_FL_datasets(sRef=sRef, sSetting = sSetting, dTrainFrac = 0.75, iRep = 5, iSeed = iSeed)

    #get inputs
    lModelReps = [0,1,2] #,3,4,5,6,7,8,9
    if sModelVersion == "FTSTSingleClient" or sModelVersion == "FISTSingleClient" or sModelVersion == "withModeSingleClient":
        ltData = [([llClientTrainDataRaw[i][0]], diMinMaxInfo, diCatUniqueValuesInfo, lDataTypes, delimiterInfo, i) for i in lModelReps]
    else:
        ltData = [(llClientTrainDataRaw[i], diMinMaxInfo, diCatUniqueValuesInfo, lDataTypes, delimiterInfo, i) for i in lModelReps]
    lMu = [ 0.01, 0.50, 0.25, 0.10, 0.05, 0.025]   #2.50, 1.00, 0.50,
    iterInputs = iter.product([sModelVersion], [sCorrBins], [sScoreType], [sMaxDegree], lMu, ltData)

    # Get federated datasets for evaluation (Federated dataset is the same for each split i, to save memory we do not save the returned datasets for every model but only this one)
    bGetFedDatasets = False
    if bGetFedDatasets:
        #get list of federated datasets
        lFedDatasets = get_fedDatasets_objects(ltData)
        #save mdatasets
        path = f"{computerPath}/models/lFedDatasets_{sSetting}_{sDataset}.pickle"
        with open(path, 'wb') as handle:
            pk.dump(lFedDatasets, handle)
        print("succes")
        return None

    #multiprocessing
    iTot = len(lModelReps) * len(lMu)
    t1 = time.time()
    #iTotal =  len([1.0 for el in iterInputs])
    if sModelType == "PGM":
        with mp.Pool(processes = 6) as pl:
            results = list(tqdm.tqdm(pl.imap_unordered(run_parallel_PGM, iterInputs), total = iTot))
            pl.close()
            pl.join()
    elif sModelType == "GAN":
        #loop of models
        results = []
        for tInputs in iterInputs:
            t2 = time.time()
            print(f"fitting dataset {tInputs[5][5]} for mu = {tInputs[4]} after {round(t2-t1,3)} seconds")
            res = run_GAN(tInputs)  
            results.append(res)
            torch.cuda.empty_cache()
            
            #make dictionary object
            diResults = dict()

            for i in range(len(results)):
                diResults[results[i][0]] = results[i][1]

            
            path = f"{computerPath}/models/{sSetting}_{sModelType}_{sModelVersion}_{sCorrBins}_{sScoreType}_{sMaxDegree}_{sDataset}.pickle"
            with open(path, 'wb') as handle:
                pk.dump(diResults, handle)
    
    print(results)
    t2 = time.time()
    print(f"Took {round(t2-t1,3)} seconds")

    #make dictionary object
    diResults = dict()

    for i in range(len(results)):
        diResults[results[i][0]] = results[i][1]

    #save models
    path = f"{computerPath}/models/{sSetting}_{sModelType}_{sModelVersion}_{sCorrBins}_{sScoreType}_{sMaxDegree}_{sDataset}.pickle"
    with open(path, 'wb') as handle:
        pk.dump(diResults, handle)


if __name__ == '__main__':
    main()