#goal: load data, and define everything needed for later steps.
#author: Julian van Erk

import pandas as pd
import numpy as np
import pickle as pk


def main():
    #get rng
    rng = np.random.default_rng(seed = 1234)

    #get adult dataset
    adult_raw = pd.concat([
        pd.read_csv("C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/original_data/adult/adult.csv", 
        usecols=["age","workclass","fnlwgt","education-num","marital-status", "occupation",
        "relationship","race","sex","capital-gain","capital-loss","hours-per-week",
        "native-country","income>50K"]),
        pd.read_csv("C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/original_data/adult/adult_test.csv", 
        usecols=["age","workclass","fnlwgt","education-num","marital-status", "occupation",
        "relationship","race","sex","capital-gain","capital-loss","hours-per-week",
        "native-country","income>50K"], escapechar='.')], ignore_index=True)
    adult_raw  = adult_raw.iloc[rng.choice(len(adult_raw), size = 40000, replace=False),:].reset_index(drop=True)
    diMapper = {" <=50K": 0, " >50K": 1}
    lDataTypes = ["numerical","categorical", "numerical", "ordinal", "categorical", 
                  "categorical", "categorical", "categorical", "categorical", "mixed",
                  "mixed", "numerical", "categorical", "categorical"]
    sYname = 'income>50K'
    adult_raw['education-num'] = adult_raw['education-num'] - 1
    minInfo = adult_raw.min()
    maxInfo = adult_raw.max()
    diMinMaxInfo = dict()
    diCatUniqueValuesInfo = dict()
    for d in range(len(lDataTypes)):
        sVarName = adult_raw.columns[d]
        if (lDataTypes[d] == "numerical") or (lDataTypes[d] == "ordinal") or (lDataTypes[d] == "mixed"):
            #get minimum and maximum info
            diMinMaxInfo[sVarName] = [minInfo[sVarName], maxInfo[sVarName]]
        else:
            #get names info
            diCatUniqueValuesInfo[sVarName] = list(pd.unique(adult_raw[sVarName]))#list(set(adult_raw[sVarName].values))

        
    delimiterInfo = { "age": 0, "fnlwgt": 0, "capital-gain":0, "capital-loss":0, "hours-per-week":0}
    
    dataInfoAdult = [adult_raw, diMapper, lDataTypes, sYname, diMinMaxInfo, delimiterInfo, diCatUniqueValuesInfo]

    
    #write to pickle 
    with open('C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/pickle_data/adult.pickle', 'wb') as handle:
        pk.dump(dataInfoAdult, handle)


    ###################################################################################################
    #get IKNL dataset
    cols = ['leeft', "gesl", "incjr", "vit_stat", "vit_stat_int", "tumsoort", "diag_basis", "topo_sublok","later","morf","diffgrad",
            "ct", "cn", "cm", "pt", "pn", "pm", "stadium", "cstadium", "pstadium", "ond_lymf", "pos_lymf","er_stat","pr_stat","her2_stat",
            "dcis_comp", "multifoc", "tum_afm", "swk_uitslag", "mari_uitslag", "okd", "org_chir", "uitgebr_chir_code", "dir_reconstr",
            "chemo", "target", "horm", "rt", "meta_rt", "meta_chir"]
    IKNL_raw = pd.read_csv("C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/original_data/IKNL/NKR_IKNL_breast_syntheticdata.csv", delimiter=";", usecols=cols)
    
    #sample
    IKNL_raw  = IKNL_raw.iloc[rng.choice(len(IKNL_raw), size = 40000, replace=False),:].reset_index(drop=True)
    
    #rewrite some things
    stadiumDict = {"0": 0, "1":1, "1A": 1, "1B": 2, "2A":3, "2B":4, "3":5, "3A":5, "3B":6, "3C":7, "4":8, "M":np.nan, "NVT":np.nan, "X":np.nan}
    IKNL_raw['stadium'].replace(stadiumDict, inplace = True)
    IKNL_raw['pstadium'].replace(stadiumDict, inplace = True)
    stadiumDict = {"0": 0, "1":1, "1A": 1, "2A":2, "2B":3, "3":4, "3A":4, "3B":5, "3C":6, "4":7, "M":np.nan, "NVT":np.nan, "X":np.nan}
    IKNL_raw['cstadium'].replace(stadiumDict, inplace = True)
    nanDict= {999: np.nan, 99: np.nan, 98:np.nan}
    IKNL_raw['tum_afm'].replace(nanDict, inplace = True)
    IKNL_raw['tum_afm'] = np.log(1.0 + IKNL_raw['tum_afm'])
    IKNL_raw['ond_lymf'].replace(nanDict, inplace = True)
    IKNL_raw['ond_lymf'] = np.log(1.0 + IKNL_raw['ond_lymf'])
    IKNL_raw['pos_lymf'].replace(nanDict, inplace = True)
    her2Dict = {4: np.nan, 7: np.nan, 9: np.nan}
    IKNL_raw['her2_stat'].replace(her2Dict, inplace = True)
    diagDict = {2:0, 5:1, 6:2, 7:3, 9:np.nan}
    IKNL_raw['diag_basis'].replace(diagDict, inplace = True)
    IKNL_raw['diffgrad'].replace(9, np.nan, inplace = True)
    yearDict = {2010:0, 2011:1, 2012:2, 2013:3, 2014:4, 2015:5, 2016:6, 2017:7, 2018:8, 2019:9}
    IKNL_raw['incjr'].replace(yearDict, inplace=True)
    
    #get datatypes
    diMapper = None #already 0-1
    lDataTypes = ["numerical", "categorical", "ordinal", "categorical", "numerical", "categorical", "ordinal", "categorical", "categorical",
        "categorical", "ordinal", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "ordinal", "ordinal",
        "ordinal", "numerical", "mixed", "categorical", "categorical", "ordinal", "categorical", "categorical", "numerical", "categorical", 
        "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", "categorical", 
        "categorical", "categorical"]
    sYname = "vit_stat"
    minInfo = IKNL_raw.min()
    maxInfo = IKNL_raw.max()
    diMinMaxInfo = dict()
    diCatUniqueValuesInfo = dict()
    for d in range(len(lDataTypes)):
        sVarName = cols[d]
        if (lDataTypes[d] == "numerical") or (lDataTypes[d] == "ordinal") or (lDataTypes[d] == "mixed"):
            #get minimum and maximum info
            diMinMaxInfo[sVarName] = [minInfo[sVarName], maxInfo[sVarName]]
        else:
            #get names info
            diCatUniqueValuesInfo[sVarName] = list(pd.unique(IKNL_raw[sVarName]))
    delimiterInfo = {"leeft": 0, "vit_stat_int": 0, "ond_lymf": 7, "pos_lymf": 0, "tum_afm": 7,
                     "incjr": 0, "diag_basis":0, "diffgrad":0 , "stadium":0 , "cstadium":0 , "pstadium": 0,
                     "her2_stat":0}
    dataInfoIKNL = [IKNL_raw, diMapper, lDataTypes, sYname, diMinMaxInfo, delimiterInfo, diCatUniqueValuesInfo]




    #write to pickle 
    with open('C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/pickle_data/IKNL.pickle', 'wb') as handle:
        pk.dump(dataInfoIKNL, handle)

    ##########################################################
    #loan dataset
    loanCols = ["age", "job", "marital", "education", "default", "housing", "loan", "contact",
    "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y"]
    loan_raw = pd.read_csv("C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/original_data/bank-additional/bank-additional-full.csv", delimiter=";", usecols=loanCols)
    
    
    #sample
    loan_raw   = loan_raw.iloc[rng.choice(len(loan_raw), size = 40000, replace=False),:].reset_index(drop=True)

    #replace some things
    loan_raw['pdays'].replace(999, np.nan, inplace = True)
    loan_raw['duration'] = np.log(1.0 + loan_raw['duration'])
    loan_raw['cons.price.idx'] = loan_raw['cons.price.idx'].round(1)
    loan_raw['cons.conf.idx'] = loan_raw['cons.conf.idx'].round(0) 
    loan_raw['euribor3m'] = loan_raw['euribor3m'].round(1)
    loan_raw['nr.employed'] = loan_raw['nr.employed']/10.0
    loan_raw['nr.employed'] = loan_raw['nr.employed'].round(0)

    #set data types
    lDataTypes = ["numerical", "categorical", "categorical", "categorical", "categorical",
        "categorical", "categorical", "categorical", "categorical", "categorical", "mixed",
        "numerical", "mixed", "ordinal", "categorical", "numerical", "numerical", "numerical",
        "numerical", "numerical", "categorical"]

    #set y var name
    sYname = "y"

    #set to 0-1
    diMapper = {"no":0, "yes": 1}

    #get min/max and cat unique values
    minInfo = loan_raw.min()
    maxInfo = loan_raw.max()
    diMinMaxInfo = dict()
    diCatUniqueValuesInfo = dict()
    for d in range(len(lDataTypes)):
        sVarName = loanCols[d]
        if (lDataTypes[d] == "numerical") or (lDataTypes[d] == "ordinal") or (lDataTypes[d] == "mixed"):
            #get minimum and maximum info
            diMinMaxInfo[sVarName] = [minInfo[sVarName], maxInfo[sVarName]]
        else:
            #get names info
            diCatUniqueValuesInfo[sVarName] = list(pd.unique(loan_raw[sVarName]))

    #set delimiter info
    delimiterInfo = {"age": 0,  "duration": 7, "campaign": 0, "pdays":0, 
        "previous": 0, "emp.var.rate": 1, "cons.price.idx": 1, "cons.conf.idx": 0,
        "euribor3m": 1, "nr.employed": 0}

    #set info
    dataInfoLoan = [loan_raw, diMapper, lDataTypes, sYname, diMinMaxInfo, delimiterInfo, diCatUniqueValuesInfo]

    #write to pickle 
    with open('C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/pickle_data/loan.pickle', 'wb') as handle:
        pk.dump(dataInfoLoan, handle)



    ##########################################################
    #credit dataset
    creditCols = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17",
                    "V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"]
    credit_fraud_raw = pd.read_csv("C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/original_data/credit_fraud/creditcard.csv", delimiter=",", usecols=creditCols)

    #sample
    credit_fraud_raw = credit_fraud_raw.iloc[rng.choice(len(credit_fraud_raw), size = 40000, replace=False),:].reset_index(drop=True)

    #replace some things
    credit_fraud_raw['Amount'] = np.log(1.0 + credit_fraud_raw['Amount'])
    def sym_log(x):
        return (x > 0) * np.log(1.0 + np.abs(x)) + (x <0 )*-np.log(1.0 + np.abs(x))
        #return (x - np.mean(x)) / np.std(x)
    credit_fraud_raw.iloc[:,1:29] = credit_fraud_raw.iloc[:,1:29].apply(sym_log, axis = 0)


    #set data types
    lDataTypes = ["numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical",
                "numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical",
                "numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical","numerical", "numerical","categorical"]

    #set y var name
    sYname = "Class"

    #set to 0-1
    diMapper = None

    #get min/max and cat unique values
    minInfo = credit_fraud_raw.min()
    maxInfo = credit_fraud_raw.max()
    diMinMaxInfo = dict()
    diCatUniqueValuesInfo = dict()
    for d in range(len(lDataTypes)):
        sVarName = creditCols[d]
        if (lDataTypes[d] == "numerical") or (lDataTypes[d] == "ordinal") or (lDataTypes[d] == "mixed"):
            #get minimum and maximum info
            diMinMaxInfo[sVarName] = [minInfo[sVarName], maxInfo[sVarName]]
        else:
            #get names info
            diCatUniqueValuesInfo[sVarName] = list(pd.unique(credit_fraud_raw[sVarName]))

    #set delimiter info
    delimiterInfo = {"Time":0, "V1":7,"V2":7,"V3":7,"V4":7,"V5":7,"V6":7,"V7":7,"V8":7,"V9":7,
                    "V10":7,"V11":7,"V12":7,"V13":7,"V14":7,"V15":7,"V16":7,"V17":7,
                    "V18":7,"V19":7,"V20":7,"V21":7,"V22":7,"V23":7,"V24":7,"V25":7,
                    "V26":7,"V27":7,"V28":7,"Amount":7}

    #set info
    dataCredit = [credit_fraud_raw, diMapper, lDataTypes, sYname, diMinMaxInfo, delimiterInfo, diCatUniqueValuesInfo]

    #write to pickle 
    with open('C:/Users/erkjrv/OneDrive - TNO/wp3-federated-synthetic-data/thesis_julian/pickle_data/credit.pickle', 'wb') as handle:
        pk.dump(dataCredit, handle)


    ######## save files #######

if __name__ == '__main__':
    main()