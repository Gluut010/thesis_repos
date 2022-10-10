#goal: Utils functions for convenience.
#author: Julian van Erk

import numpy as np
import pandas as pd
from itertools import chain, combinations
import torch
from torch.nn import functional as F
from opacus.accountants import RDPAccountant
from opacus import GradSampleModule#, GradSampleModuleExpandedWeights
from opacus.optimizers import DPOptimizer
import pickle as pk
from scipy.optimize import fsolve
import numpy as np
import math
from scipy.stats import norm

def isNaN(num):
    """
    Goal: detect whether num is NaN or not
    Input:  an object num
    Output: True/False boolean
    """
    return num != num

# divide data function
def FL_split_data(dfData, dTrainFrac = 0.70, iC = 5, lFrac = None, sYname = None, yFrac = None, rng = None, diMapper = None, seed = 1234):
    """
    goal: split data in FL setting
    input:
        - dfData:       pandas dataframe with original data
        - dTrainFrac    train fraction of dataset
        - iC            integer number of clients (only for equal fraction)
        - lFrac:        list with fraction of observations for each client
        - sYname:       name of dependent variable (only if non-iid)
        - yFrac:        list with fraction of y observations for each client
        - rng           random number generator 
        - diMapper      dictionary mapper for categorical y variable to numerical in form {"majority_class_name": 0, "minority_class_name": 1}
        - seed          seed of random number generator if no rng was given
    output:
        - lClientTrainData  list of pandas dataframes with client training data
        - lClientTestData   list of pandas dataframes with client test data

    """



    #get number of instances
    iN = dfData.shape[0]

    #create rng if there is none
    if rng is None:
        rng = np.random.default_rng(seed)

    #initialize empty list of dataset
    lClientTrainData = []
    lClientTestData = []

    # fill lFrac
    if lFrac is None:
        lFrac = iC*[1/iC]
    else:
        #check length
        if len(lFrac) != iC:
            raise ValueError("length of lFrac should be iC")

    if yFrac is None:
        #create array of shuffled indicies
        vShuffle = rng.choice(iN, size = iN, replace = False)
        vSplits = [0] + [int(el) for el in np.round(np.cumsum([iN*el2 for el2 in lFrac]))]

        for iClient in range(iC):
            #get indices
            vClientIndices = vShuffle[(vSplits[iClient]):(vSplits[iClient + 1])]

            #get total data
            dfClientData = dfData.iloc[vClientIndices]

            #split in train and test
            iNClient = dfClientData.shape[0]
            vShuffledInd = rng.choice(iNClient, size = iNClient, replace=False)
            iClientTrainSize = int(round(dTrainFrac*iNClient))
            vTrainInd = vShuffledInd[:iClientTrainSize]
            vTestInd = vShuffledInd[iClientTrainSize:]

            #split train test
            vClientTrainData = dfClientData.iloc[vTrainInd]
            vClientTestData = dfClientData.iloc[vTestInd]

            #fill list
            lClientTrainData += [vClientTrainData]
            lClientTestData += [vClientTestData]
    else:

        #check if yFrac, lFrac have same lengts
        if len(yFrac) != len(lFrac):
            raise ValueError("lFrac, yFrac should be same size iC")
        #check if yFrac sums to one
        if sum(yFrac) != 1:
            #normalize
            yFrac = [yFrac[i]/sum(yFrac) for i in range(iC)]

        #map categorical y to numerical
        if diMapper is None:
            vYNum = dfData[sYname]
        else:
            vYNum = dfData[sYname].map(diMapper)
        

        #check if number of y-s is not larger than number of observations.
        dyFrac = np.mean(vYNum)
        for i in range(len(yFrac)):
            if yFrac[i]*dyFrac > lFrac[i]:
                raise ValueError(f"Number of y = 1 we should assign larger than dataset can be for client {i}")

        #get indices where y = 1 and y = 0
        lYisOneIndices = dfData.index[vYNum == 1].tolist() #dfData.index[dfData[sYname] == 1].tolist()
        lYisZeroIndices = dfData.index[vYNum == 0].tolist()#dfData.index[dfData[sYname] == 0].tolist()

        #get lengths
        dNumberOfOnes = len(lYisOneIndices)
        dNumberOfZeros = len(lYisZeroIndices)
        
        #shuffle indices
        vShuffleOne = rng.choice(lYisOneIndices, size = dNumberOfOnes, replace = False)
        vShuffleZero = rng.choice(lYisZeroIndices, size = dNumberOfZeros, replace=False)

        #get observations per client
        lObs = np.diff((np.cumsum(np.array([0.0]+lFrac)*dfData.shape[0]).astype(int)))

        #get positive observations per client (y = 1)
        lPosSplits = np.cumsum(np.array([0.0]+yFrac)*dNumberOfOnes).astype(int)
        lPosObs = np.diff(lPosSplits)

        #get negative observations per client (y = 0)
        lNegObs = [lObs[i] - lPosObs[i]  for i in range(iC)]
        lNegSplits = [0] + np.cumsum(lNegObs).tolist()
    

        for iClient in range(iC):
            #get indices
            vClientIndicesPos = vShuffleOne[(lPosSplits[iClient]):(lPosSplits[iClient + 1])]
            vClientIndicesNeg = vShuffleZero[(lNegSplits[iClient]):(lNegSplits[iClient + 1])]

            #get total data
            dfClientData = dfData.iloc[vClientIndicesPos.tolist() + vClientIndicesNeg.tolist()]

            #split in train and test
            iNClient = dfClientData.shape[0]
            vShuffledInd = rng.choice(iNClient, size = iNClient, replace=False)
            iClientTrainSize = int(round(dTrainFrac*iNClient))
            vTrainInd = vShuffledInd[:iClientTrainSize]
            vTestInd = vShuffledInd[iClientTrainSize:]

            #split train test
            vClientTrainData = dfClientData.iloc[vTrainInd]
            vClientTestData = dfClientData.iloc[vTestInd]

            #fill list
            lClientTrainData += [vClientTrainData]
            lClientTestData += [vClientTestData]

    return lClientTrainData, lClientTestData

def opt_function(mu,  delta , eps):
    """
    Goal: helper function for estimating mu form delta-eps or eps form delta-mu
    Input: 
        - mu       Gaussian differential privacy parameter
        - delta    delta value
        - epsilon  epsilon value
    Output:
        - difference between mu-delta en eps-delta
    """
    return (norm.cdf( - (eps/mu) + (mu/2.0)) - np.exp(eps)*(norm.cdf( - (eps/mu) - (mu/2.0)) ) - delta)

def mu_from_eps_delta(eps, delta = 10**(-6)):
    """
    Goal: Get mu estimate from a delta-epsilon combination using numerical optimization
    Input:
        - delta     delta value
        - eps       epsilon value
    Ouptut:
        - mu        Guassian differential privacy parameter
    """
    #optimize
    mu =  fsolve(lambda x: opt_function(x, delta, eps), x0 = 10.5, maxfev=10000)
    return mu

def eps_from_mu_delta(mu, delta = 10**(-6)):
    """
    Goal: Get mu estimate from a delta-epsilon combination using numerical optimization
    Input:
        - delta     delta value
        - eps       epsilon value
    Ouptut:
        - mu        Guassian differential privacy parameter
    """
    eps = fsolve(lambda x: opt_function(mu, delta, x), x0 = 0.0, maxfev=1)
    return eps




def get_FL_datasets(sRef, sSetting = "homogeneous", dTrainFrac = 0.75, iRep = 5, iSeed = 1234):
    """
    Goal: get the federated datasets for iC = 5. Function to always get the same splits
    Input:
        - sRef           string, link to pickle data 
        - sSetting       string, FL setting, homogenous vs heterogeneous
        - dTrainFrac     double, fraction of train observations for training
        - iRep           integer, number of repetitions
        - iSeed          integer, seed
    Output:
        - llClientTrainDataRaw      list of pandas dataframes with client training data
        - llClientTestDataRaw       list of pandas dataframes with client test data
        - lTotalTrainData           list of total train datasets
        - lTotalTestData            list of total test datasets
        - diMapper                  dictionary, mapper from string to 1/0 for dependent variable
        - lDataTypes                list of data types of variables
        - sYname                    string, name of dependent variable
        - diMinMaxInfo              dictionary, information on min and max info
        - delimiterInfo             dictionary, information on number of decimals
        - diCatUniqueValuesInfo     dictionary, infomation unique values of categorical values
    """

    #set seed
    rng = np.random.default_rng(iSeed)

    #load raw dataset
    with open(sRef, 'rb') as handle:
        data, diMapper, lDataTypes, sYname, diMinMaxInfo, delimiterInfo, diCatUniqueValuesInfo = pk.load(handle)

    #set setting
    iC = 5
    if sSetting == "homogeneous_equalsize":
        yFrac = [1.0,1.0,1.0,1.0,1.0]
        lFrac = iC * [1/iC]
    elif sSetting == "heterogeneous_equalsize":
        yFrac = [5.0,2.0,1.0,1.0,1.0]
        lFrac = iC * [1/iC]
    elif sSetting == "homogeneous_diffsize":
        yFrac = [0.5,0.2,0.1,0.1,0.1]
        lFrac = [0.5,0.2,0.1,0.1,0.1]

    #split data
    llClientTrainDataRaw = iRep*[None]
    llClientTestDataRaw = iRep*[None]
    lTotalTrainData = iRep*[None]
    lTotalTestData = iRep*[None]
    for i in range(iRep):
        llClientTrainDataRaw[i], llClientTestDataRaw[i] = FL_split_data(data, dTrainFrac = dTrainFrac, iC = iC, lFrac = lFrac, sYname = sYname, yFrac = yFrac, rng = rng, diMapper = diMapper)
            
        #create total train, test
        lTotalTrainData[i] = pd.concat(llClientTrainDataRaw[i])
        lTotalTestData[i] = pd.concat(llClientTestDataRaw[i])

    return llClientTrainDataRaw, llClientTestDataRaw, lTotalTrainData, lTotalTestData,  diMapper, lDataTypes, sYname, diMinMaxInfo, delimiterInfo, diCatUniqueValuesInfo


def get_lipschitz_constant(engine, measurements):
        """
        Goal: get the lipschitz constant needed for dual averaging (special case where Q = I)
        input:
            - engine        engine object 
            - measurements  all measurements
        output:
            - dEigenMax     the max eigenvalue, or the lipschitz constant
        """
        #define temp engine
        tempEngine = engine
        tempEngine._setup(measurements, total = None)
        #predefine eigenvalues
        diEigenValues = { clique : 0.0 for clique in tempEngine.model.cliques}
        #find largest eigenvalue
        for measurement in measurements:
            dSigma = measurement[2]
            proj = measurement[3]
            Q = measurement[0]
            if not isinstance(Q, np.ndarray):
                Q = Q.todense()
            for clique in tempEngine.model.cliques:
                if set(proj) <= set(clique):
                    iN = tempEngine.domain.size(clique)
                    iP = tempEngine.domain.size(proj)
                    diEigenValues[clique] += np.max(np.linalg.eigvalsh(Q.T @ Q)) * iN / iP / dSigma**2
                    break
        return(max(diEigenValues.values()))

def map_to_other_discritization(dfData, lDatatypes, diFinalBinsSelf,  diFinalBinsOther, rng):
    """
    Goal: map discritezed dataset to other (hierarchical) discritization
    Input:
        - dfData        discretized pandas dataframe that needs to have a different discritization
    """

    #get number of columns
    iD = dfData.shape[1]

    #predefine new data dict
    diNewData = dict()

    #loop over columns
    for d in range(iD):
        #get variable name
        sVarName = dfData.columns[d]

        #get bins
        lBinsSelf = diFinalBinsSelf[sVarName]
        lBinsOther = diFinalBinsOther[sVarName]

        #initialize map dictionary
        diMap = dict()

        #get number of different values original and other
        iNumValuesSelf = len(lBinsSelf)
        iNumValuesOther = len(lBinsOther)

        #perform transformation
        if (lDatatypes[d] == "numerical") or (lDatatypes[d] == "ordinal"):   
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

def bin_subset_bin(bin, superBin):
    """
    Goal: check if bin is a subset of another bin
    Input: 
        - bin       first bin 
        - superBin  second bin 
    return
        - binary    1/0. 1 if bin is a subset of the superbin, 0 otherwise.
    """

    if len(bin) == 1:
        if bin == superBin:
            return 1
        else:
            return 0
    else:
        binMin = bin[0]
        binMax = bin[1]
        if len(superBin) == 1:
            return 0
        superBinMin = superBin[0]
        superBinMax = superBin[1]
        if ( (binMin >= superBinMin) and (binMax <= superBinMax)):
            return 1
        else:
            return 0



def powerset(iterable, iMaxLevel = 3, iMinLevel = 0):
    """
    Goal: get powerset up to level iMaxLevel
    Input:
        - iterable  iterable of variable names 
        - iMaxLevel maximum level combination

    Output:
        - lPowerset powerset of all combinationas up to level iMaxLevel
    """
    s = list(iterable)
    lPowerset = list(chain.from_iterable(combinations(s, r) for r in range(iMinLevel, iMaxLevel+1)))
    return lPowerset


def downward_closure(lCliques):
    """
    Goal: get the downward closure of a list of cliques
    Input:
        - lCliques  list of cliques
    Output:
        - lDownwardClosure   list of downward closure of the list of cliques
    """
    ans = set()
    for cl in lCliques:
        ans.update(powerset(cl))
    lDownwardClosure = list(sorted(ans, key=len))
    return lDownwardClosure

def sum_dimensions(mCounts, clique, diCorrBins):
    """
    Goal: sum counts of bins taken together for correlation measuring
    Input:
        - mCounts    Original counts 
        - clique     tuple/list of variables measured in this clique
        - diCorrBins dictionairy with the correlation bins per variable
    Output:
        - mCounts   Updated counts
    
    """

    for i in range(len(clique)):
        sVarName = clique[i]
        llCorrBins = diCorrBins[sVarName]
        idx = np.array([x for xs in llCorrBins for x in xs])
        idSum = np.cumsum([[0] + [len(x) for x in llCorrBins [:-1]]])
        mCounts = np.take(mCounts, indices = idx, axis = i)
        mCounts = np.add.reduceat(mCounts, idSum, axis = i)

    return mCounts

def get_counts(countobject, clique, diCorrBins):
    """
    Goal: Get counts of a dataset
    Input: 
        - countobject  a dataset object of mbi, or a graphical model object of mbi
        - clique       the clique to be measured
        - diCorrBins   dictionary with correlation-bins
    Output:
        - vCounts       vector with counts
    """
    #get original counts
    mCounts = countobject.project(clique).datavector(flatten = False)

    #get counts
    vCounts = sum_dimensions(mCounts, clique, diCorrBins).flatten()
    return vCounts

def delim_ceil(a, delim=0):
    return np.true_divide(np.ceil(a * 10**delim), 10**delim)

def delim_floor(a,  delim=0):
    return np.true_divide(np.floor(a * 10**delim), 10**delim)




def privatise_model_optimizer(model, optimizer, batch_size, dClipBound, dSigma, sample_rate = 1):
    """
    make an optimizer and the asociated model private
    """
    # initialize privacy accountant
    accountant = RDPAccountant()

    # wrap model
    #dp_model = GradSampleModule(model.module)
    #print(dp_model.state_dict().keys())
    #dp_model = GradSampleModuleExpandedWeights(model, loss_reduction="mean")
    dp_model = GradSampleModule(model, loss_reduction="mean")
    #dp_model._modules = model._modules

    # wrap optimizer
    dp_optimizer = DPOptimizer(
        optimizer=optimizer,
        noise_multiplier=dSigma, # same as make_private arguments
        max_grad_norm=dClipBound, # same as make_private arguments
        expected_batch_size=batch_size, # if you're averaging your gradients, you need to know the denominator
    )

    # attach accountant to track privacy for an optimizer
    dp_optimizer.attach_step_hook(
        accountant.get_optimizer_hook_fn(
        # this is an important parameter for privacy accounting. Should be equal to batch_size / len(dataset)
        sample_rate=sample_rate
        )
    )

    return dp_model, dp_optimizer


def get_st_ed(target_col_index,output_info):
    
    """
    NOTE: from https://github.com/Team-TUD/CTAB-GAN/blob/main/model/synthesizer/ctabgan_synthesizer.py
    Used to obtain the start and ending positions of the target column as per the transformed data to be used by the classifier 
    Inputs:
    1) target_col_index -> column index of the target column used for machine learning tasks (binary/multi-classification) in the raw data 
    2) output_info -> column information corresponding to the data after applying the data transformer
    Outputs:
    1) starting (st) and ending (ed) positions of the target column as per the transformed data
    
    """
    # counter to iterate through columns
    st = 0
    # counter to check if the target column index has been reached
    c= 0
    # counter to iterate through column information
    tc= 0
    # iterating until target index has reached to obtain starting position of the one-hot-encoding used to represent target column in transformed data
    for item in output_info:
        # exiting loop if target index has reached
        if c==target_col_index:
            break
        if item[1]=='tanh':
            st += item[0]
        elif item[1] == 'softmax':
            st += item[0]
            c+=1 
        tc+=1    
    
    # obtaining the ending position by using the dimension size of the one-hot-encoding used to represent the target column
    ed= st+output_info[tc][0] 
    
    return (st,ed)


def apply_activate(data, output_info):
    
    """
    NOTE: from https://github.com/Team-TUD/CTAB-GAN/blob/main/model/synthesizer/ctabgan_synthesizer.py
    This function applies the final activation corresponding to the column information associated with transformer
    Inputs:
    1) data -> input data generated by the model in the same format as the transformed input data
    2) output_info -> column information associated with the transformed input data
    Outputs:
    1) act_data -> resulting data after applying the respective activations 
    """
    
    data_t = []
    # used to iterate through columns
    st = 0
    # used to iterate through column information
    for item in output_info:
        # for numeric columns a final tanh activation is applied
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        # for one-hot-encoded columns, a final gumbel softmax (https://arxiv.org/pdf/1611.01144.pdf) is used 
        # to sample discrete categories while still allowing for back propagation 
        elif item[1] == 'softmax':
            ed = st + item[0]
            # note that as tau approaches 0, a completely discrete one-hot-vector is obtained
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed
    
    act_data = torch.cat(data_t, dim=1) 

    return act_data


def cond_loss(data, output_info, c, m):
    #NOTE: from CTABGAN https://github.com/Team-TUD/CTAB-GAN/blob/32ec57cc2772e52e325fedff54ab25442cd55bff/model/synthesizer/ctabgan_synthesizer.py#L81
    """
    Used to compute the conditional loss for ensuring the generator produces the desired category as specified by the conditional vector
    Inputs:
    1) data -> raw data synthesized by the generator 
    2) output_info -> column informtion corresponding to the data transformer
    3) c -> conditional vectors used to synthesize a batch of data
    4) m -> a matrix to identify chosen one-hot-encodings across the batch
    Outputs:
    1) loss -> conditional loss corresponding to the generated batch 
    """
    
    # used to store cross entropy loss between conditional vector and all generated one-hot-encodings
    tmp_loss = []
    # counter to iterate generated data columns
    st = 0
    # counter to iterate conditional vector
    st_c = 0
    # iterating through column information
    for item in output_info:
        # ignoring numeric columns
        if item[1] == 'tanh':
            st += item[0]
            continue
        # computing cross entropy loss between generated one-hot-encoding and corresponding encoding of conditional vector
        elif item[1] == 'softmax':
            ed = st + item[0]
            ed_c = st_c + item[0]
            tmp = F.cross_entropy(
            data[:, st:ed],
            torch.argmax(c[:, st_c:ed_c], dim=1),
            reduction='none')
            tmp_loss.append(tmp)
            st = ed
            st_c = ed_c

    # computing the loss across the batch only and only for the relevant one-hot-encodings by applying the mask 
    tmp_loss = torch.stack(tmp_loss, dim=1)
    loss = (tmp_loss * m).sum() / data.size()[0]

    return loss

def mode_loss(data, output_info, vOneWayProbs, mCond, mMask):
    #NOTE: from CTABGAN https://github.com/Team-TUD/CTAB-GAN/blob/32ec57cc2772e52e325fedff54ab25442cd55bff/model/synthesizer/ctabgan_synthesizer.py#L81
    """
    Used to compute the mode loss for ensuring the generator produces the desired category as specified by the conditional vector
    Inputs:
    1) data -> raw data synthesized by the generator 
    2) output_info -> column informtion corresponding to the data transformer
    3) vOneWayProbs
    4) m -> a matrix to identify chosen one-hot-encodings across the batch
    Outputs:
    1) loss ->mode loss corresponding to the generated batch 
    """
    
    # used to store cross entropy loss between conditional vector and all generated one-hot-encodings
    loss = []
    # counter to iterate generated data columns
    st = 0
    # counter to iterate conditional vector
    st_c = 0
    i = 0
    # iterating through column information
    for item in output_info:
        # ignoring numeric columns
        if item[1] == 'tanh':
            st += item[0]
            continue
        # computing cross entropy loss between generated one-hot-encoding and corresponding encoding of conditional vector
        elif item[1] == 'softmax':
            ed = st + item[0]
            ed_c = st_c + item[0]
            vProbsEst = torch.mean(data[:, st:ed], dim = 0)
            loss_temp = torch.linalg.norm(vOneWayProbs[st_c:ed_c]-vProbsEst)**2
            #print(vProbsEst)
            #print(torch.sum(vProbsEst))
            i = i+ 1
            loss.append(loss_temp)
            st = ed
            st_c = ed_c 
    loss =  torch.stack(loss).sum()
    return loss


def get_zero_init(net):
    """
    Goal: Get a zero initialization of the network parameters
    Input:
        - net       A network we need the shape of
    Ouput:
        - diZeros    A dictionary with zeros in the shape of the network
    """
    #get shape
    diStart = dict(net.state_dict())

    #set all values to zero
    for key, value in diStart.items():
        diStart[key] = torch.zeros_like(value)

    diZeros = diStart
    return diZeros


def get_average_dict(ldi):
    """
    Goal: find average of list of dictionaries of params (or gradients)
    Input:
        - ldi       list of dictionaries
    Output
        - diAvg     dictionary with average values
    """
    #initialize dictionary
    diAvg = dict()
    for key, value in ldi[0].items():
        diAvg[key] = torch.zeros_like(value)
    
    #get length
    iLen = len(ldi)

    #sum dictionaries
    for di in ldi:
        for key, value in di.items():
            diAvg[key] += value * (1.0/iLen)

    return diAvg
    

def get_param_dict(net):
    """
    Goal: Get the dictionary of parameters of a network
    Input:
        - net       A network we need the parameters of
    Ouput:
        - diParams   A dictionary with the parameters
    """
    return dict(net.state_dict())

def get_grad_dict(net):
    """
    Goal: get a dictionary of the current gradients of network net
    Input:
        - net       A network we would like the current values of the gradients of 
    Ouput:
        - diGrad    A dictionary with the gradients
    """
    diGrad = {k:v.grad for k,v in net.named_parameters()}
    return diGrad

def set_gradients_net(net, diGrad):
    """
    Goal: set the gradients of net using diGrad
    Input:
        - net       A network we would like to set the gradients of
        - diGrad    dictionary with new gradients
    Output:
        - net       The network with the updated gradients
    """
    for k, v in net.named_parameters():
        v.grad = diGrad[k]
    return net

