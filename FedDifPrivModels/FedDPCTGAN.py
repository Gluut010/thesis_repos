
#goal: define federated differentially private tabular GAN model object
#author: Julian van Erk
from torch.nn import functional as F
from FedDifPrivModels.FedDataset import FedDataset
import numpy as np
import torch
from torch import conv2d, int32, less_equal, optim
from torch import nn
import copy
import torch.utils.data
from torch.nn import( BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, Sigmoid, LayerNorm, GroupNorm,
                    Conv2d, ConvTranspose2d, BatchNorm2d, init, BCELoss, CrossEntropyLoss, SmoothL1Loss)
import opacus
from FedDifPrivModels.FedGANDataset import FedGANDataset, ImageTransformer
from FedDifPrivModels.Utils import cond_loss, mode_loss, privatise_model_optimizer, apply_activate, get_average_dict, get_grad_dict, set_gradients_net, get_zero_init, get_param_dict #get_st_ed, 


class FedDPGANOptimizer():
    """
    Class for federated differentially private optimization of GANs
    """

    def __init__(self,  globalDiscriminator, globalGenerator, iNoiseDim, iDiscrUpdatesPerStep =5, iGenUpdatesPerStep = 1,
                 iStepsPerRound = 10, dSigma = None, iSteps = None, rng = None, iSeed = None,
                 sFedType = "fedAvg", sClientOpt = "SGD", dMuTrain = 1.0, dClipBound = 1.0, dEtaDiscr = 0.001, 
                 dEtaGen = 0.001, tBetasGen = (0.5, 0.9), tBetasDiscr = (0.5, 0.9), dWeightDecayGen = 1e-6, 
                 dWeightDecayDiscr = 1e-6, dNumStabAdam = 1e-8, sLoss = "WGAN-GP", sDevice = "cpu", bVerbose = False,
                 iInitCommRounds=0, iB = 5000, iNGenSample = 500, bModeLoss = True):
        """
        Goal: constructor of fedOptimizer object. Each step there are iDiscrUpdatesPerStep and iGenUpdatesPerStep. Each Epoch consist of
              iStepsPerEpoch steps (determines the batch size, note that values other than one results in correct but not tight privacy guarentees).
              Each Communication round there are iEpochsPerRound epochs before sending to the global model and updating the global model.
        Input:
            - fedGANDataset         A FedGANDataset object
            - globalDiscriminator   A Discriminator object, the global discriminator (or critic)
            - globalGenerator       A Generator object, the global generator
            - iNoiseDim             integer, the input noise dimension for the generator
            - iDiscrUpdatesPerStep  integer, the number of discriminator updates per step
            - iGenUpdatesPerStep    integer, the number of generator updates per step
            - iStepsPerRound        integer, the number of steps per communication round
            - dSigma                double, the standard deviation of the noise added to the discriminator gradient
                                    Either pass dSigma or iSteps
            - iSteps                integer, the number of steps for the optimization procedure (1 step is iDiscrUpdatesPerStep
                                    updater for the discriminator and iGenUpdatesPerStep for the generator)
            - rng                   A (numpy) random number generator object. Either pass rng or iSeed
            - iSeed                 integer, seed for a random number generator. Either pass rng or iSeed
            - sFedType              string, the Federated learning type, either "FedAvg" for federated averaging or "SCAFFOLD"
                                    for the SCAFFOLD algorithm.
            - sClientOpt            string, the client optimizer used. Either "Adam" or "SGD" (SGD withoud momentum)
            - dMuTrain              double, the privacy budget available for training
            - dClipBound            double, the clipping bound for gradients. Default 1 is "optimal"for sLoss="WGAN-GP".
            - dEtaDiscr             double, the learning rate for the discriminator
            - dEtaGen               double, the learning rate for the generator
            - tBetasGen             tuple, betas for adam algorithm for the generator (only applicable if sClientOpt = "Adam")
            - tBetasDiscr           tuple, betas for adam algorithm for the discriminator (only applicable if sClientOpt = "Adam")
            - dWeightDecayGen       double, weight decay parameter for the neural network of the generator
            - dWeightDecayDiscr     double, weight decay parameter of the neural network of the discriminator
            - dNumStabAdam          double, numerical stability parameter for adam (only applicable if sClientOpt = "Adam")
            - sLoss                 string, loss used, either "cross-entropy" for basic GAN or "WGAN-GP" for the wasserstein gan
                                    with gradient penalty
            - sDevice               string, Device for computation (e.g. cpu or cuda:0)
            - bVerbose              boolan, print verbose messages
            - iInitCommRounds       integer number of initial communication rounds where only the generator is updated using the conditional loss
            - iB                    integer, number of generated samples to compute mode estimates for mode loss
            - iNGenSample           integern, number of generated samples for conditional and generator loss
            - bModeLoss             boolean, use the mode loss or not
        Set
            - self      A FedDPCTGAN object
        """

        #set random number generator
        if rng is None:
            if iSeed == None:
                self.rng = np.random.default_rng(1234)
            else:
                self.rng = np.random.default_rng(iSeed)
        else:
            self.rng = rng

        #check loss type
        if sLoss == "WGAN-GP":
            self.sLoss = sLoss
        elif sLoss == "cross-entropy":
            self.sLoss = sLoss
        else:
            raise ValueError(f"Loss type: {sLoss} is not available, only WGAN-GP or cross-entropy")

        if self.sLoss == "WGAN-GP":
            # Monkeypatches the _create_or_extend_grad_sample function when calling opacus
            opacus.grad_sample.utils.create_or_accumulate_grad_sample = (
                _custom_create_or_extend_grad_sample
            )

        
        #check federated learning type
        if sFedType == "fedAvg":
            self.sFedType = "fedAvg"
        elif sFedType == "SCAFFOLD":
            self.sFedType = "SCAFFOLD"
        else:
            raise ValueError(f"federated optimizer: {sFedType} is not available, only fedAvg or SCAFFOLD")

        #check client optimizer
        if sClientOpt == "SGD":
            self.sClientOpt = sClientOpt
        elif sClientOpt == "Adam":
            self.sClientOpt = sClientOpt
        else:
            raise ValueError(f"Client optimizer: {sClientOpt} is not available, use SGD or Adam")

        #check typediscriminator, generator
        if isinstance(globalDiscriminator, CnnDiscriminator):
            self.sDiscrType = "CNN"
        elif isinstance(globalDiscriminator, MlpDiscriminator):
            self.sDiscrType = "MLP"
        else:
            raise ValueError(f"Type: {type(globalDiscriminator)} is not available for the discriminator, use CnnDiscriminator or MlpDiscriminator")

        #check typediscriminator, generator
        if isinstance(globalGenerator, CnnGenerator):
            self.sGenType = "CNN"
        elif isinstance(globalGenerator, MlpGenerator):
            self.sGenType = "MLP"
        else:
            raise ValueError(f"Type: {type(globalGenerator)} is not available for the generator, use CNN or MLP")

        #set optimizer specifics
        self.iDiscrUpdatesPerStep = iDiscrUpdatesPerStep
        self.iGenUpdatesPerStep = iGenUpdatesPerStep
        self.iStepsPerRound = iStepsPerRound
        self.iInitCommRounds = iInitCommRounds
        self.dEtaGen = dEtaGen
        self.dEtaDiscr = dEtaDiscr
        self.tBetasGen = tBetasGen
        self.tBetasDiscr = tBetasDiscr
        self.dClipbound = dClipBound
        self.dWeightDecayGen = dWeightDecayGen
        self.dWeightDecayDiscr = dWeightDecayDiscr
        self.dNumStab = dNumStabAdam

        #other
        self.sDevice = sDevice
        self.iNoiseDim = iNoiseDim
        self.iB = iB
        self.iNGenSample = iNGenSample
        self.bModeLoss = bModeLoss

        #set privacy accounter
        self.dMuTrain = dMuTrain
        self.dMuUsed = 0.0
        self.bExhausted = False

        #set sigma and number of steps
        if dSigma is None:
            if iSteps is None:
                ValueError("Set either number of steps or sigma")
            else:
                self.iSteps = iSteps
                self.dSigma = (self.dClipbound * np.sqrt(self.iSteps) * np.sqrt(self.iDiscrUpdatesPerStep))/self.dMuTrain
        if dSigma is not None:
            if iSteps is not None:
                ValueError("Set either number of steps or sigma")
            else:
                self.dSigma = dSigma
                self.iSteps = int(np.ceil( ((self.dSigma * self.dMuTrain)/(self.dClipbound * np.sqrt(self.iDiscrUpdatesPerStep) ))**2))
                if self.iSteps < 1:
                    print("increase sigma or mu")


        
        #set steps, rounds
        self.iRounds = 0
        self.vDiscrLoss = np.zeros((self.iSteps,))
        self.vGenLoss = np.zeros((self.iSteps,))
        self.vCondLoss = np.zeros((self.iSteps,)) #added
        self.vModeLoss = np.zeros((self.iSteps,))
        self.vCondLossInit = np.zeros((self.iStepsPerRound*self.iInitCommRounds,))
        self.vModeLossInit = np.zeros((self.iStepsPerRound*self.iInitCommRounds,))
        

        #set global discriminator and generator
        self.globalDiscriminator = globalDiscriminator.to(self.sDevice)
        self.globalGenerator = globalGenerator.to(self.sDevice)

        if self.sFedType == "SCAFFOLD":
            self.globalControlDiscr = get_zero_init(self.globalDiscriminator)
            self.globalControlGen = get_zero_init(self.globalGenerator)

        #set image transformers if cnn structure
        if self.sDiscrType == "CNN":
            self.Dtransformer = ImageTransformer(self.globalDiscriminator.iSides)
        if self.sGenType == "CNN":
            self.Gtransformer = ImageTransformer(self.globalGenerator.iSides)

        #set verbose
        self.bVerbose = bVerbose

    def set_client_data(self, fedGANDataset):
        """
        Goal: set client data, proportions, conditional sampler of data, get number of clients, set optimizers
        Input:
            - self          A FedDPGANOptimizer object
            - fedGANDataset A FedGANDataset object 
        """
        #set data
        ldf = fedGANDataset.ldf

        #set number of clients
        self.iC = fedGANDataset.iC
        self.liN = [ldf[c].shape[0] for c in range(self.iC)]
        self.vNoisyCounts = fedGANDataset.vNoisyCounts

        #to float 32
        #ldf = [ldf[c].astype(np.float32) for c in range(self.iC)]
        ldft = []
        for item in ldf:
           ldft.append(torch.tensor(item, device=self.sDevice).float()) #changed 
        self.ldf = ldft

        #get marg probs
        self.llOneWayMargProbs = fedGANDataset.llOneWayMargProbs
        self.vOneWayMargProbs = torch.tensor(np.concatenate(self.llOneWayMargProbs).astype(np.float32)).to(self.sDevice)
 
        #set steps
        self.lSteps = [0 for _ in range(self.iC)]
        self.lStepsInit = [0 for _ in range(self.iC)]

        #privacy things for clients
        self.ldMuUsedPerClient = [0.0 for _ in range(self.iC)]
        self.lbExhausted = [False for _ in range(self.iC)]

        #set local control covariates
        if self.sFedType == "SCAFFOLD":
            self.lClientControlDiscr = [get_zero_init(self.globalDiscriminator) for _ in range(self.iC)]
            self.lClientControlGen = [get_zero_init(self.globalGenerator) for _ in range(self.iC)]

        #set dictionary of samplers
        self.lSamplers = [Sampler( fedGANDataset.lCondDimPerVariable, fedGANDataset.ldfCond[c], self.rng, self.llOneWayMargProbs) for c in range(self.iC)]

        #set dataset proportions
        self.vNoisyFrac = torch.from_numpy(fedGANDataset.vNoisyFrac.astype(np.float32)).to(self.sDevice) #changed

        #set output info
        self.lOutputInfo = fedGANDataset.lOutputInfo

        #set dictionary of discriminators, initialized by global discriminator
        self.lClientDiscriminators = [Discriminator(iInputDim = self.globalDiscriminator.iInputDim,
                                                     sLoss = self.globalDiscriminator.sLoss, 
                                                     dLambda = self.globalDiscriminator.dLambda,
                                                     lSeq = self.globalDiscriminator.lSeq#, lSeqInfoLoss = self.globalDiscriminator.lSeqInfoLoss
                                                    ).to(self.sDevice) for _ in range(self.iC)]

        #set dictionary of generators, initialized by global generator
        self.lClientGenerators = [Generator(iDataDim = self.globalGenerator.iDataDim, 
                                            iRandomInputDim = self.globalGenerator.iRandomInputDim,
                                            sLoss = self.globalGenerator.sLoss,
                                            lSeq = self.globalGenerator.lSeq).to(self.sDevice) for _ in range(self.iC)]
        
        #set list of discriminator optimizers
        if self.sClientOpt == "SGD":
            self.lOptimizersDiscr = [torch.optim.SGD(self.lClientDiscriminators[c].parameters(), lr=self.dEtaDiscr) for c in range(self.iC)]
        elif self.sClientOpt == "Adam":
            self.lOptimizersDiscr = [torch.optim.Adam(self.lClientDiscriminators[c].parameters(), 
                                                    lr=self.dEtaDiscr,
                                                    eps = self.dNumStab,
                                                    weight_decay = self.dWeightDecayDiscr,
                                                    betas = self.tBetasDiscr) for c in range(self.iC)]

        #now make private
        for c in range(self.iC):
            self.lClientDiscriminators[c], self.lOptimizersDiscr[c] = privatise_model_optimizer(self.lClientDiscriminators[c], self.lOptimizersDiscr[c], dClipBound=self.dClipbound, dSigma=float(self.dSigma), batch_size = self.liN[c])
        self.globalDiscriminator._modules = self.lClientDiscriminators[0]._modules
        
        #set list of generator optimizers

        if self.sClientOpt == "SGD":
            self.lOptimizersGen = [torch.optim.SGD(self.lClientGenerators[c].parameters(), lr=self.dEtaGen) for c in range(self.iC)]
        elif self.sClientOpt == "Adam":
            self.lOptimizersGen = [torch.optim.Adam(self.lClientGenerators[c].parameters(), 
                                                    lr=self.dEtaGen,
                                                    eps = self.dNumStab,
                                                    weight_decay = self.dWeightDecayGen,
                                                    betas = self.tBetasGen) for c in range(self.iC)]

        
    def update_privacy_params_full_batch(self, c):
        """
        Goal: Update the privacy parameters after a full batch iteration for the discriminator
        Input:
            - A FedDPGANOptimizer object
            - c client number
        """
        #update privacy parameter (parallel composition)
        dMuUsed =  self.dClipbound / self.dSigma
        self.ldMuUsedPerClient[c] = np.sqrt(self.ldMuUsedPerClient[c]**2 + dMuUsed**2)

        #update privacy budget used
        self.dMuUsed = max(self.ldMuUsedPerClient)

        #check if enough budget for another update
        dMuFuture = np.sqrt(self.ldMuUsedPerClient[c]**2 + dMuUsed**2)
        if dMuFuture > self.dMuTrain:
            self.lbExhausted[c] = True
        if all(self.lbExhausted):
            self.bExhausted = True  

    #added
    def gan_only_step(self, c, iRows = 5000, bUniform = False):
        """
        Goal: update only generator
        Input: 
            - self      A FedDPGANOptimizer object
            - c         client number
            - iRows     integer number of rows to generate for GAN
            - bUniform  boolean, uniform sampling of conditions or using the estimated probabilities by the initial model.
        """
        #get sampler
        sampler = self.lSamplers[c]

        for _ in range(self.iGenUpdatesPerStep):

            #get noise z
            mNoise = torch.randn(size = (iRows, self.iNoiseDim), device = self.sDevice) 

            #get conditional vector
            mCond, mMask, vVariableSelection, vModeSelection = sampler.sample_cond_matrix_dp(sDevice= self.sDevice, bUniform = bUniform, iRows = iRows)#.astype(np.float32)  #changed
            #mCond = mCond.to(self.sDevice) #changed
            #mMask = mMask.to(self.sDevice) #changed

            #add conditional vector to noise
            mNoise = torch.cat([mNoise, mCond], dim = 1)
            if self.sGenType == "CNN":
                mNoise = mNoise.view(iRows, self.iNoiseDim + sampler.iTotalVectorLength, 1, 1)

            #set gradient to zero'
            self.lOptimizersGen[c].zero_grad()

            #generate synthetic data
            fake = self.lClientGenerators[c](mNoise)
            if self.sGenType == "CNN":
                fake = self.Gtransformer.inverse_transform(fake)
            fakeact = apply_activate(fake, self.lOutputInfo)

            #concatenate with conditon vectors
            fakeCat = torch.cat([fakeact, mCond], dim = 1)

            #transofrm to image domain if applicable
            if self.sDiscrType == "CNN":
                fakeCat  = self.Dtransformer.transform(fakeCat)

            #get conditional loss
            dCondLoss = cond_loss(fake, self.lOutputInfo, mCond, mMask) 

            #get mode loss
            dModeLoss = mode_loss(fakeact, self.lOutputInfo, self.vOneWayMargProbs, mCond, mMask)
            
            #backprop
            if self.bModeLoss:
                dTotalLoss = dCondLoss + dModeLoss
            else:
                dTotalLoss = dCondLoss
            dTotalLoss.backward()

            #scaffold for updating
            if self.sFedType == "SCAFFOLD":
                #get gradient
                diGrad = get_grad_dict(self.lClientGenerators[c])
                #add global control and substract local control g + c - c_i
                diGrad = {k: diGrad.get(k, 0) + self.globalControlGen.get(k, 0) - self.lClientControlGen[c].get(k,0) for k in set(diGrad)}
                #set new gradient
                self.lClientGenerators[c] = set_gradients_net(self.lClientGenerators[c], diGrad)

            #update
            self.lOptimizersGen[c].step()

        return dCondLoss, dModeLoss

    def full_batch_step(self, c):
        """
        Goal: perform one full-batch optimization step for a single client
        Input: 
            - self      A FedDPGANOptimizer object
            - c         client number
        Output
            - dLoss     Negative Critic (discriminator) loss 
        """

        #get sampler
        sampler = self.lSamplers[c]

        i = 0
        while ((i < self.iDiscrUpdatesPerStep) and (self.lbExhausted[c] is False)):
            #update count
            i = i + 1
            #sample real data
            vInd = np.arange(self.liN[c])

            #potential sampling for further purposes
            vSampled= self.rng.binomial(n=1, p = 1.0, size = len(vInd))
            vInd = vInd[vSampled == 1]
            iBatchSize = len(vInd)
            mReal = self.ldf[c][vInd,:]
            
            #get noise z
            mNoise = torch.randn(size = (iBatchSize, self.iNoiseDim), device = self.sDevice)

            #get conditional vector
            mCond, mMask, vVariableSelection, vModeSelection = sampler.sample_cond_matrix(vBatchIndices = vInd, sDevice = self.sDevice)

            #add conditional vector to noise
            mNoise = torch.cat([mNoise, mCond], dim = 1)
            if self.sGenType == "CNN":
                mNoise = mNoise.view(iBatchSize, self.iNoiseDim + sampler.iTotalVectorLength, 1, 1)

            #generate synthatic data
            fake = self.lClientGenerators[c](mNoise)
            if self.sGenType == "CNN":
                fake = self.Gtransformer.inverse_transform(fake)
            fakeact = apply_activate(fake, self.lOutputInfo)

            #print(torch.Tensor(mReal).to(self.sDevice))
            fake = torch.cat([fakeact, mCond], dim = 1)
            real = torch.cat([mReal, mCond], dim = 1)

            #transform real and synthetic data to image domain if cnn
            if self.sDiscrType == "CNN":
                fake = self.Dtransformer.transform(fake)
                real = self.Dtransformer.transform(real)

            #set gradients to zero
            self.lOptimizersDiscr[c].zero_grad()

            if self.sLoss == "cross-entropy":

                #get probabilities
                vProbReal = self.lClientDiscriminators[c](real)
                vProbFake = self.lClientDiscriminators[c](fake)

                #get loss
                dDiscrLoss = (-(torch.log(vProbReal + 1e-4).mean()) - (torch.log(1. - vProbFake + 1e-4).mean()))
                dDiscrLossNoPenal = dDiscrLoss

                #get gradients
                dDiscrLoss.backward()

                #adjust gradients using scaffold
                if self.sFedType == "SCAFFOLD":
                    #get gradient
                    diGrad = get_grad_dict(self.lClientDiscriminators[c])
                    #add global control and substract local control g + c - c_i
                    diGrad = {k: diGrad.get(k, 0) + self.globalControlDiscr.get(k, 0) - self.lClientControlDiscr[c].get(k,0) for k in set(diGrad)}
                    #set new gradient
                    self.lClientDiscriminators[c] = set_gradients_net(self.lClientDiscriminators[c], diGrad)
                #get update
                self.lOptimizersDiscr[c].step()

            elif self.sLoss == "WGAN-GP":
                #dOne = torch.tensor(1, dtype = torch.float).to(self.sDevice)
                #dMinOne = -1*dOne

                #get estimates
                vEstReal = self.lClientDiscriminators[c](real)
                vEstFake = self.lClientDiscriminators[c](fake)

                #get means
                dEstRealMean = torch.mean(vEstReal)
                dEstFakeMean = torch.mean(vEstFake)

                #get gradient penal - wgan option does not work...
                #self.lClientDiscriminators[c].disable_hooks()
                #dGradPen = calc_gradient_penalty(self.lClientDiscriminators[c], real, fake, dLambda =  self.globalDiscriminator.dLambda, rng = self.rng, sDevice =self.sDevice)
                
                #print(dGradPen)
                #self.lOptimizersDiscr[c].disable_hooks() 
                #dGradPen.backward()
                #self.lOptimizersDiscr[c].step()
                #self.lOptimizersDiscr[c].zero_grad()
                #self.lOptimizersDiscr[c].enable_hooks()
                #dGradPen.backward()
                #self.lClientDiscriminators[c].enable_hooks()

                #get loss
                dDiscrLossNoPenal =  - dEstRealMean + dEstFakeMean 
                dDiscrLoss = dDiscrLossNoPenal #+ dGradPen

                #get gradients
                dDiscrLoss.backward()

                #adjust gradients using scaffold
                if self.sFedType == "SCAFFOLD":
                    #get gradient
                    diGrad = get_grad_dict(self.lClientDiscriminators[c])
                    #add global control and substract local control g + c - c_i
                    diGrad = {k: diGrad.get(k, 0) + self.globalControlDiscr.get(k, 0) - self.lClientControlDiscr[c].get(k,0) for k in set(diGrad)}
                    #set new gradient
                    self.lClientDiscriminators[c] = set_gradients_net(self.lClientDiscriminators[c], diGrad)   
                #get update
                self.lOptimizersDiscr[c].step()

                #if no gradient penalty, we need to clip (original wgan):
                for p in self.lClientDiscriminators[c].parameters():
                    p.data.clamp_(-.01, .01)

            #update privacy params
            self.update_privacy_params_full_batch(c)
            

        #update generator
        for _ in range(self.iGenUpdatesPerStep):

            #get noise z
            iBatchSize = self.iB
            mNoise = torch.randn(size = (iBatchSize, self.iNoiseDim), device = self.sDevice) #torch.from_numpy(self.rng.standard_normal(size = (iBatchSize, self.iNoiseDim), dtype = np.float32)).to(self.sDevice) #changed

            #get conditional vector
            mCond, mMask, vVariableSelection, vModeSelection = sampler.sample_cond_matrix_dp(sDevice= self.sDevice, bUniform = False, iRows = iBatchSize)
            #mCond = mCond.to(self.sDevice) #changed
            #mMask = mMask.to(self.sDevice) #changed

            #add conditional vector to noise
            mNoise = torch.cat([mNoise, mCond], dim = 1)
            if self.sGenType == "CNN":
                mNoise = mNoise.view(iBatchSize, self.iNoiseDim + sampler.iTotalVectorLength, 1, 1)

            #set gradient to zero'
            self.lOptimizersGen[c].zero_grad()

            #generate synthetic data
            fake = self.lClientGenerators[c](mNoise)
            if self.sGenType == "CNN":
                fake = self.Gtransformer.inverse_transform(fake)
            fakeact = apply_activate(fake, self.lOutputInfo)

            #get mode loss
            if self.bModeLoss:
                dModeLoss = mode_loss(fakeact, self.lOutputInfo, self.vOneWayMargProbs, mCond, mMask)
            else:
                dModeLoss = 0.0

            #number of observations for which we backpropagate is smaller to save time
            iBatchSize2 = self.iNGenSample
            fake = fake[0:iBatchSize2,:]
            fakeact = fakeact[0:iBatchSize2,:]
            mCond = mCond[0:iBatchSize2,:]
            mMask = mMask[0:iBatchSize2,:]

            #get conditional loss
            dCondLoss = cond_loss(fake, self.lOutputInfo, mCond, mMask)

            #concatenate with conditon vectors
            fakeCat = torch.cat([fakeact, mCond], dim = 1)

            #transofrm to image domain if applicable
            if self.sDiscrType == "CNN":
                fakeCat  = self.Dtransformer.transform(fakeCat)

            if self.sLoss == "cross-entropy":

                #get probabilities
                vProbFake = self.lClientDiscriminators[c](fakeCat)

                #get generator loss
                dGenLoss = torch.log(1. - vProbFake + 1e-4).mean() 
                if self.bModeLoss:
                    dGenLossTot = dGenLoss  + dModeLoss + dCondLoss
                else:
                    dGenLossTot = dGenLoss  + dCondLoss

                #backprop
                dGenLossTot.backward() #added

                #scaffold for updating
                if self.sFedType == "SCAFFOLD":
                    #get gradient
                    diGrad = get_grad_dict(self.lClientGenerators[c])
                    #add global control and substract local control g + c - c_i
                    diGrad = {k: diGrad.get(k, 0) + self.globalControlGen.get(k, 0) - self.lClientControlGen[c].get(k,0) for k in set(diGrad)}
                    #set new gradient
                    self.lClientGenerators[c] = set_gradients_net(self.lClientGenerators[c], diGrad)

                #set step
                self.lOptimizersGen[c].step()

            elif self.sLoss == "WGAN-GP":

                #get estimates
                vEstFake = self.lClientDiscriminators[c](real)

                #get gen loss
                dGenLoss = -torch.mean(vEstFake) 
                dGenLossTotal = dGenLoss + dCondLoss + dModeLoss

                #backprop
                dGenLossTotal.backward()

                #scaffold for updating
                if self.sFedType == "SCAFFOLD":
                    #get gradient
                    diGrad = get_grad_dict(self.lClientGenerators[c])
                    #add global control and substract local control g + c - c_i
                    diGrad = {k: diGrad.get(k, 0) + self.globalControlGen.get(k, 0) - self.lClientControlGen[c].get(k,0) for k in set(diGrad)}
                    #set new gradient
                    self.lClientGenerators[c] = set_gradients_net(self.lClientGenerators[c], diGrad)

                #update
                self.lOptimizersGen[c].step()

        # return loss
        return dDiscrLossNoPenal, dGenLoss, dCondLoss, dModeLoss #added


    def round(self, c, init = False):
        """
        Goal: perform one optimization round for one client (client c)
        Input:
            - self      A FedDPCTGAN object
            - c         integer client number
        """
        if init:
            for step in range(self.iStepsPerRound):
                tLosses = self.gan_only_step(c)
                self.vCondLossInit[self.lStepsInit[c]] = tLosses[0]
                self.vModeLossInit[self.lStepsInit[c]] = tLosses[1]
                self.lStepsInit[c] += 1
        else:
            #set steps in communication round
            for step in range(self.iStepsPerRound):
                #apply step if not exhausted
                if self.lbExhausted[c] is False:

                    tLosses = self.full_batch_step(c)
                    self.vDiscrLoss[self.lSteps[c]] += tLosses[0]
                    self.vGenLoss[self.lSteps[c]] += tLosses[1]
                    self.vCondLoss[self.lSteps[c]] += tLosses[2] #added
                    self.vModeLoss[self.lSteps[c]] += tLosses[3]
                    #if self.bVerbose:
                    #    print(f"Client: {c}. Communication Round: {self.iRounds}, Step: {self.lSteps[c] + 1}, Squared train priv budget prop used: {round(100*(self.dMuUsed/self.dMuTrain)**2,2)}%, loss: {round(self.vDiscrLoss[self.lSteps[c]], 4)}")
                    self.lSteps[c] += 1

        #update scaffold control
        if self.sFedType == "SCAFFOLD":
            #get global parameters
            diGlobalParamDiscr = get_param_dict(self.globalDiscriminator)
            diGlobalParamGen = get_param_dict(self.globalGenerator)

            #get local parameters
            diLocalParamDiscr = get_param_dict(self.lClientDiscriminators[c])
            diLocalParamGen = get_param_dict(self.lClientGenerators[c])

            #update control states
            self.lClientControlDiscr[c] = {k: self.lClientControlDiscr[c].get(k, 0) - self.globalControlDiscr.get(k, 0) + (1.0/ (self.iDiscrUpdatesPerStep * self.iStepsPerRound * self.dEtaDiscr))*(diGlobalParamDiscr.get(k,0) - diLocalParamDiscr.get(k,0)) for k in set(self.globalControlDiscr)}
            self.lClientControlGen[c] = {k: self.lClientControlGen[c].get(k,0) - self.globalControlGen.get(k, 0) + (1.0/ (self.iGenUpdatesPerStep * self.iStepsPerRound * self.dEtaGen))*(diGlobalParamGen.get(k,0) - diLocalParamGen.get(k,0)) for k in set(self.globalControlGen)}


    def server_to_clients(self):
        """
        Goal: set global weights to client weight
        Input:  
            - self  A FedDPCTGAN object
        """

        #with torch.no_grad()???

        #get global weights 
        diGlobalGen = self.globalGenerator.state_dict()
        diGlobalDiscr = self.globalDiscriminator.state_dict()

        #set for all clients
        for c in range(self.iC):
            self.lClientGenerators[c].load_state_dict(diGlobalGen)
            self.lClientDiscriminators[c].load_state_dict(diGlobalDiscr)

    def clients_to_server(self): 
        """
        Goal: Get averaged weights, send to server and update server
        Input:
            - self A FedDPCTGAN object
        """

        #with torch.no_grad()???

        #get all weights client models
        lWeightDictDiscr = [self.lClientDiscriminators[c].state_dict() for c in range(self.iC)]
        lWeightDictGen = [self.lClientGenerators[c].state_dict() for c in range(self.iC)]

        # get new parameters discriminator
        diStateNewDiscr = self.globalDiscriminator.state_dict() 

        for key in diStateNewDiscr:
            diStateNewDiscr[key] = torch.zeros(size = diStateNewDiscr[key].size(), device = self.sDevice) #changed
            for c in range(self.iC):
                diStateNewDiscr[key] += self.vNoisyFrac[c] * lWeightDictDiscr[c][key]
        #set global parameters
        self.globalDiscriminator.load_state_dict(diStateNewDiscr)

        # get new parameters generat0r
        diStateNewGen = self.globalGenerator.state_dict() 
        for key in diStateNewGen:
            diStateNewGen[key] = torch.zeros(size = diStateNewGen[key].size(), device = self.sDevice) #changed
            for c in range(self.iC):
                diStateNewGen[key] += self.vNoisyFrac[c] * lWeightDictGen[c][key]
        #set global parameters
        self.globalGenerator.load_state_dict(diStateNewGen)

        #get new global control variables for scaffold
        if self.sFedType == "SCAFFOLD":
            self.globalControlDiscr = get_average_dict(self.lClientControlDiscr)
            self.globalControlGen = get_average_dict(self.lClientControlGen)

    def communication_round_all_clients(self, init = False):
        """
        Goal: send global params to client, perform round for every/subset of client, update local client params to global and average
        Input:
            - self A FedDPCTGAN object
            - init boolean is it the inital gan optimization
        """

        #update count
        self.iRounds +=1

        #sent global params to clients
        self.server_to_clients()

        #init only gan updates for conditional loss
        if init:
            for c in range(self.iC):
                self.round(c, init = True)
        else:
            #update client parameters
            for c in range(self.iC):
                self.round(c)

        #average 
        self.clients_to_server()
        
    def fit(self):
        """
        Goal: fit the entire model
        Input:
            - self A FedDPCTGAN object
        Output:
            - diResults
        """
        #initialization
        for i in range(self.iInitCommRounds):
            self.communication_round_all_clients(init = True)
            if self.bVerbose:
                print(f"init round: {i}, Steps: {self.lStepsInit[0]}, CondLoss: {round(self.vCondLossInit[self.lStepsInit[0] - 1], 4)}, ModeLoss: {round(self.vModeLossInit[self.lStepsInit[0] - 1], 4)}")
            
        #loop over communication rounds
        while not self.bExhausted and self.iSteps > 0:
            self.communication_round_all_clients()
            if self.bVerbose:
                print(f"Communication Round: {self.iRounds}, Steps: {self.lSteps[0]}, Squared train priv budget prop used: {round(100*(self.dMuUsed/self.dMuTrain)**2,2)}, GenLoss: {round(self.vGenLoss[self.lSteps[0]- 1], 4)}, CondLoss =  {round(self.vCondLoss[self.lSteps[0]- 1], 4)}, ModeLoss: {round(self.vModeLoss[self.lSteps[0]- 1], 4)}, DiscrLoss: {round(self.vDiscrLoss[self.lSteps[0]-1], 4)}") #added
       
        #save results
        diResults = dict()
        diResults['rounds'] = self.iRounds
        diResults['generator'] = self.globalGenerator
        diResults['discriminator'] = self.globalDiscriminator
        diResults['steps'] = self.iSteps
        diResults['discriminator_loss_per_step'] = self.vDiscrLoss
        diResults['generator_loss_per_step'] = self.vGenLoss
        diResults['conditional_loss_per_step'] = self.vCondLoss #added
        diResults['mode_loss_per_step'] = self.vModeLoss
        #return global models
        return diResults
        
class Discriminator(Module):
    """
    General Discriminator
    """       

    def __init__(self, iInputDim, sLoss = "WGAN-GP", dLambda = 10, lSeq = None, lSeqInfoLoss = None):
        """
        Goal: Constructor of discriminator object
        Input:
            - iInputDim     integer, dimension of input data = dim(r) + dim(c)
            - sLoss         string, loss to use for the discriminator
        Set:
            - self          A discriminator (or critic) object
        """
        #super construct
        super().__init__()

        #set input dimension
        self.iInputDim = iInputDim

        #set seq seqinfoloss
        self.lSeq = lSeq 
        #self.lSeqInfoLoss = lSeqInfoLoss

        #set loss (and lambda)
        if sLoss == "WGAN-GP":
            self.sLoss = "WGAN-GP"
            self.dLambda = dLambda
        elif sLoss == "cross-entropy":
            self.sLoss = "cross-entropy"
            self.dLambda = None
        else: 
            raise ValueError(f"Loss {sLoss} not available, use WGAN-GP or cross-entropy")

    def forward(self, input):
        return self.lSeq(input)

def calc_gradient_penalty(discriminator, mReal, mFake, dLambda, rng, sDevice = "cpu"):
    """
    Goal: calculate gradient penalty for Wasserstein gan (WGAN-GP)
    Input:
        - self      A discriminator object
        - mReal     Matrix of real data
        - mFake     Matrix of fake data
        - sDevice   String, device on which we should compute the gradient penalty  
    Output:
        - dGradientPenalty  Double, value of gradient penalty     
    """
    #generate alpha
    #vAlpha = torch.rand(mReal.size(0), 1, 1, device = sDevice)
    vAlpha = torch.from_numpy(rng.random(size = (mReal.size(0),1,1), dtype = np.float32)).to(sDevice) 
    mAlpha = vAlpha.repeat(1,1,mReal.size(1))
    mAlpha = mAlpha.view(-1, mReal.size(1))

    #interpolation
    mInterpolate = mAlpha * mReal + ((1 - mAlpha) * mFake)

    #get discrimator values
    vDiscrInterpolates = discriminator(mInterpolate)

    #get gradients
    #discriminator.enable_hooks() 
    #discriminator.disable_hooks() 
    mGradients = torch.autograd.grad(
        outputs = vDiscrInterpolates,
        inputs = mInterpolate,
        grad_outputs = torch.ones(vDiscrInterpolates.size(), device=sDevice),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]
    #discriminator.enable_hooks() 
    #discriminator.disable_hooks() 

    #get penalty
    dGradientPenalty =  ((mGradients.view(-1,mReal.size(1)).norm(2, dim=1) -1)**2).mean()*dLambda

    return dGradientPenalty

# custom for calcuate grad_sample for multiple loss.backward()
def _custom_create_or_extend_grad_sample(
    param: torch.tensor, grad_sample: torch.tensor, batch_dim: int #changed
) -> None:
    """
    Create a 'grad_sample' attribute in the given parameter, or accumulate it
    if the 'grad_sample' attribute already exists.
    This custom code will not work when using optimizer.virtual_step()
    """

    if hasattr(param, "grad_sample"):
        param.grad_sample = param.grad_sample + grad_sample
        # param.grad_sample = torch.cat((param.grad_sample, grad_sample), batch_dim)
    else:
        param.grad_sample = grad_sample

class CnnDiscriminator(Discriminator):
    """
    Discriminator of the GAN with Cnn structure
    """
    
    def __init__(self, iInputDim, iChannels = 64, sLoss = "WGAN-GP", dLambda = 10):
        """
        Goal: constructor of convolutional neural network discriminator (as in DCGAN)
        Input: 
            - iInputDim     integer, dimension of input = dim(r) + dim(c)
            - sLoss         string, name of loss used, either WGAN-GP or cross-entropy
            - dLambda       double, gradient penalty (only applicable for WGAN-GP)
            - tHiddenNodes  tuple, number of hidden nodes per layer.
        Set:
            -self           a CnnDiscriminator object
        """
        #super constructor
        super().__init__(iInputDim = iInputDim, sLoss = sLoss, dLambda = dLambda)

        #set dimensions
        self.iSides = int(np.ceil(np.sqrt(iInputDim))) 

        #get hidden layers. the number of channels should increase a factor two, while height/wieght decreases with factor two
        lLayerDims = [(1, self.iSides), (iChannels, self.iSides//2)]
        while lLayerDims[-1][1] >3 and len(lLayerDims) <4:
            lLayerDims.append((lLayerDims[-1][1] *2, lLayerDims[-1][1] // 2))
        
        #construct layers
        lLayersD = []
        for prev, curr in zip(lLayerDims, lLayerDims[1:]):
            if self.sLoss == "WGAN-GP":
                lLayersD += [
                    Conv2d(prev[0], curr[0], 4, 2, 1, bias = True),
                    GroupNorm(1, curr[0]),
                    LeakyReLU(0.2, inplace = True)
                ]
            elif self.sLoss == "cross-entropy":
                lLayersD += [
                    Conv2d(prev[0], curr[0], 4, 2, 1, bias = False),
                    BatchNorm2d(curr[0]),
                    LeakyReLU(0.2, inplace = True)
                ]
        #last layer
        if self.sLoss == "WGAN-GP":
            lLayersD += [Conv2d(lLayerDims[-1][0], 1, lLayerDims[-1][1], 1, 0)]
        elif self.sLoss == "cross-entropy":
            lLayersD += [Conv2d(lLayerDims[-1][0], 1, lLayerDims[-1][1], 1, 0), Sigmoid()]

        #set structure
        self.lSeq = Sequential(*lLayersD)
        #if self.sLoss == "WGAN-GP":
        #    self.lSeqInfoLoss = Sequential(*lLayersD[:len(lLayersD) - 1])
        #elif self.sLoss == "cross-entropy":
        #    self.lSeqInfoLoss = Sequential(*lLayersD[:len(lLayersD) - 2])
   

class MlpDiscriminator(Discriminator):
    """
    Discriminator of the GAN with fully connected Mlp layers
    """
    
    def __init__(self, iInputDim, sLoss = "WGAN-GP", dLambda = 10, tHiddenNodes = (64,8) ): #256, 256(25,25)
        """
        Goal: constructor of fully connected discriminator (MLP)
        Input: 
            - iInputDim     integer, dimension of input = dim(r) + dim(c)
            - sLoss         string, name of loss used, either WGAN-GP or cross-entropy
            - dLambda       double, gradient penalty (only applicable for WGAN-GP)
            - tHiddenNodes  tuple, number of hidden nodes per layer.
        Set:
            -self           a MlpDiscriminator object
        """
         #super constructor
        super().__init__(iInputDim = iInputDim, sLoss = sLoss, dLambda = dLambda)

        #set dimensions
        lLayersD = []
        iDim = self.iInputDim
        for iHiddenNodes in tHiddenNodes:
            lLayersD += [Linear(iDim, iHiddenNodes),
                        LayerNorm(iHiddenNodes),
                        LeakyReLU(0.2),
                        Dropout(0.5)]
            iDim = iHiddenNodes
        
        #last layer
        if self.sLoss == "WGAN-GP":
            lLayersD += [Linear(iDim, 1)]
        elif self.sLoss == "cross-entropy":
            lLayersD += [Linear(iDim, 1), Sigmoid()]

        #set structure
        self.lSeq = Sequential(*lLayersD)
        #if self.sLoss == "WGAN-GP":
        #    self.lSeqInfoLoss = Sequential(*lLayersD[:len(lLayersD) - 1])
        #elif self.sLoss == "cross-entropy":
        #    self.lSeqInfoLoss = Sequential(*lLayersD[:len(lLayersD) - 2])


class Residual(Module):
    def __init__(self, i, o):
        super().__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)


class Generator(Module):
    """
    Generator of the GAN
    """
    def __init__(self, iDataDim, iRandomInputDim, sLoss = "WGAN-GP", lSeq = None):
        """
        Goal: super constructor of generator
        Input:
            - iDataDim              integer, dimension of data (generator output) = dim(r)
            - iRandomInputDims      integer, dimension of random input (generator input) = dim(z) + dim(c)
            - sLoss                 string, type of gan, either wgan-gp or cross-entropy
        Set:
            - self                  A generator object
        """

        #super construct
        super().__init__()

        #set values
        self.iDataDim = iDataDim
        self.iRandomInputDim = iRandomInputDim
        self.lSeq = lSeq

        #set loss (and lambda)
        if sLoss == "WGAN-GP":
            self.sLoss = "WGAN-GP"
        elif sLoss == "cross-entropy":
            self.sLoss = "cross-entropy"
        else: 
            raise ValueError(f"Loss {sLoss} not available, use WGAN-GP or cross-entropy")

    def forward(self, input):
        return self.lSeq(input)


class CnnGenerator(Generator):
    """
    Generator of the GAN
    """
    def __init__(self, iDataDim, iRandomInputDim, iChannels = 64,  sLoss = "WGAN-GP"):
        """
        Goal: Construct a CNN Generator
        Input:
            - iDataDim              integer, dimension of data (generator output) = dim(r)
            - iRandomInputDims      integer, dimension of random input (generator input) = dim(z) + dim(c)
            - iChannels             integer, number of channels for the cnn
            - sLoss                 string, type of gan, either wgan-gp or cross-entropy
        Set:
            - self                  A Cnn Generator object
        """
        #construct super
        super().__init__(iDataDim = iDataDim, iRandomInputDim = iRandomInputDim, sLoss = sLoss)

        #determine sides
        self.iSides = int(np.ceil(np.sqrt(iDataDim))) 

        #compute layer dimensions
        lLayerDims = [(1,self.iSides), (iChannels, self.iSides //2)]
        while(lLayerDims[-1][1] >3 and len(lLayerDims) < 4):
            lLayerDims.append((lLayerDims[-1][0] * 2, lLayerDims[-1][1 // 2]))

        #construct layers
        lLayersG = [ ConvTranspose2d( iRandomInputDim, lLayerDims[-1][0],lLayerDims[-1][1], 1, 0, output_padding=0, bias = False)] #sqrt(ceil(iRandomInputDim))??
        for prev, curr in zip(reversed(lLayerDims), reversed(lLayerDims[:-1])):
            lLayersG += [BatchNorm2d(prev[0]), ReLU(True), ConvTranspose2d(prev[0], curr[0], 4,2,1, output_padding=0, bias = True)] #batchnormalization can be done here, but maybe CTABGAN actually uses layernorm, not sure!

        #set sequence
        self.lSeq = Sequential(*lLayersG)

    def forward(self, input):
        return self.lSeq(input)


class MlpGenerator(Generator):
    """
    Class for a multilayer perceptron/ fully connected generator network
    """

    def __init__(self, iDataDim, iRandomInputDim, tHiddenNodes = (128, 128), sLoss = "WGAN-GP"): #256, 256
        """
        Goal: Construct a MLP Generator
        Input:
            - iDataDim              integer, dimension of data (generator output) = dim(r)
            - iRandomInputDims      integer, dimension of random input (generator input) = dim(z) + dim(c)
            - tHiddenNodes          tuple, dimensions of hidden nodes (e.g. (32,16) is a 2 hidden layer NN with 32 and 16 nodes)
            - sLoss                 string, type of gan, either wgan-gp or cross-entropy
        Set:
            - self                  A MLP generator object
        """
        #construct super
        super().__init__(iDataDim = iDataDim, iRandomInputDim = iRandomInputDim, sLoss = sLoss)

        #construct network
        iTempDim = self.iRandomInputDim
        lLayersG = []
        for iHiddenNodes in tHiddenNodes:
            lLayersG += [ Residual(iTempDim, iHiddenNodes)] #residual layer
            iTempDim += iHiddenNodes

        #last layer
        lLayersG.append(Linear(iTempDim, iDataDim))

        #set sequence
        self.lSeq = Sequential(*lLayersG)

class Sampler():
    """
    Conditional vector sampler for one dataset
    """
    def __init__(self, lCondDimPerVariable, mConditional, rng, llOneWayMargProbs):
        """
        Goal: Constructor of a sampler object
        Input:
            - lCondDimPerVariable       list with dimension of conditional vector per variable
            - mConditional              matrix with valid condition per variable for all instances
            - rng                       A random number generator
            - llOneWayMargProbs         list of list of marginal probabilities for DP sampling
        Set:
            - self                      A sampler object
        """
        #get conditional values
        self.iD = len(lCondDimPerVariable)
        self.iN = mConditional.shape[0]
        self.mConditional = mConditional.astype(np.int32)
        self.lCondDimPerVariable = lCondDimPerVariable
        self.lCumCondDimPerVariable = np.cumsum([0] + lCondDimPerVariable[:-1])
        self.iTotalVectorLength = sum(lCondDimPerVariable)
        self.rng = rng
        self.llOneWayMargProbs = llOneWayMargProbs

    def sample_cond_matrix(self, sDevice, vBatchIndices = None):
        """
        Goal: sample for a batch the conditional vectors and put in a matrix
        Input:
            - self              A sampler object
            - vBatchIndices     vector of indices of batches. If none full batch
        Output:
            - mCond               A sampled conditional vector
            - mMask               matrix of mask
            - vVariableSelection  vector that selects the variable
            - vModeSelection      vector of mode selection
        """  
        #check if batch is non
        if vBatchIndices is None:
            vBatchIndices = np.arange(self.iN)
        #get batch
        mCondBatch = self.mConditional[vBatchIndices,:]
        iBatchSize = len(vBatchIndices)

        #variable selection
        vVariableSelection = self.rng.integers(self.iD, size = iBatchSize)
        mMask = torch.zeros((iBatchSize, self.iD), device = sDevice)#.to(self.sDevice)#np.zeros((iBatchSize, self.iD)) #changed
        mMask[np.arange(iBatchSize), vVariableSelection] = 1  

        #fill conditional vector
        mCond = torch.zeros((iBatchSize, self.iTotalVectorLength), device = sDevice)#.to(self.sDevice)#np.zeros((iBatchSize, self.iTotalVectorLength), dtype = np.float32) #changed
        vAllIndices = np.arange(iBatchSize)
        vModeSelection = mCondBatch[vAllIndices, vVariableSelection]
        vConditions = np.array([self.lCumCondDimPerVariable[el] for el in vVariableSelection]) + vModeSelection

        mCond[vAllIndices, vConditions] = 1
        #for i in range(iBatchSize):
        #    #get random variable
        #    iCondition = self.lCumCondDimPerVariable[vVariableSelection[i]] + mCondBatch[i,vVariableSelection[i]]
        #    mCond[i, iCondition] = 1

        return mCond, mMask, vVariableSelection, vModeSelection

    #added
    def sample_cond_matrix_dp(self, sDevice, bUniform = False, iRows = 10000):
        """
        Goal: sample a conditional matrix in a DP way (Only use dp info of the dataset)
        Input:
            - sDevice       string of device to compute everything on
            - bUniform      boolean, uniform sampling of conditions or relative to estimated probs in initialization
            - iRows         integer, number of rows to sample
        Output
            - mCond               A sampled conditional vector
            - mMask               matrix of mask
        """

        #pick variable
        vVariableSel = self.rng.choice(np.arange(self.iD), size = iRows)
        mMask = torch.zeros((iRows, self.iD), device = sDevice)#.to(self.sDevice)#np.zeros((iBatchSize, self.iD)) #changed
        mMask[np.arange(iRows), vVariableSel] = 1  

        #fill conditional vector
        mCond = torch.zeros((iRows, self.iTotalVectorLength ), device = sDevice)
        vAllIndices = np.arange(iRows)
        vModeSelection = np.zeros((iRows,))
        lvValues = [np.arange(self.lCondDimPerVariable[d]) for d in range(self.iD)]

        #check if uniform sampling
        if bUniform:
            for i in range(iRows):
                iVarSel = vVariableSel[i]
                vModeSelection[i] = self.rng.choice(a = lvValues[iVarSel])
        else: 
            vP = self.llOneWayMargProbs
            for i in range(iRows):
                iVarSel = vVariableSel[i]
                vModeSelection[i] = self.rng.choice(a = lvValues[iVarSel], p = vP[iVarSel])
        vConditions = np.array([self.lCumCondDimPerVariable[el] for el in vVariableSel]) + vModeSelection
        mCond[vAllIndices, vConditions.astype(np.int32)] = 1

        #return
        return mCond, mMask, vVariableSel, vModeSelection

    


class FedDPCTGAN():
    """
    Goal: create a differentially private, federated CTABGAN model.
    """

    def __init__(self, dMu = 1.0, iSeed = 1234, rng = None, dQuadInitProp = 0.25, iClusters=10, dEps=0.005, bDiscretize = False, sLoss = "WGAN-GP",
                sStructure = "CTGAN", sFedType = "fedAvg", sClientOpt = "SGD", iNoiseDim = None,  dSigma = None, iSteps = None,
                dClipBound = 1.0, iDiscrUpdatesPerStep = 5, iGenUpdatesPerStep = 1, iStepsPerRound = 10, sDevice = "cpu", bVerbose = False, bModeLoss = True):
        """
        Goal: initialize training proceducre
        Input:
            - dMu                   double, privacy budget 
            - iSeed                 integer, seed for rng (supply either iSeed or rng)
            - rng                   a random number generator object (supply either iSeed or rng)
            - dQuadinitProp         double, (quadratic) privacy budget proportion for initialization
            - iClusters             integer, number of clusters/modes for variational gaussian model normalization
            - dEps                  double, min percentage of observations a mode should have to keep it
            - bDiscretize           boolean, do we discritize all variables or not
            - sLoss                 string, loss of GAN, either "WGAN-GP" for the wasserstein loss with gradient penalty,
                                    or "cross-entropy" for the default GAN
            - sStructure            sStructure, either "CTAB-GAN" for the generator and discriminator structure of the CTAB-GAN
                                    paper (CNN), or "CTGAN" for the structures of the CTGAN paper (MLP)
            - sFedType              string, federated learning type, either "fedAvg" for regular federated averging, or "SCAFFOLD"
            - sClientOpt            string, the optimizer used at client level. Either "SGD" (without momentum) or "Adam"
            - iNoiseDim             integer, dimension of the input noise for generating new samples
            - dSigma                double, standard deviation for the noise added to ensure differential privacy. Supply either
                                    dSigma or iSteps
            - iSteps                integer, number of steps we would like to take for optimization. Supply either dSigma or iSteps
            - dClipBound            double, clipping bound to ensure differential privacy. Default of 1 is "optimal"for WGAN-GP
            - iDiscrUpdatesPerStep  integer, number of discrimator updates per step
            - iGenUpdatesPerStep    integer, number of generator updates per step
            - iStepsPerRound        integer, number of step per communication round
            - sDevice               string, device (e.g. "cpu" or "cuda:0")
            - bVerbose              boolean, print intermediate verbose messages or not
            - bModeLoss             boolean, use mode loss?
        Set:
            - self          A FedDPCTGAN object
        
        """
        #set privacy budgets
        self.dMuUsed = 0.0
        self.dMuPreprocess = np.sqrt(dQuadInitProp)*dMu
        self.dMuTrain = np.sqrt(1-dQuadInitProp)*dMu

        #set preprocessing parameters
        self.iClusters = iClusters
        self.dEps = dEps
        self.bDiscretize = bDiscretize

        #get loss, structur, noise dimesnion, federated learning type, clientoptimizer
        self.sLoss = sLoss
        self.sStructure = sStructure
        self.iNoiseDim = iNoiseDim
        if self.iNoiseDim is None:
            if self.sStructure == "CTABGAN":
                self.iNoiseDim = 100
            elif self.sStructure == "CTGAN":
                self.iNoiseDim = 128
        self.sFedType = sFedType
        self.sClientOpt = sClientOpt

        #set rng
        self.iSeed = iSeed
        if rng is None:  
            if iSeed == None:
                self.rng = np.random.default_rng(1234)
            else:
                self.rng = np.random.default_rng(iSeed)
        else:
            self.rng = rng

        #set clip bound
        self.dClipBound = dClipBound
        
        #set sigma OR number of steps
        if dSigma is None:
            if iSteps is None:
                ValueError("Set either number of steps or sigma")
            else:
                self.iSteps = iSteps
                self.dSigma = None
        if dSigma is not None:
            if iSteps is not None:
                ValueError("Set either number of steps or sigma")
            else:
                self.dSigma = dSigma
                self.iSteps = None

        #steps discriminator, generator
        self.iDiscrUpdatesPerStes = iDiscrUpdatesPerStep 
        self.iGenUpdatesPerStep = iGenUpdatesPerStep 
        self.iStepsPerRound = iStepsPerRound

        #set device, verbose
        self.sDevice = sDevice 
        self.bVerbose = bVerbose
        self.bModeLoss = bModeLoss
        
        
    def fit(self, ldf, lDataTypes, diMinMaxInfo, diCatUniqueValuesInfo, delimiterInfo, diMixedModes = {}, lLogColumn = []):
        """
        Goal: fit the GAN 
        Input:
            - ldf                       list of pandas dataframes with training data
            - lDataTypes                list with datatypes of variables ("categorical", "numerical", "ordinal", "mixed")
            - diMinMaxInfo              dictionary with minimum and maximum values of non-categorical variables
            - diCatUniqueValuesInfo     dictionary with unique values of categorical variables
            - deliminterInfo            dictionary with information about the number of decimals of each variable
            - diMixedModes              dictionary with information on the modes of mixed variables (0.0)              
            - lLogColumn                list of columns that should be log-transformed
        Output
            - diResults                 dictionary, results of the fit, e.g., global discrimator, generator and other information
            - fedGANDataset             fedGANDataset, the FedGANDataset object created for fitting.
        """
        #set seed
        if self.sDevice == "cuda":
            torch.cuda.manual_seed(self.iSeed)
        elif self.sDevice == "cpu":
            torch.manual_seed(self.iSeed)


        #preprocessing of the data
        if self.bVerbose:
            print("start preprocessing")
        fedGANDataset = FedGANDataset(ldf = ldf, lDataTypes = lDataTypes, iClusters=self.iClusters, dEps=self.dEps, diMinMaxInfo = diMinMaxInfo,
                                     diCatUniqueValuesInfo = diCatUniqueValuesInfo, dMuPreprocess= self.dMuPreprocess, rng = self.rng, diMixedModes = diMixedModes, 
                                     bDiscretize = self.bDiscretize, lLogColumn=lLogColumn, delimiterInfo=delimiterInfo)
        fedGANDataset.transform_all_client_data()
        if self.bVerbose:
            print("end preprocessing")

        #get specific elements ctabgan vs ctgan
        if self.sStructure == "CTABGAN":
            dWeightDecayGen = 1e-5
            dWeightDecayDiscr = 1e-5
            dEtaGen = 2e-4 
            dEtaDiscr = 2e-4 * ((fedGANDataset.dNoisyTotal/ np.sqrt(fedGANDataset.iC) ) / 500) 
            tBetasGen = (0.5,0.9)
            tBetasDiscr = (0.5,0.9)
            dNumStabAdam = 1e-3

        elif self.sStructure == "CTGAN":
            dWeightDecayGen = 1e-6
            dWeightDecayDiscr = 1e-6
            dEtaGen = 2e-4
            dEtaDiscr = 2e-4 * ((fedGANDataset.dNoisyTotal / np.sqrt(fedGANDataset.iC) )/ 500) 
            tBetasGen = (0.5,0.9)
            tBetasDiscr = (0.5,0.9)
            dNumStabAdam = 1e-8

        #get diminsions
        iConditionalDim = fedGANDataset.iCondDim #dimension of conditional vector
        iRandomInputDim = iConditionalDim + self.iNoiseDim
        iDataDim = fedGANDataset.iOutputDim
        iInputDim = iDataDim + iConditionalDim

        #construct the generator and discriminator networks
        if self.sStructure == "CTABGAN":
            self.globalDiscriminator = CnnDiscriminator(iInputDim = iInputDim, sLoss = self.sLoss)
            self.sDiscrType = "CNN"
        elif self.sStructure == "CTGAN":
            self.globalDiscriminator = MlpDiscriminator(iInputDim = iInputDim, sLoss = self.sLoss)
            self.sDiscrType = "MLP"
        else:
            raise ValueError("define own discriminator or set one of default structures CTABGAN (DCGAN)/ CTGAN (MLP)")
        
        if self.sStructure == "CTABGAN":
            self.globalGenerator = CnnGenerator(iDataDim = iDataDim, iRandomInputDim = iRandomInputDim, sLoss = self.sLoss)
            self.sGenType = "CNN"
        elif self.sStructure == "CTGAN":
            self.globalGenerator = MlpGenerator(iDataDim = iDataDim, iRandomInputDim = iRandomInputDim, sLoss = self.sLoss)
            self.sGenType = "MLP"
        else:
            raise ValueError("define own generator or set one of default structures CTABGAN (DCGAN)/ CTGAN (MLP)")

        if self.sDiscrType == "CNN":
            self.Dtransformer = ImageTransformer(self.globalDiscriminator.iSides)
        if self.sGenType == "CNN":
            self.Gtransformer = ImageTransformer(self.globalGenerator.iSides)

        #define optimizer
        if self.bVerbose:
            print("prepare optimizer")
        optimizer = FedDPGANOptimizer(self.globalDiscriminator, self.globalGenerator, iNoiseDim = self.iNoiseDim,
                 iDiscrUpdatesPerStep = self.iDiscrUpdatesPerStes, iGenUpdatesPerStep = self.iGenUpdatesPerStep,
                 iStepsPerRound = self.iStepsPerRound, dSigma = self.dSigma, iSteps = self.iSteps, rng = self.rng, 
                 sFedType = self.sFedType, sClientOpt = self.sClientOpt, dMuTrain = self.dMuTrain, dClipBound = self.dClipBound, 
                 dEtaDiscr = dEtaDiscr, dEtaGen = dEtaGen, tBetasGen = tBetasGen, tBetasDiscr = tBetasDiscr,
                 dWeightDecayGen = dWeightDecayGen, dWeightDecayDiscr = dWeightDecayDiscr, dNumStabAdam = dNumStabAdam,
                 sLoss = self.sLoss, sDevice = self.sDevice, bVerbose=self.bVerbose, bModeLoss=self.bModeLoss)
        optimizer.set_client_data(fedGANDataset)

        #fit
        if self.bVerbose:
            print("Start optimization")
        diResults = optimizer.fit()

        #set and return results
        self.globalGenerator = diResults['generator']
        self.globalDiscriminator = diResults['discriminator']
        self.iRounds = diResults['rounds']
        self.iSteps = diResults['steps']
        self.vDiscrLoss = diResults['discriminator_loss_per_step']
        self.vGenLoss = diResults['generator_loss_per_step']
        self.vCondLoss = diResults['conditional_loss_per_step']
        self.vModeLoss = diResults['mode_loss_per_step']
        self.fedGANDataset = fedGANDataset
        self.fedGANDataset.ldf = None

        return diResults, fedGANDataset

    def synthetic_data(self, rng_new = None, iRows = 1000, device = "cuda"):
        """
        Goal: Create synthetic data after fitting the model
        Input:
            - self              A FedDPCTGAN object (should be fitted)
            - rng_new           A new rng for creating samples (if not supplied, use default rng)
            - iRows             integer, number of rows to be generated 
        Output:
            - dfData            The synthetic dataset in the original domain
        """
        if device == "cpu":
            self.globalGenerator.to("cpu")
            #self.fedGANDataset.to("cpu")

        #set rng
        if rng_new is None:
            rng_new = self.rng

        #generate conditional vectors using the intialization
        #pick variable
        vVariableSel = rng_new.choice(np.arange(self.fedGANDataset.iD), size = iRows)

        #fill conditional vector
        iTotalVectorLength = sum(self.fedGANDataset.lCondDimPerVariable)
        mCond = torch.zeros((iRows, iTotalVectorLength ), device = device)
        vAllIndices = np.arange(iRows)
        vModeSelection = np.zeros((iRows,))
        lvValues = [np.arange(self.fedGANDataset.lCondDimPerVariable[d]) for d in range(self.fedGANDataset.iD)]
        for i in range(iRows):
            iVarSel = vVariableSel[i]
            vModeSelection[i] = rng_new.choice(a = lvValues[iVarSel], p = self.fedGANDataset.llOneWayMargProbs[iVarSel])
        lCumCondDimPerVariable = np.cumsum([0] + self.fedGANDataset.lCondDimPerVariable[:-1])
        vConditions = np.array([lCumCondDimPerVariable[el] for el in vVariableSel]) + vModeSelection
        mCond[vAllIndices, vConditions.astype(np.int32)] = 1

        #get noise z
        mNoise = torch.from_numpy(self.rng.standard_normal(size = (iRows, self.iNoiseDim), dtype = np.float32)).to(device)

        #add conditional vector to noise
        mNoise = torch.cat([mNoise, mCond], dim = 1)
        if self.sGenType == "CNN":
            mNoise = mNoise.view(iRows, self.iNoiseDim + iTotalVectorLength, 1, 1)

        #generate synthatic data
        fake = self.globalGenerator(mNoise)
        if self.sGenType == "CNN":
            fake = self.Gtransformer.inverse_transform(fake)
        fakeact = apply_activate(fake, self.fedGANDataset.lOutputInfo)

        #concatenate with conditon vectors
        #fake = torch.cat([fakeact, mCond], dim = 1)

        #convert to numpy
        fakeact = fakeact.detach().cpu().numpy()

        #transform to original space
        dfData = self.fedGANDataset.inverse_transform(fakeact)
        dfData = self.fedGANDataset.inverse_preprocess(ganSynthData = dfData, bDiscretized = self.bDiscretize)

        return dfData

