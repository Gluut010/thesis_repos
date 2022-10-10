# thesis_repos

Main package in FedDifPrivModels. Illustration.ipynb gives an example of how to use this package.

Data is loaded with data_loading.py, credit fraud dataset has been deleted as it is larger than 100 MB. Models in are trained using multiprocessing in mainTrain (using FedDifPrivModels), and evaluated in mainEvaluation (using FedDifPrivModels). After that resultPlots is used to create plots for the results. 

### Overview pcode
A general overview of the code
- FedDifPrivModels: Main “package” with the following structure:
  - FedDifPrivPGM (Class). This is the main class of the federated, differentially private PGM models. These models learn a graphical model on Also includes score functions.
    - FedAllInOnePGM (Class). A subclass of FedDifPrivPGM. Creates an instance of FedDifPrivPGM that uses the FSTS (Federated private Static marginal Tree Selection) method. That is, a tree as graph, and  all marginals are measured in one go. Also includes fit functions and helper functions for fitting.
    - FedAdapIterPGM (Class). A subclass of FedDifPrivPGM. Creates an instance of FedDifPrivPGM that uses the FIGS (Federated private Iterative marginal Graph Selection) method. That is, a graph with restricted junctiontree size, and marginals are iteratively selected. Also includes fit functions and helper functions for fitting.
  - FedDPCTGAN (Class). The main class for the federated, differentially private GAN models. These models learn a generative adversarial network.
  - FedDataset (Class). Creates a Federated dataset objects. This object consists of a list of dataframes, a dictionary with the min and max value for mixed, ordinal and continuous variable, and a dictionary with the unique values of categorical variables. Our methods assume these attributes are public knowledge.
    - FedPGMDataset (Class). A subclass of FedDataset for PGM models. Has functions for discritization of the (numerical variables of the) dataset. 
    - FedGANDataset (Class). A subclass of FedDataset of GAN models. Has functions to apply the Mode-specific normalization procedure (and to transform back).
  -	Utils. Helper functions. For example, a helper function to split a dataset in a homogeneous or heterogeneous dataset.

- data_loading.py. program to load and preprocess the datasets used in this thesis.
- illustration.ipynb. Illustration of the FedDifPrivModel "package" .
- mainEvaluation.py. Program to evaluate all trained models with multiprocessing.
- mainTemp.ipynb. ....
- mainTrain.py. Program to train all models, using multiprocessing.
- plots.R. Some plots for the thesis.
- resultsPlots.ipynb.