Data and code for the paper "A Clustering-based uncertainty set for Robust Optimization]{A Clustering-based uncertainty set for Robust Optimization" by Alireza Yazdani, Ahmadreza Marandi, Rob Basten, and Lijia Tan

Structure:
  - Uset_construction contains training code for constructing uncertainty sets for p=2 and p=+Inf
  - Experiment 1 contains the needed code and data for generating figures for different randomly generated 2d cases
  - Experiment 2 contains the needed codes and data for the second experiment on an RO problem
      + Data contains test and train data for 30 different data sets
      + Uncertainty_set contains the parameters for different uncertainty sets, and the codes (The ones for NN and Kernel are the ones provided in https://github.com/goerigk/RO-DNN)
      + Solutions contains all of the solutions generated using provided uncertainty sets and the codes needed to achieve them
      + Evaluate_solutions.jl gives you the Excel file of the result   
      
