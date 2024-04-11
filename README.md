Data and code for the paper "A Clustering-based uncertainty set for Robust Optimization]{A Clustering-based uncertainty set for Robust Optimization" by Alireza Yazdani, Ahmadreza Marandi, Rob Basten, and Lijia Tan

Structure:
  - Uset_construction contains tarining code for constructiong uncertainty sets for p=2 and p=+Inf
  - Experiment 1 contains needed code and data for generating figures for different random generated 2d cases
  - Experiment 2 contains needed codes and data for second experiment on a RO problem
      + Data containts test and train data for 30 different data sets
      + Uncertainty_set containts the parameters for different uncertainty sets, and the codes provided in https://github.com/goerigk/RO-DNN
      + Solutions contains all of the solutions generated using provided uncertainty sets, and the codes needed to achieve them
      + Evaluate_solutions.jl gives you the excel file of the result   
      
