# har #
## Algorithm for Human Activity Recognition using wearables ##

This portfolio presents a new algorithm for human activity recognition (HAR) using wearables' data (accelerometers and gyroscopes).
The folders "dailac" and "mhealth" contain public datasets with labelled wearables' signals collected while subjects were perfoming daily  activities such as walking, sitting, biking etc.

Details about the datasets can be found in the dedicated articles: 

Leutheuser H, Schuldhaus D, Eskofier BM. Hierarchical, Multi-Sensor Based Classification of Daily Life Activities: Comparison with State-of-the-Art Algorithms Using a Benchmark Dataset. PLOS ONE. 2013;8:e75196.

Banos O, Garcia R, Holgado-Terriza JA, Damas M, Pomares H, Rojas I, et al. mHealthDroid: A Novel Framework for Agile Development of Mobile Health Applications. In: Pecchia L, Chen LL, Nugent C, Bravo J, editors. Ambient Assisted Living and Daily Activities. Cham: Springer International Publishing; 2014. p. 91–8.

The files "har_dailac.ipynb" and "har_mhealth.ipynb" are commented Jupyter notebooks walking the reader through the script for processing of data reading, processing, models training and predicting. The Python code in the notebooks draws upon the module "har_utils.py", containing customized function for HAR. 

"TableA.xlsx" reports the full results for the tests run against the DaiLAc dataset. "TableB.xlsx" reports a comparison of the classification accuracy for DaiLAc with versus without gyroscopes.

