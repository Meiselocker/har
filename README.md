# har #
## Human Activity Recognition using wearables ##
**har** is a fast, performant and easy-to-use package for processing wearables' data (accelerometers, gyroscopes, or even magnetometers) and detecting human activity. 

**har** relies on robust features extraction in the temporal and frequential domains, which allows for excellent performance in various contexts even with simple machine learning models such as Support Vector Machine or Logistic Regression. **har** supports hierarchical learning, which can significantly boosts detection accuracy. 

A description of the algorithm and a comparison with existing state-of-the-art algorithms can be found [here](https://www.mdpi.com/1424-8220/20/11/3090).  

## What you can do with **har** ## 
* Read, window, process your raw data and extract meaningful features from it.
* Train, test and save your model against labelled data using hierachical learning. 
* Predict unlabelled data using your trained model.

## Technology
**har** uses Python 3.x with standard data science libraries (pandas, numpy, scipy, scikit-learn). 
See requirements for the full list of dependencies. 



## Performance
**har** was tested on the [*DaiLAc*](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0075196) and [*mHealth*](https://link.springer.com/content/pdf/10.1007/978-3-319-13105-4.pdf) benchmark datasets with [great success](https://www.mdpi.com/1424-8220/20/11/3090). The datasets are stored in the repository, as well as the test scripts.

## What this repository contains ##

* `Requirements.txt`: List of dependencies.
* `har`: Library with the modules `har.py` (containing the `Har` class) and `funs.py` (auxiliary functions). 
* `dailac`: Folder containing the DaiLAc dataset.
* `mhealth`: Folder containing the mHealth dataset.
* `test_dailac.py`: Tests the main functionalities on the *DaiLAc* data sets and reports results.
* `test_mhealth.py`: Tests the main functionalities on the *mHealth* data sets and reports results.
* `TableA.xlsx`: Reports the full results for the tests run against the DaiLAc dataset.
* `TableB.xlsx`: Reports a comparison of the classification accuracy for DaiLAc with versus without gyroscopes.
* `README.md` and `README.html`


## Usage ##
Please refer to the fully commented `test_dailac.py` script for an overview of the package's functionalities.

## Author ## 
Isaac Debache
