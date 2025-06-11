# Phishing Website Detection by Machine Learning Techniques

## Objective
A phishing website is a common social engineering method that mimics trustful uniform resource locators (URLs) and webpages. The objective of this project is to train machine learning models and deep neural nets on the dataset created to predict phishing websites. Both phishing and benign URLs of websites are gathered to form a dataset and from them required URL and website content-based features are extracted. The performance level of each model is measures and compared.

## Data Collection
The set of phishing URLs are collected from opensource service called **PhishTank**. This service provide a set of phishing URLs in multiple formats like csv, json etc. that gets updated hourly. To download the data: https://www.phishtank.com/developer_info.php. From this dataset, 5000 random phishing URLs are collected to train the ML models.

The legitimate URLs are obatined from the open datasets of the UCI, https://archive.ics.uci.edu/ml/machine-learning-databases/00327/phishing.csv. This dataset has a collection of benign, spam, phishing, malware & defacement URLs. Out of all these types, the benign url dataset is considered for this project. From this dataset, 5000 random legitimate URLs are collected to train the ML models.

The above mentioned datasets are uploaded to the '[DataFiles](https://github.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques/tree/master/DataFiles)' folder of this repository.

## Feature Extraction
The below mentioned category of features are extracted from the URL data:

1.   Address Bar based Features <br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In this category 9 features are extracted.
2.   Domain based Features<br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In this category 4 features are extracted.
3.   HTML & Javascript based Features<br>
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In this category 4 features are extracted.

*The details pertaining to these features are mentioned in the [URL Feature Extraction.ipynb.](https://github.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques/blob/master/URL%20Feature%20Extraction.ipynb)[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques/blob/master/URL%20Feature%20Extraction.ipynb)*

So, all together 17 features are extracted from the 10,000 URL dataset and are stored in '[5.urldata.csv](https://github.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques/blob/master/DataFiles/5.urldata.csv)' file in the DataFiles folder.<br>

## Models & Training

Before stating the ML model training, the data is split into 80-20 i.e., 8000 training samples & 2000 testing samples. From the dataset, it is clear that this is a supervised machine learning task. There are two major types of supervised machine learning problems, called classification and regression.

This data set comes under classification problem, as the input URL is classified as phishing (1) or legitimate (0). The supervised machine learning models (classification) considered to train the dataset in this project are:

* Decision Tree
* Random Forest
* Multilayer Perceptrons
* XGBoost
* Autoencoder Neural Network
* Support Vector Machines

All these models are trained on the dataset and evaluation of the model is done with the test dataset. The elaborate details of the models & its training are mentioned in [Phishing Website Detection_Models & Training.ipynb](https://github.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques/blob/master/Phishing%20Website%20Detection_Models%20%26%20Training.ipynb)[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques/blob/master/Phishing%20Website%20Detection_Models%20%26%20Training.ipynb)

## How to run it

1. Prerequisites<br>
* Python (recommended version: 3.8+)<br>
* Jupyter Notebook or JupyterLab<br>
* Git<br>
* pip (Python package installer) <br>


2. Steps to Run It<br>
* Step 1: Clone the Repository<br>
* Open your terminal or command prompt and run:<br>

       git clone https://github.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques.git
       cd Phishing-Website-Detection-by-Machine-Learning-Techniques

* Step 2: Set Up a Virtual Environment (Recommended)

       python -m venv venv
       source venv/bin/activate      # On macOS/Linux
       venv\Scripts\activate         # On Windows

* Step 3: Install Dependencies<br>
Check if a requirements.txt file exists. In this repo, it doesn’t, but from the .ipynb file, you can see it uses libraries like:<br>
* pandas<br>
* numpy<br>
* sklearn<br>
* matplotlib<br>
Install them manually:<br>

        pip install pandas numpy scikit-learn matplotlib
  
* Step 4: Launch Jupyter Notebook<br>
Start Jupyter in the project directory:<br>

        jupyter notebook
  
Then open the notebook:<br>

        Phishing_Website_Detection_by_Machine_Learning_Techniques.ipynb

* Step 5: Run the Notebook<br>
Run all cells step-by-step:<br>
* The notebook reads the dataset from phishing.csv.<br>
* It performs feature analysis, preprocessing, and model training using various classifiers (like Decision Trees, Random Forest, SVM, etc.).<br>

If you get an error related to missing files, ensure phishing.csv is present in the same folder.<br>

## End Results
From the obtained results of the above models, XGBoost Classifier has highest model performance of 86.4%. So the model is saved to the file '[XGBoostClassifier.pickle.dat](https://github.com/shreyagopal/Phishing-Website-Detection-by-Machine-Learning-Techniques/blob/master/XGBoostClassifier.pickle.dat)'

### Next Steps

This project can be further extended to creation of browser extention or developed a GUI which takes the URL and predicts it's nature i.e., legitimate of phishing. *As of now, I am working towards the creation of browser extention for this project. And may even try the GUI option also.* The further developments will be updated at the earliest. 
