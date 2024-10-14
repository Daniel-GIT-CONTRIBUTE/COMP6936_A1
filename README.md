# COMP6936_A1
Assignment1 for COMP 6936

# Dependencies:  
  - Jupyter Lab, Jupyter Notebook, Google Colab, or any program that reads .ipynb file
  - numpy, pandas, seaborn, pygam, sklearn, matplotlib
# How to run:
  - use a .ipynb software like Google Colab or Jupyter
  - OR run the .py file but the dependent libraries must all be installed
  - OR view the compiled project here in GITHUB
# Data used:
  - source: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
  - description: Context
  - This data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, including the predicted attribute, but all published experiments refer to using a subset of 14 of them. The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no disease and 1 = disease.

# Methods:
  - Model: 5fold Cross Validation using GAMs
  - Scores: AUC (with visualization), F1

# There are 13 variables (X) and 1 output of zero or one(y)
  - The 13 variables in order are:
    - 0.age
    - 1.sex
    - 2.chest pain type (4 values)
    - 3.resting blood pressure
    - 4.serum cholestoral in mg/dl
    - 5.fasting blood sugar > 120 mg/dl
    - 6.resting electrocardiographic results (values 0,1,2)
    - 7.maximum heart rate achieved
    - 8.exercise induced angina
    - 9.oldpeak = ST depression induced by exercise relative to rest
    - 10.the slope of the peak exercise ST segment
    - 11.number of major vessels (0-3) colored by flourosopy
    - 12.thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
    - **The names and social security numbers of the patients were recently removed from the database, replaced with dummy values.

  # Sites Consulted:
    - https://www.geeksforgeeks.org/generalized-additive-model-in-python/
    - https://pygam.readthedocs.io/en/latest/notebooks/tour_of_pygam.html
  
    
