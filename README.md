# Hospital-Readmissions

![Pearson Correlation](Images/pearsoncorrelation.png)

## Overview

Machine learning models offer the ability to make predictions about future events from past records.  By learning patterns in data, a model can generate informed guesses that often exceed chance and even the predictions of human subject matter experts.  With this project, I worked with a team of fellow University of California, Irvine graduate students to build a series of machine learning algorithms that predict the probability diabetes patients might be readmitted to hospital.  This demonstrates my facility with **data grooming, data manipulation, model training, model evaluation, model improvement, model selection, and data science teamwork**.

This project is directly relevant to any organization that has extensive records and wants to extract predictions from those records.  Those predictions could be useful for modifying future plans.  For example, this project deals with predicting hospital readmissions.  It would be directly relevant for doctors, hospital administrators, insurance companies, and patients themselves to know patient risks for readmission.  Using that knowledge could allow for precautions to be taken to reduce those risks.

The machine learning methods that myself and my teammates employed include **linear classifiers, decision trees, neural networks, and random forests**.  These are all examples of **supervised learning**, where models are trained like students at school based on their ability to give correct answers and learn from their mistakes.  We estimated model accuracy with **ROC-AUC** and performed model evaluation via **K-fold cross-validation**.  The software we used included **pandas, numpy, matplotlib, jupyter notebook, scikit-learn, and pytorch**.

My project teammates were [Kate Deyneka](https://www.linkedin.com/in/edeyneka/) and [Brian Tran](https://www.linkedin.com/in/brian-d-tran/).  **The contributions I provided in the project include help with data grooming and manipulation and sole work on linear classifiers and neural networks.**  I also helped author our final report.

## Data Sets

The data set we used is the [diabetes 130-US hospitals set](https://archive-beta.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008) found on the UCI Machine Learning Repo.  It is data taken from 130 US hospitals from 1999 to 2008, and it represents patient and hospital outcomes.  This data set has 50 variables and 101,766 samples.  Examples of variables include patient race, age, time in the hospital, outcomes of various tests run, and so on.

Importantly, the data as presented is a **dirty data set**:  it contains missing entries, duplicate entries, and outlandish values.  We had to do extensive cleaning.  We preprocessed the data using `pandas`, and we leveraged the `pandas-profiling` (found [here](https://pypi.org/project/pandas-profiling/)) module to generate initial diagnostics about the data.  The profile report can be found [here](report.html).  An example of the dirty data can be found below.

[dirty data](Images/OriginalData.png)

To clean our data we did the following.  Missing entries for features and those labeled with a `?` were entered with a `NaN`.  Almost all of the missing entries were in a select few features such as *weight*, which had $97\%$ of its entries missing and *payer_code*, which had $39\%$ of its entries missing.  Because these features were either largely missing or seemed to have little application to the target problem, we opted to drop those features.  This reduced our features to 47.  Duplicate entries (duplicate patients) were also dropped, where we only kept the initial admission into the hospital.  Finally, many of our features had extreme values.  This can bias machine learning algorithms.  Consequently, we scaled these values:  for numeric entries we subtracted the mean and divided by the variance for the specific features and for categorical features we turned them into one-hot encodings.  We also used integer encodings for categorical data and experimented for their effectiveness.  We found this integer data to be problematic.  So we used the one-hot data for our experiments.

## Statistical Methods

## Machine Learning Methods

## Experiments

## Results

## Discussion
