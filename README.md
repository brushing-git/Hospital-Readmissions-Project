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

![dirty data](Images/OriginalData.png)

To clean our data we did the following.  Missing entries for features and those labeled with a `?` were entered with a `NaN`.  Almost all of the missing entries were in a select few features such as *weight*, which had $97\%$ of its entries missing and *payer_code*, which had $39\%$ of its entries missing.  Because these features were either largely missing or seemed to have little application to the target problem, we opted to drop those features.  Duplicate entries (duplicate patients) were also dropped, where we only kept the initial admission into the hospital.  Finally, many of our features had extreme values.  This can bias machine learning algorithms.  Consequently, we scaled these values:  for numeric entries we subtracted the mean and divided by the variance for the specific features and for categorical features we turned them into one-hot encodings.  We also used integer encodings for categorical data and experimented for their effectiveness.  We found this integer data to be problematic.  So we used the one-hot data for our experiments.

An example of the cleaned data can be seen below:

![clean data](Images/CleanedData1Hot.png)

The dropping of features and addition of one-hot encoding resulted in a cleaned data set with 47 features.  This can be found in the clean data set folder [here](CleanedData/).

For our ground truth-label, we wanted to predict hospital readmission.  The original hospital admission feature had multiple entries, indicating the number of days since readmission.  We decided to use *any readmission* as a readmission; our cut-off dates were infinitely long.  Readmission is now a binary random variable.

After preprocessing, the data set was split into training and test data with the labels separated from the features.  Lastly, we noticed that there was some imbalances in the number of readmits based on dropped duplicate entries.  So we rebalanced the data, and used the balanced data for our training and testing.

## Statistical Methods

The primary methods we used to evaluate our models was **Area under the Receiver Operating Charcteristic curve (ROC-AUC)** and **K-fold cross-validation**.

ROC-AUC is a measure of accuracy by measuring a model's true positive and false positive rates.  The true positive rate ($TPR$) is the probability that a model predicts a label given the ground truth label being that label; the false positive rate ($FPR$) is the probability that a model predicts some other label given the ground truth label.  We compute the true positive rate by taking the ratio of the quantity of true positives $TP$ (number of hits on the correct label) by the sum of the true positives plus the false negatives $FN$ (number of misses on the correct label):

$$ TPR = \frac{TP}{TP + FN} $$

We compute the false positive rate by taking the ratio of the quantity of false positives $FP$ (number of hits on the incorrect label) by the sum of the false positives plus the true negatives $TN$ (number of misses on the incorrect label):

$$ FPR = \frac{FP}{FP + TN} $$

Intuitively, we can think about ROC-AUC as measuring how well our model guesses correctly to incorrectly (hits relative to misses).  Our model estimates probabilities for the underlying distribution of categorical labels.  This is called the ROC curve.  Random guessing would equate to equal TPR and FPR.  Better than random guessing would be to maximize TPR relative to FPR.  We then measure the relative area under our ROC curve with respect to random guessing.  This area under the curve indicates how well our model can separate the distributions corresponding to our different labels.  So a higher ROC-AUC, the better our model:  1 being perfect, 0.5 indicating random guessing, and 0 being the worst.

K-fold cross-validation is a method of validating model performance.  We do not want our models to simply memorize the data they have learned in training.  To prevent this, we always measure model performance relative to a validation set:  training data that has been held out to test our model on.  Unfortunately, the set of data we may validate our model on might be configured in a way so as to allow our model to be lucky; its actual performance in the world may not match how it does on the validation set.  To hedge against this, we split our hold out data into separate validation sets and test our model in each validation set and then average its performance.

For our project, we split our validation data into **10** different sets and ran cross-validation on those sets.

## Machine Learning Methods

We used linear classifiers, decision trees, neural networks, and random forests for predicting hospital readmission.  My contribution was work on the **linear classifiers and neural networks**.  Both of my methods were trained using **stochastic gradient descent** in `pytorch`.

A linear classifier is a prediction based on a weighted sum of features.  It is the discrete version of a linear regression where we have to make categorical predictions instead of continous ones.  Our prediction is the label whose parameters maximize a linear sum of the features. If $\overline{x} = (x_{1}, \dots, x_{k})$ are our features and $\overline{\theta} = (\theta_{0}^{y}, \theta_{1}^{y}, \dots, \theta_{k}^{y})$ are our weights for label $y$ of our available labels $Y$ and $\Theta$ is the set of every weights, then the prediction is $f$:

$$ f(\overline{x}; \Theta) = \underset{y \in Y}{\arg \max} \underset{i=1}{\overset{k}{\sum}} \theta_{i}^{y}x_{i} + \theta_{0}^{y} $$

Intuitively, if we plot our samples in space whose dimensions are given by our $k$ features, the linear classifier separates samples by drawing a "line" (hyperplane) through that space.  Our model is characterized by this decision line, and we can think of our weights as specifying where that decision line lies in feature space.

Neural networks are an extension of linear classifiers.  Often, our features cannot be separated by such a "decision line".  However, by transforming our feature space, we can sort our samples so that they can be linearly separated.  A neural network is a method for learning how to transform our features.  We do this by passing our features through successive linear and non-linear transformations.  Each successive transformation consists of a linear operation followed by a non-linear operation.  The linear operation is the same as in linear classifiers:  features from earlier in the model are weighted by some parameters and those multiplications are summed.  The non-linear operation is some function, such as the sigmoid function or $\max (0,x)$ function, whose transformation is not equivalent with a weighting and sum.  After the non-linear operation is done, it is passed as a feature onto the next stage of the model.  Mathematically, the output function $g$ is a series of non-linear and linear operations, where $\phi$ is a non-linear function and $\overline{\theta}_{i} \in \Theta$ are a matrix of  weights for each $i = \{1, \dots, j \}$:

$$ g(\overline{x}; \Theta) = \phi (\overline{\theta}_{j} \cdot \phi (\overline{\theta}_{j-1} \cdot \phi (\overline{\theta}_{j-2} \cdot \dots \phi (\overline{\theta}_{1} \cdot \overline{x}) \dots )) $$

Here we think about the sum and products as dot products.  The name of neural network comes from interpreting each matrix as a layer of neurons and each column in the matrix as corresponding to the weights a particular neuron assigns to its inputs.

## Experiments

## Results

## Discussion
