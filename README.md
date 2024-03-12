# PROJECT DOCUMENTATION

Based on https://www.kaggle.com/datasets/ulrikeherold/tech-layoffs-2020-2024/data

## Data pre-processing

- Min-max scaling: Since we are using data across various domains (stock, GDP, etc.), different parts of the data vary greatly in scale. To ensure that we can understand the weightage of features in whatever models we want to run, we choose to use min-max scaling and put all features on the same scale. This prevents a high-magnitude feature from dominiating the algorithm's learning process.
- Drop null values: Null values can distort every step of the machine learning pipeline, from EDA to results. We drop null values to ensure a complete dataset with high quality and reliability.
- Drop non-US companies: Since our project's goal is to predict tech layoffs within the United States, we choose to drop non-US companies. To the same end, we will only be working with the US's GDP and foreign aid statistics, and our stock data is from the NASDAQ technological sector.
- Drop irrelevant columns: Dropping unnecessary columns reduces the dimensionality of our dataset (this also helps to make output more readable) and ensures that our model is not affected by noisy data that does not relate to the prediction. By streamlining the dataset, we can efficiently train our model and increase performance.

## FINALIZED Data pre-processing

- Location_HQ: Changed into region by timezone based on the city, numerically encoded
- Stage, Industry, Location_HQ: one-hot encoded
- Stock Delta: calculated delta based on the past 90 days (to signify last quarter)
- Dates: converted to UNIX to be able to categorize it
- Removing Outliers: removed companies with an intial employee size of <10 to prevent outliers such as small startups going bankrupt. Functionally, users of our tool are unlikely to predict whether or not they will be laid off since they likely know the CEO and how well their company is doing. Additionally, a small startup is more likely to go bankrupt then lay anyone off leading to a "100%" layoff rate.

## Data Exploration & EDA

**Tools/Plots**

- Pairplot: A pairplot can help us understand linear and nonlinear relationships between various features and the chance of a major tech layoff. Discerning these relationships can help us choose certain features as stronger predictors, customize our model, and visualize predictions.
- Histograms: Histograms can help us visualize the features we choose to incorporate the model. Primarily, being able to see if the data is skewed or has outliers can help us normalize and clean noisy data. Furthermore, since we are using some detailed 'time-series'-esque data, histograms can give us a better understanding of large-scale trends.
- Features description: While not a quantitative factor, describing our features makes it easier to understand the meaning of each variable. We can get a better idea of what our model weights mean contextually, and we can reveal a deeper analysis in the project.
- Heatmap: A huge problem in machine learning tasks is multicollinearity, or when multiple predictors are linearly dependent. To ensure statistical significance, we always strive to use independent variables. A heatmap can help us identify dependent features and filter out any data where needed.
- Correlation analysis: Similar to a pairplot, a correlation analysis can help us find relationships between a feature and the target (tech layoff). Specificially, we can find the strength of a linear relationship, with the added benefit of getting a quantitative value instead of just a plot.

**General Analysis**

- We also did some general preprocessing. Our tech layoff dataset luckily had little to no nulls, but some names and column values were off, so we fixed it (for example, Salesforce was incorrectly named). We also began bringing in financial data and global GDP data so that we could correlate external factors to our prediction as well.
- Number of Layoffs per Year in the USA and Top Industries Affected: While these are tech companies, we wanted the insights as to which industry the tech companies catered to were most affected. It seemed that Consumer, Transportation, Financial companies were the most affected.
- Because we want to identify the attributes that will affect our prediction the most, we also looked at which geographical areas were affected the most (San Fran Bay Area, Seattle, etc.), the companies with most layoffs (Google, Meta, etc.)
- We also printed a lot of general statistics about our dataset, like the number of observations, deviation, etc.
- Lastly, we also dropped columns we believed would be unnecessary, priming our dataset for the coming machine learning tasks.

**In-depth, Comprehensive Analysis of Relevant Variables**

- Besides all of the tasks listed above, we also performed in depth exploration and analsis on some data variables that we deem as important to our machine learning project's goal.
- The variables we took into consideration were: money raised, size of company before and after layoff, stage of the company, and industry of the company.
- By performing correlation analysis on layoff percentage, and money raised + size of company before and after layoff, we helped answer the following questions: Would a company have more layoffs if its company size was relatively big before the layoff? Does the size of the company after a layoff be indicative of how big the layoff was? If a company has more money raised, then would the size of their layoff be smaller?
- Additionally, by exploring industry and stages of the companies that had layoffsm we also answered the following questions: Are certain industries more prone to experiencing layoffs compared to others? Does the current stage of a company serve as a significant indicator of the likelihood of layoffs?

## Model 1 - Polynomial Regression

[Finalized Data Preprocessing + Model 1 Analysis Notebook](./FinalizedDataPreprocessingCharisseKevinKenneth.ipynb)

- We decided to use polynomial regression to predict the percentage of a company layed off based on our input features.
- Polynomials degrees 1 through 4 were used to predict the percentage layed off and MSE was used to determine the performance of the model.

  ![Polynomial Regression MSE](images/polyreg-mse.png)

- As we increased the polynomial degree, our model's MSE on the training data set shrunk, but on the testing data it grew, so we conclude that our model is overfitting. Based on the graph, we can say that at about the third degree, the model begins to overfit.
- For our next two models, we are considering using the following models:
  - Automatic Relevance Determination regression to see if a different regularization technique will better assign weights depending on the feature relevance.
  - Neural Networks to better find patterns within our data.
- In conclusion, polynomial regression is not the best way for us to model our data. While it can be improved by using higher degrees or limiting our features, the time required to compute a polynomial regression makes us less likely to use this model.

**Next Two Models**

- One model we could find would be from using Grid Search to find a better tuned Deep Neural Network.
- Another model we could try is Automatic Relevance Determination (ARD) since we aren't sure if all of our features are relevant in predicting the layoff percentage.

## Model 2 - Neural Network

[Model 2 – Neural Net](https://github.com/katulevskiy/tech_layoffs_ml/blob/main/FinalizedDataPreprocessingCharisseKevinKenneth.ipynb)
All of our data, labels, and loss function remained the same; however, because our original loss function did not perform well, we decided to add the below changes to better tune it and improve our model predictions. 

For model 2 we decide to train a Neural Net to predict the probability of tech layoffs given a company. We use the same features as Model 1 (Date*layoffs, stock_delta, Company_Size_before_Layoffs, Money_Raised_in*$\_mil, Industry, Stage, and Region), and decide to use MSE to evaluate our model. We decided to use MSE as we work with continuous data for our target (percentage likely to be laid off), which MSE reflects well. Our features and loss functions were therefore sufficient, and we did not have to change them from Model 1.

### Results

We had the following MSE results for the base neural network model:

- Train MSE - 786.0688282061325
- Test MSE - 584.9784896957286

To improve our model, we ran Grid Search, during which we modified the number of units in each hidden layer of our network, as well as the activation function in the hidden and output layers. We chose the best model based on the set of hyperparameters that performed best on the validation set. The following results were obtained:

- Train MSE - 407.6795840293022
- Test MSE - 344.07788384916205
- Validation MSE - 222.8351758414751

After optimizing the model with Grid Search, Model 2 performs a lot better and seems to not be overfitting on training data, but seems to pick the model that does the best on validation data (it is better on validation compared to training). This occurs because during Grid Search, we choose the set of hyperparameters that result in the best validation MSE. The result of this could potentially be from random choice, where it may have randomly done the best on that validation dataset.

Given the MSE's above, we infer that the neural network model did not overfit on the training data. Therefore, we claim that our model falls on the lower end of the fitting graph, corresponding to low model complexity. Compared to our first model (polynomial regression), we fit a lot less to the training data; we had a 17% higher MSE on the test data than the training data on the first model, whereas this model had a lower testing MSE than on train.

### Future changes

In the future, we would like to implement K-fold cross validation with Grid Search to perhaps find an even better model. Our main shortcoming was that we maintained the validation set on every iteration, which may have led to hyperparameters that performed exceptionally well on that dataset. Randomly selecting the validation set through K-fold cross validation may result in a better model.

### Next model: Random Forest Regressor

The next model we are thinking of implementing is a Random Forest Regressor. A Random Forest is an ensemble method that combines the results of various decision trees to make its prediction. When it comes to regression, it specifically takes the mean output from each tree. Random Forest Regression works well for a variety of reasons. Firstly, because it is an ensemble method, it can help reduce variance and bias in predictions, since one specific tree (each tree is its own ‘model’) will not skew predictions. For the same reason, it tends to not overfit on data. When it comes to financial data, a Random Forest may perform well because it is good at capturing complex nonlinear relationships. Our data consists of a lot of features that interact in ways we cannot understand due to complicated market dynamics. A Random Forest model can capture such relationships while also outputting feature importance, which can be useful for analysis.

### Model 2 Conclusion

Although our initial base model for our second model performed much worse than the first (786.068 training MSE compared to 455.386 training MSE in the first model), after performing hyperparameter tuning, our second model improved a lot more. Our final MSE for model 2 was 407.679 training and 344.077 testing. This performed a lot better because we did grid search which computes the optimum hyperparameters to use to predict our data. It performed a lot better than our first model because our first model was just polynomial regression, and neural networks can adapt much better to complex datasets. In order to improve it, we could have done k-fold cross validation instead of using the same validation set every time, because otherwise it gets too specific on one single validation set that doesn’t necessarily train it on all of the different sets it could be on. Also, because not every feature is really impactful for our predictions, we could have isolated only the more influential features and made our predictions from there. This is why we are thinking of doing a random forest model as our model 3 because it will allow us to be more focused in what we use to train.

## Final Project Submission 

### Introduction 
In today’s ever changing economy, companies are constantly needing to make restructuring adjustments, including layoffs. Though this is often a necessary change, employees at companies planning to do layoffs might feel completely blindsided as they often don’t know how many people are actually being laid off. 
We wanted to create a layoff predictor to help employees know what to expect when they find out that their company is already planning to lay off people. It helps employees think about what their position is at the company, and they can make preparations based on how likely they are to be laid off based on the percentage of people we predict to be laid off. They can start preparing and looking for jobs at other companies if necessary. 


Additionally, it can also be helpful for companies who know that they should lay off employees, but don’t know what percentage of their staff that they should actually lay off. Our predictor can be used as a guiding indicator of what is recommended based on other companies in that situation and the status of the economy at the time. It can help prevent companies from accidentally laying off too many people, or not laying off enough people and needing to do a second round of layoffs. 


