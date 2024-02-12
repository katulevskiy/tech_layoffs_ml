# PROJECT DOCUMENTATION

Based on https://www.kaggle.com/datasets/ulrikeherold/tech-layoffs-2020-2024/data

## Data pre-processing

- Min-max scaling: Since we are using data across various domains (stock, GDP, etc.), different parts of the data vary greatly in scale. To ensure that we can understand the weightage of features in whatever models we want to run, we choose to use min-max scaling and put all features on the same scale. This prevents a high-magnitude feature from dominiating the algorithm's learning process.
- Add external data to the dataset: The original dataset does not cover GDP, foreign aid, and stock values. However, these macroeconomic trends reflect in company behavior, and therefore can be useful to understand layoffs in tech companies. We incorporate these values to improve our model's knowledge base.
- Drop null values: Null values can distort every step of the machine learning pipeline, from EDA to results. We drop null values to ensure a complete dataset with high quality and reliability.
- Drop non-US companies: Since our project's goal is to predict tech layoffs within the United States, we choose to drop non-US companies. To the same end, we will only be working with the US's GDP and foreign aid statistics, and our stock data is from the NASDAQ technological sector.
- Drop irrelevant columns: Dropping unnecessary columns reduces the dimensionality of our dataset (this also helps to make output more readable) and ensures that our model is not affected by noisy data that does not relate to the prediction. By streamlining the dataset, we can efficiently train our model and increase performance.

## Data Exploration & EDA

__Tools/Plots__
- Pairplot: A pairplot can help us understand linear and nonlinear relationships between various features and the chance of a major tech layoff. Discerning these relationships can help us choose certain features as stronger predictors, customize our model, and visualize predictions.
- Histograms: Histograms can help us visualize the features we choose to incorporate the model. Primarily, being able to see if the data is skewed or has outliers can help us normalize and clean noisy data. Furthermore, since we are using some detailed 'time-series'-esque data, histograms can give us a better understanding of large-scale trends.
- Features description: While not a quantitative factor, describing our features makes it easier to understand the meaning of each variable. We can get a better idea of what our model weights mean contextually, and we can reveal a deeper analysis in the project.
- Heatmap: A huge problem in machine learning tasks is multicollinearity, or when multiple predictors are linearly dependent. To ensure statistical significance, we always strive to use independent variables. A heatmap can help us identify dependent features and filter out any data where needed.
- Correlation analysis: Similar to a pairplot, a correlation analysis can help us find relationships between a feature and the target (tech layoff). Specificially, we can find the strength of a linear relationship, with the added benefit of getting a quantitative value instead of just a plot.

__General Analysis__
- We also did some general preprocessing. Our tech layoff dataset luckily had little to no nulls, but some names and column values were off, so we fixed it (for example, Salesforce was incorrectly named). We also began bringing in some financial data and global GDP data so that we could correlate external factors to our prediction as well.
- Number of Layoffs per Year in the USA and Top Industries Affected: While these are tech companies, we wanted the insights as to which industry the tech companies catered to were most affected. It seemed that Consumer, Transportation, Financial companies were the most affected.
- Because we want to identify the attributes that will affect our prediction the most, we also looked at which geographical areas were affected the most (San Fran Bay Area, Seattle, etc.), the companies with most layoffs (Google, Meta, etc.)
- We also printed a lot of general statistics about our dataset, like the number of observations, deviation, etc.
- Lastly, we also dropped columns we believed would be unnecessary, priming our dataset for the coming machine learning tasks.
