#!/usr/bin/env python
# coding: utf-8

# # Case Study: Wine Quality
# in this Jupyter notebook we will be analysing data of red and white wine from a dataset of the UCI - follow this [link](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) to download the data, since it will not be stored in the repository.

# In[25]:


# import pandas and csv file. IMPORTANT! csv file is separated by ";"
import pandas as pd
df_red = pd.read_csv('winequality-red.csv', sep = ";")
df_white = pd.read_csv('winequality-white.csv', sep = ";")
df_red.head()


# In[26]:


df_white.head()


# ## Let's understand the data we are working with

# In[3]:


# get the information about the data type in each dataframe
df_red.info()
df_white.info()


# In[5]:


# Number of duplicates in the whitewine dataframe
df_white.duplicated().sum()


# In[6]:


# Number of unique values for each column in the redwhine dataframe
df_red.nunique()


# In[8]:


df_white.nunique()


# In[10]:


# Statistical information about each column in the redwine dataframe
df_red.describe()


# ## Appendig the Data to get one DataFrame

# In[14]:


# import numpy
import numpy as np


# ### Create Color Columns
# Create two arrays as long as the number of rows in the red and white dataframes that repeat the value “red” or “white.” NumPy offers really easy way to do this. Here’s the documentation for [NumPy’s repeat](https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html) function.

# In[17]:


red_array = np.repeat('RED', len(df_red.index))
white_array = np.repeat('WHITE', df_white.shape[0]) 
print(len(red_array), len(white_array))


# Add arrays to the red and white dataframes. Do this by setting a new column called 'color' to the appropriate array.

# In[18]:


df_red['color'] = red_array
df_white['color'] = white_array
df_red.head()


# ### Combine DataFrames with Append
# Check the documentation for [Pandas' append](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.append.html) function and use this to combine the dataframes. (Bonus: Why aren't we using the [merge](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html) method to combine the dataframes?) If you don’t get it, I’ll show you how afterwards.

# In[23]:


wine_df = df_red.append(df_white, ignore_index=True)
wine_df.tail()


# In[24]:


wine_df.to_csv('winequality_edited.csv', index = False)


# # Exploratory Data Analysis (EDA) with Visuals
# lets explore the newly created df

# In[28]:


# some histograms
wine_df.hist(figsize = (12, 12));


# In[30]:


# scatterplot matrix
pd.plotting.scatter_matrix(wine_df, figsize = (15,15));


# # Using Groupby to draw conclusions

# ## Q1: Is a certain type of wine (red or white) associated with higher quality?

# In[32]:


# use groupby to find out
wine_df.groupby(['color'])['quality'].mean()


# we can see that the mean quality of white wine is greater than the red wine mean quality

# ## Q2: What level of acidity (pH value) receives the highest average rating?
# This question is more tricky because unlike `color`, which has clear categories you can group by (red and white) `pH` is a quantitative variable without clear categories. However, there is a simple fix to this. You can create a categorical variable from a quantitative variable by creating your own categories. [pandas' cut](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html) function let's you "cut" data in groups. Using this, create a new column called acidity_levels with these categories:
# 
# **Acidity Levels**:
# 
#    * High: Lowest 25% of pH values
#    * Moderately High: 25% - 50% of pH values
#    * Medium: 50% - 75% of pH values
#    * Low: 75% - max pH value
# 

# In[37]:


# find the distribution of the data
wine_df['pH'].describe()


# In[49]:


# define the bin's intervals
bin_edges = [2.27, 3.11, 3.21, 3.32, 4.01]

# define the labels
bin_names = ['High', 'Moderately High', 'Medium', 'Low']


# In[52]:


# create a new column with the acidity_levels
wine_df['acidity_levels'] = pd.cut(wine_df['pH'], bin_edges, labels = bin_names)
wine_df.head()


# In[51]:


wine_df.groupby(['ph_category'])['quality'].mean()


# # Using the query command to draw conclusions

# ## Q1: Do wines with higher alcoholic content receive better ratings?
# 
# To answer this question, use query to create two groups of wine samples:
# 
#    * Low alcohol (samples with an alcohol content less than the median)
#    * High alcohol (samples with an alcohol content greater than or equal to the median)
# 
# Then, find the mean quality rating of each group.

# In[56]:


# get the median and furhter information about the alcohol column
alcohol_median = wine_df['alcohol'].median()
alcohol_min = wine_df['alcohol'].min()
alcohol_max = wine_df['alcohol'].max()


# In[58]:


# set the bin's edges
bin_edges_alcohol = [alcohol_min, alcohol_median, alcohol_max]

# set the names of the bins
bin_names_alcohol = ['Low alcohol', 'High alcohol']


# In[60]:


# create a column with the alcohol levels
wine_df['alcohol_level'] = pd.cut(wine_df['alcohol'], bin_edges_alcohol, labels = bin_names_alcohol)
wine_df.head()


# In[62]:


# get the means of quality using groupby
wine_df.groupby(['alcohol_level'])['quality'].mean()


# In[79]:


# get the same result using query()
wine_df_low_alcohol = wine_df.query('alcohol < 10.3')
wine_df_high_alcohol = wine_df.query('alcohol >= 10.3')
wine_df_low_alcohol['quality'].mean(), wine_df_high_alcohol['quality'].mean()


# There are some minor differences in the way we explore the data - groupby depends on the way the bins edges are considered for the classification. However we can clearly see that wines with higher alcohol levels have a higher quality rate.

# ## Q2: Do sweeter wines (more residual sugar) receive better ratings?
# 
# Similarly, use the median to split the samples into two groups by residual sugar and find the mean quality rating of each group.

# In[68]:


# find the median 
wine_df['residual sugar'].median()


# In[76]:


# define new dataframes depending on the median
low_sugar = wine_df.query('`residual sugar` < 3.0')
high_sugar = wine_df.query('`residual sugar` >= 3.0')


# In[77]:


# get the quality mean for each sugar level (low, high)
low_sugar['quality'].mean(), high_sugar['quality'].mean() 


# # Communicating the conclusions with Matplotlib and Seaborn

# ### #1: Do wines with higher alcoholic content receive better ratings?
# Create a bar chart with one bar for low alcohol and one bar for high alcohol wine samples.

# In[116]:


# import matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid') # get a nice gray background


# In[80]:


# define the quality mean as a separated variable
low_alcohol_qlt_mean = wine_df_low_alcohol['quality'].mean()
high_alcohol_qlt_mean = wine_df_high_alcohol['quality'].mean()


# In[84]:


# create a bar chart with proper labels and title
coordinates = [1, 2]
bar_heights = [low_alcohol_qlt_mean, high_alcohol_qlt_mean]
labels = ['Low', 'High']
plt.bar(coordinates, bar_heights, tick_label = labels)
plt.title('Average quality ratings of wine by level of alcohol')
plt.xlabel('Alcohol level')
plt.ylabel('Average quality rating');


# ### #2: Do sweeter wines receive higher ratings?
# Create a bar chart with one bar for low residual sugar and one bar for high residual sugar wine samples.

# In[85]:


# again, define the quality average as a separate value
low_sugar_mean = low_sugar['quality'].mean()
high_sugar_mean = high_sugar['quality'].mean()


# In[86]:


# create a bar plot with all proper labels
coordinates_sugar = [1, 2]
heights_sugar = [low_sugar_mean, high_sugar_mean]
labels = ['Low', 'High']
plt.bar(coordinates_sugar, heights_sugar, tick_label = labels)
plt.title('Average quality rating of wine by sugar levels')
plt.xlabel('Sugar level')
plt.ylabel('Average quality rating')


# ### #3: What level of acidity receives the highest average rating?
# Create a bar chart with a bar for each of the four acidity levels.

# In[93]:


# define a variable for each acidity level
acidity_levels_mean_list = wine_df.groupby(['ph_category'])['quality'].mean().to_list()
print(acidity_levels_mean_list)
print(wine_df.groupby(['ph_category'])['quality'].mean())


# In[98]:


# plot the bar graphic
coordinates_acidity = [1,2,3,4]
heights_acidity = acidity_levels_mean_list
plt.bar(coordinates_acidity, heights_acidity, tick_label = bin_names)
plt.title('Average quality rating of wine by acidity level')
plt.xlabel('Acidity level')
plt.ylabel('Average quality rating')
plt.ylim(bottom=5.7, top=5.9)


# # Advanced Plotting - Plotting Wine Type and Quality with Matplotlib
# 
# ### Create arrays for red bar heights white bar heights
# Remember, there's a bar for each combination of color and quality rating. Each bar's height is based on the proportion of samples of that color with that quality rating.
# 1. Red bar proportions = counts for each quality rating / total # of red samples
# 2. White bar proportions = counts for each quality rating / total # of white samples

# In[102]:


# count the amount of times a red or white wine received an specific rating
wine_type_counts = wine_df.groupby(['color', 'quality']).count()['pH']
wine_type_counts


# since more white wines have been reviewed than red wines, we need to get the counts in relation to the total amount of counts for each type of wine

# In[106]:


# get the total amount of counts per wine type
total_count_wine_type = wine_df.groupby('color').count()['pH']
total_count_wine_type


# this is true, since it matches the number of entries. Now lets call the relative counts "proportions"

# In[107]:


# create count variables (in relation / proportion)
red_proportions = wine_type_counts['RED'] / total_count_wine_type['RED']
red_proportions


# we need to add a value of "0" for quality ratings of "9", since some wihte wines have been awarded with this qualification

# In[111]:


red_proportions['9']=0


# In[108]:


white_proportions = wine_type_counts['WHITE'] / total_count_wine_type['WHITE']
white_proportions


# **this looks great, now it is time to plot these results in a bar chart**. 
# 
# Set the x coordinate location for each rating group and width of each bar

# In[113]:


ind = np.arange(len(red_proportions)) # defining the x-coordinates 
width = 0.35 # widht of the bars


# let's create the plot! **Be careful!** the two graphs should be next to each other, which means that they are shifted (or moved) by the "width" - take this into account when defining the *x-coordinates*

# In[118]:


redwine_bars = plt.bar(ind, red_proportions, width, color = 'red', alpha = 0.7, label = 'Red Wine')
whitewine_bars = plt.bar(ind + width, white_proportions, width, color = 'white', label = 'White Wine')

# title and labels
plt.ylabel('Proportion')
plt.xlabel('Quality')
plt.title('Proportion by Wine Color and Quality')
coordinates_xlabels = ind + width / 2
xlabels_names = np.arange(3,10)
plt.xticks(coordinates_xlabels, xlabels_names);


# In[ ]:




