
# coding: utf-8

# # Tutorial: Foundations of autonomic nervous system activity analysis 
# ## Transformative Technology Conference 2018
# ## Watson Ξ 2018.11.10

# This notebook is a tutorial on some basic analysis (and data plumbling) of biometric data from a bluetooth heart rate strap, which delivers RRI (R-peak to R-peak heart beat interval) data on a beat by beat basis. This data can be analyzed and compared with conditions under which it was gathered to begin to understand how to infer subjective cognitive and emotional state through dynamic physiological biometric proxy signals. 
# 
# Technical recommedations:
# * Polar chest strap
# * Jupyter notebook via Anaconda Python distribution
# 
# Note: this looks best with a Jupyter dark theme: https://github.com/dunovank/jupyter-themes

# In[178]:


import json
import numpy as np
import pandas as pd


# In[179]:


# set up some plotting configuration
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('figure', figsize=(21, 9)) #set figure size

plt.rcParams['axes.facecolor'] = 'black'
mpl.rcParams['text.color'] = 'yellow'


# # Read in RRI data and plot
# ### (first take a look at the data file)

# In[180]:


data_json = json.load(open('labeled_data_RRI.json','r'))

print("{:<10} {:<10} {:<10} {:<10} {:<10}".format('collect #','condition','length','minimum','maximum'))
for k in data_json.keys():
    for l in data_json[k].keys():
        print("{:<10} {:<10} {:<10} {:<10} {:<10}".format(k,l,len(data_json[k][l]['data']),min(data_json[k][l]['data']),max(data_json[k][l]['data'])))


# In[181]:


condition_keys = ['pre','ex1','ex2','post']
collection_keys = ["collect1","collect2","collect3","collect4"]

data_json = json.load(open('labeled_data_RRI.json','r'))
for k in collection_keys:
    for k2 in condition_keys:
        plt.figure()
        plt.plot(data_json[k][k2]['data'])
        plt.title("{} {}".format(k,data_json[k][k2]['description']))
        plt.xlabel('interval index')
        plt.ylabel('ms')


# In[182]:


for k2 in condition_keys:
    plt.figure()
    plt.title(k2)
    plt.xlabel('interval index')
    plt.ylabel('ms')
    for k in collection_keys:
        plt.plot(data_json[k][k2]['data'])


# # Load into Pandas Dataframes for convenient manipulation

# In[183]:


data_df = {} #hold each of our collections

for c in collection_keys:
    print(c)
    #gather the data for this specific collect keyed by condition
    collect_data_dict = {k:v['data'] for k,v in data_json[c].items()}
    data_df[c] = pd.DataFrame(collect_data_dict,columns=condition_keys) #columns in logical order
    print(data_df[c].head())


# ## Pandas can tell us about the data very easily
# ### Some basic stats we get "for free"

# In[184]:


for c in collection_keys:
    print(c)
    display(data_df[c].describe())


# # Plot running windowed STD (standard deviation) of RRI

# In[185]:


#mpl.rc('figure', figsize=(21, 9)) #set figure size

for c in collection_keys:
    data_df[c].rolling(window=120).std().plot()
    plt.title("HRV SDRR for {}".format(c))
    plt.xlabel('interval index')
    plt.ylabel('ms')


# In[186]:


for c in collection_keys:
    print(c)
    print(data_df[c].std()) #index into data to skip "transient": data_df[c][:50]


# # Plot running windowed RMSSD (root mean square of successive differences)

# In[187]:


def root_mean_square(vals):
    return np.sqrt(np.mean(np.square(vals)))

for c in collection_keys:
    diff_df = data_df[c] - data_df[c].shift(1)
    diff_df.rolling(window=120).apply(func=root_mean_square).plot()
    plt.title("HRV RMSSD for {}".format(c))
    plt.xlabel('interval index')
    plt.ylabel('ms')


# In[188]:


for c in collection_keys:
    print(c)
    diff_df = data_df[c] - data_df[c].shift(1)
    print(diff_df.apply(func=root_mean_square)) #may want to skip transient here too (see above)


# # Plot Poincare Return Map of RRI

# In[189]:


mpl.rc('figure', figsize=(21, 21)) #set figure size
rri_min = 500
rri_max = 1200

#sort keys / put in array manually
for c in collection_keys:
    for k in condition_keys:
        plt.figure()
        plt.scatter(data_df[c][k],data_df[c][k].shift(1))
        plt.xlim(rri_min,rri_max)
        plt.ylim(rri_min,rri_max)
        plt.xlabel('previous interval')
        plt.ylabel('current interval')
        plt.title("{} {}".format(c,k))


# In[190]:


rri_min = 300
rri_max = 1200

for c in collection_keys:
    for k in condition_keys:
        plt.figure()
        plt.hist2d(data_df[c][k][:-1],data_df[c][k][1:],bins=20);
        plt.xlabel('previous interval')
        plt.ylabel('current interval')
        plt.title("{} {}".format(c,k))#         
        plt.xlim(rri_min,rri_max)
        plt.ylim(rri_min,rri_max)


# ## Outlier in collection 2, ex1 is causing issues... lets prune it
# ### Logical treatment: interpolate the two values around it
# #### Rerun cells above after

# In[192]:


print(data_df['collect2'].min())
print(data_df['collect2'].idxmin())
data_df['collect2']['ex1'][28]
data_df['collect2']['ex1'][28] = (data_df['collect2']['ex1'][27] + data_df['collect2']['ex1'][29])/2
data_df['collect2']['ex1'][28]


# # Homework (student exercises)
# 
# ## statistical tests (e.g. t-test)
#     within each collect (aggregate RRIs)
#     across multiple collects (aggregate the outputs 
# ## box plot
# ## categorization (use sci-kit learn)
# ## freq domain HRV 
# ## visualization 
#     e.g.: extension of Poincare: current vs weighted average of previous several values
#     e.g.2: visualize in freq domain PSD
# ## threshold crossing: play with when do you shift into another state
