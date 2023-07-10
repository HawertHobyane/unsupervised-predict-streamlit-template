#!/usr/bin/env python
# coding: utf-8

# In[51]:


# Exploratory Data Analysis
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

# Data Preprocessing
import random
from time import time
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn.preprocessing import StandardScaler
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

# Models
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Performance Evaluation
from sklearn.metrics import mean_squared_error

# Ignore warnings
import warnings
warnings.simplefilter(action='ignore')


# In[52]:


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print(train_df.shape, test_df.shape)
train_df.head()


# In[53]:


#Movie Dataset


# In[54]:


movies_df =  pd.read_csv('movies.csv', index_col='movieId')
imdb_df =  pd.read_csv('imdb_data.csv', index_col='movieId')
links_df =  pd.read_csv('links.csv', index_col='movieId')
genome_scores =  pd.read_csv('genome_scores.csv', index_col='movieId')
genome_tags =  pd.read_csv('genome_tags.csv', index_col='tagId')
tags =  pd.read_csv('tags.csv')
print(movies_df.shape, imdb_df.shape, links_df.shape, genome_scores.shape, genome_tags.shape)


# In[55]:


#EDA


# In[56]:


train_df.info()


# In[57]:


test_df.info()


# In[58]:


movies_df.info()


# In[59]:


imdb_df.info()


# In[60]:


links_df.info()


# In[61]:


genome_scores.head()


# In[62]:


genome_tags.info()


# In[63]:


tags.info()


# In[64]:


print("Train: ")
print(str(train_df.isnull().sum()))
print("============")
print("Test: ")
print(str(test_df.isnull().sum()))
print("============")
print("Movies: ")
print(str(movies_df.isnull().sum()))
# print("============")
# print("Tags: ")
# print(str(tags_df.isnull().sum()))
print("============")
print("Links: ")
print(str(links_df.isnull().sum()))
print("============")
print("IMDB: ")
print(str(imdb_df.isnull().sum()))
print("============")
print("Genome scores: ")
print(str(genome_scores.isnull().sum()))
print("============")
print("Genome tags: ")
print(str(genome_tags.isnull().sum()))
print("============")
print("tags: ")
print(str(tags.isnull().sum()))


# In[65]:


#Drop missing rows
#tags_df.dropna(axis=0, inplace=True)
links_df.dropna(axis=0,inplace=True)
tags.dropna(axis=0,inplace=True)


# In[66]:


#Most rated Movies by users


# In[67]:


def user_ratings_count(df, n):
    plt.figure(figsize=(12,10))
    data = df['userId'].value_counts().head(n)
    ax = sns.barplot(x = data.index, y = data, order= data.index, palette='brg', edgecolor="black")
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), fontsize=12, ha='center', va='bottom')
    plt.title(f'Top {n} Users by Number of Ratings', fontsize=14)
    plt.xlabel('User ID')
    plt.ylabel('Number of Ratings')
    print("Combined number of ratings:\t",df['userId'].value_counts().head(n).sum(),
         "\nTotal number of movies:\t\t", df['movieId'].nunique())
    plt.show()


# In[68]:


user_ratings_count(train_df,10)


# In[69]:


# Exclude user 72315 for EDA
# user 72315 is considered as outlier/anomaly/noise within the data frame.
eda_df = train_df[train_df['userId']!=72315]


# In[70]:


# Distribution of ratings without the outliers
user_ratings_count(eda_df,10)


# In[71]:


# How many ratings have we lost?
ratings_lost = 38970 - 28296
print(ratings_lost)  ## ratings lost from removing the outlier


# In[72]:


#Check how do users tend to rate movies?


# In[73]:


def ratings_distplot(df, column='rating'):
    plt.figure(figsize=(8,6))
    ax = sns.distplot(df[f'{column}'],bins=10, kde=False, hist_kws=dict(alpha=0.6),color="#4D17A0")
    mean = df[f'{column}'].mean()
    median = df[f'{column}'].median()
    plt.axvline(x=mean, label = f'mean {round(mean,2)}' , color='#4D17A0', lw=3, ls = '--')
    plt.axvline(x=median, label = f'median {median}' , color='#4DA017', lw=3, ls = '--')
    plt.xlim((0.5,5))
    plt.ylim((0,2500000))
    plt.title(f'Distribution of Ratings', fontsize=16)
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


# In[74]:


ratings_distplot(eda_df)


# In[75]:


# Here we check if there is a relationship between the number of movies a user has rated and the rating that they give?


# In[76]:


def mean_ratings_scatter(df, color='#4DA017', column='userId'):
    plt.figure(figsize=(6,4))
    mean_ratings = df.groupby(f'{column}')['rating'].mean()
    user_counts = df.groupby(f'{column}')['movieId'].count().values
    sns.scatterplot(x=mean_ratings, y = user_counts, color=color)
    plt.title(f'Mean Ratings by Number of Ratings', fontsize=14)
    get_ipython().set_next_input("    plt.xlabel('Rating')Qs:Here we check if there is a relationship between the number of movies a user has rated and the rating that they give");get_ipython().run_line_magic('pinfo', 'give')
    plt.ylabel('Number of Ratings')
    plt.show()


# In[77]:


# Mean user ratings by number of ratings
mean_ratings_scatter(eda_df,'#4D17A0')


# In[78]:


# Mean movie ratings by number of ratings
mean_ratings_scatter(eda_df, column='movieId')


# In[79]:


# Which are the best and worst rated movies of all time?


# In[80]:


def plot_ratings(count, n, color='#4DA017', best=True, method='mean'):
    # What are the best and worst movies
    # Creating a new DF with mean and count
    if method == 'mean':
        movie_avg_ratings = pd.DataFrame(eda_df.join(movies_df, on='movieId', how='left').groupby(['movieId', 'title'])['rating'].mean())
    else:
        movie_avg_ratings = pd.DataFrame(eda_df.join(movies_df, on='movieId', how='left').groupby(['movieId', 'title'])['rating'].median())
    movie_avg_ratings['count'] = eda_df.groupby('movieId')['userId'].count().values
    movie_avg_ratings.reset_index(inplace=True)
    movie_avg_ratings.set_index('movieId', inplace=True)

    # Remove movies that have been rated fewer than n times
    data = movie_avg_ratings[movie_avg_ratings['count']>count]
    data.sort_values('rating', inplace= True,ascending=False)
    if best == True:
        plot = data.head(n).sort_values('rating', ascending=True)
        title='Best Rated'
    else:
        plot = data.tail(n).sort_values('rating', ascending=False)
        title='Worst Rated'
    plt.figure(figsize=(9,5))
    sns.scatterplot(x=plot['rating'], y=plot['title'], size=plot['count'], color=color)
    plt.xlabel('Rating')
    plt.ylabel('')
    plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
    plt.title(f'Top {n} {title} Movies with Over {count} Ratings', fontsize=20)
    plt.show()


# In[81]:


# What are the top 10 highest rated titles?
plot_ratings(10000, 10, '#4D17A0', True, 'mean')


# In[82]:


# What are the 10 worst rated titles?
plot_ratings(500, 10,'#4DA017', False, 'mean')


# In[83]:


#Percentage of users per rating

movieRatingDistGroup = train_df['rating'].value_counts().sort_index().reset_index()
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(data=movieRatingDistGroup, x='index', y='rating', palette="brg", edgecolor="black", ax=ax)
ax.set_xlabel("Rating")
ax.set_ylabel('Number of Users')
ax.set_yticklabels(['{:,}'.format(int(x)) for x in ax.get_yticks().tolist()])
total = float(movieRatingDistGroup['rating'].sum())
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2., height+350, '{0:.2%}'.format(height/total), fontsize=11, ha="center", va='bottom')
plt.title('Number of Users Per Rating', fontsize=14)
plt.show()


# In[84]:


def feature_frequency(df, column):
    # Creat a dict to store values
    df = df.dropna(axis=0)
    genre_dict = {f'{column}': list(),
                 'count': list(),}
    # Retrieve a list of all possible genres
    print('retrieving features...')
    for movie in range(len(df)):
        gens = df[f'{column}'].iloc[movie].split('|')
        for gen in gens:
            if gen not in genre_dict[f'{column}']:
                genre_dict[f'{column}'].append(gen)
    # count the number of occurences of each genre
    print('counting...')
    for genre in genre_dict[f'{column}']:
        count = 0
        for movie in range(len(df)):
            gens = df[f'{column}'].iloc[movie].split('|')
            if genre in gens:
                count += 1
        genre_dict['count'].append(count)
        
        # Calculate metrics
    data = pd.DataFrame(genre_dict)
    print('done!')
    return data
genres = feature_frequency(movies_df, 'genres')


# In[85]:


def feature_count(df, column):
    plt.figure(figsize=(10,6))
    ax = sns.barplot(y = df[f'{column}'], x = df['count'], palette='brg', orient='h')
    plt.title(f'Number of Movies Per {column}', fontsize=14)
    plt.ylabel(f'{column}')
    plt.xlabel('Count')
    plt.show()


# In[86]:


feature_count(genres.sort_values(by = 'count', ascending=False), 'genres')


# In[87]:


#  Who are the most common directors?


# In[88]:


def count_directors(df, count = 10):
    directors = pd.DataFrame(df['director'].value_counts()).reset_index()
    directors.columns = ['director', 'count']
    # Lets only take directors who have made 10 or more movies otherwise we will have to analyze 11000 directors
    directors = directors[directors['count']>=count]
    return directors.sort_values('count', ascending = False)
directors = count_directors(imdb_df)


# In[89]:


feature_count(directors.head(10), 'director')


# In[90]:


def dir_mean(df):
    df.set_index('director', inplace=True)

    direct_ratings = []
    directors_eda = eda_df.join(imdb_df, on = 'movieId', how = 'left')
    for director in df.index:
        rating = round(directors_eda[directors_eda['director']==director]['rating'].mean(),2)
        direct_ratings.append(rating)
    df['mean_rating'] = direct_ratings
    return df.sort_values('mean_rating', ascending = False)


# In[91]:


directors = dir_mean(directors)
directors.head()


# In[92]:


# Subset the data to cut down computation time for now
genome_score = genome_scores[:10000000]


# In[93]:


# Although scores are in the range of 0-1, there is no harm in scaling
scaler_mds = StandardScaler()
mds_genome = scaler_mds.fit_transform(genome_score.sample(frac=0.0001))


# In[94]:


tsne = TSNE(3, n_jobs = -1, verbose = 2, perplexity = 10, learning_rate = 0.1)


# In[95]:


tsne.fit(mds_genome)


# In[ ]:




