#!/usr/bin/env python
# coding: utf-8

# In[14]:


import re
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.cluster.hierarchy import fcluster
import scipy.cluster.hierarchy as cluster_model

data = pd.read_csv('online_retail.csv',  parse_dates = ["InvoiceDate"])
print(data.shape)
data.head(10)


# In[15]:


data.info(memory_usage="deep")


# In[17]:


data["InvoiceDate"].describe()


# In[18]:


print(f"Number of unique transactions: {data['InvoiceNo'].nunique()}")


# In[19]:


data["first_letter"] = data['InvoiceNo'].apply(lambda x: "" if x[0].isdigit() else x[0])
data["first_letter"].value_counts()


# In[20]:


data.drop_duplicates(subset="InvoiceNo")["first_letter"].value_counts()


# In[22]:


data['cancellation'] = data['InvoiceNo'].apply(lambda x: 1 if x[0]=='C' else 0)


# In[23]:


display(data[data["first_letter"]=="A"])
display(data[data["UnitPrice"]<0])


# In[24]:


data = data[data["first_letter"]!="A"]
print(data.shape)


# In[25]:


data['test__InvoiceNo_int'] = data['InvoiceNo'].apply(lambda x: int(x) if x[0] not in ('C', 'A') else int(x[1:]))


# In[26]:


data_InvoiceDate_stat = data.groupby("InvoiceNo").agg({'InvoiceDate': ['min', 'max']}).reset_index()
(data_InvoiceDate_stat["InvoiceDate"]["max"] - data_InvoiceDate_stat["InvoiceDate"]["min"]).value_counts()


# In[27]:


data.drop_duplicates(subset="InvoiceNo").iloc[55:60]


# In[28]:


data.drop_duplicates(subset="InvoiceNo")["test__InvoiceNo_int"].diff().iloc[55:60]


# In[29]:


invoiceno_uniqdf = data.drop_duplicates(subset="InvoiceNo")


# In[30]:


invoiceno_uniqdf["test__InvoiceNo_int_diff"] = invoiceno_uniqdf["test__InvoiceNo_int"].diff()
invoiceno_uniqdf["InvoiceDate_day"] = invoiceno_uniqdf["InvoiceDate"].dt.date
invoiceno_uniqdf.head(3)


# In[31]:


invoiceno_uniqdf[["InvoiceDate", "test__InvoiceNo_int_diff"]]
plt.figure(figsize=(20, 5))
plt.plot(invoiceno_uniqdf["InvoiceDate"], invoiceno_uniqdf["test__InvoiceNo_int_diff"]);


# In[32]:


plt.figure(figsize=(30, 5))
g = sns.scatterplot(
    x = "InvoiceDate",
    y = "test__InvoiceNo_int_diff",
    data=invoiceno_uniqdf[invoiceno_uniqdf["InvoiceDate"].dt.month==3]
)
g.xaxis.set_major_locator(ticker.MultipleLocator(2))
plt.show()

plt.figure(figsize=(30, 5))
g1 = sns.countplot(
    x = "InvoiceDate_day",
    data=invoiceno_uniqdf[invoiceno_uniqdf["InvoiceDate"].dt.month==3]
)
g1.xaxis.set_major_locator(ticker.MultipleLocator(1))
plt.show()


# In[33]:


filter_expr = (invoiceno_uniqdf["cancellation"]==1)

plt.figure(figsize=(20, 5))
g = sns.countplot(
    x = "InvoiceDate_day",
    data=invoiceno_uniqdf[filter_expr]
)
g.xaxis.set_major_locator(ticker.MultipleLocator(20))
plt.show()


# In[34]:


filter_expr = (invoiceno_uniqdf["InvoiceDate"].dt.month==3) & (invoiceno_uniqdf["cancellation"]==1)
plt.figure(figsize=(20, 5))
g = sns.countplot(
    x = "InvoiceDate_day",
    data=invoiceno_uniqdf[filter_expr]
)
g.xaxis.set_major_locator(ticker.MultipleLocator(2))
plt.show()


# In[35]:


data = data.drop(["test__InvoiceNo_int", "first_letter"], axis=1)


# In[36]:


data.head(5)


# In[37]:


data["StockCode"].nunique(), data["Description"].nunique()


# In[38]:


data[data["StockCode"].isin(["84406B"])][["StockCode", "Description"]].drop_duplicates()


# In[39]:


data[
    (data["StockCode"].isin(["84406B"])) &
    (data["Description"].isin(["incorrectly made-thrown away.", "?", np.nan]))
]


# In[40]:


unique_descr = sorted(data["Description"].dropna().unique(), key=len)
print(unique_descr[:150])


# In[41]:


data[
    data["Description"].isin(unique_descr[:15])
]["UnitPrice"].describe()


# In[42]:


data[data["cancellation"]==1].sample(5)


# In[43]:


data[data["cancellation"]==1]["Quantity"].describe()


# In[45]:


data = data.drop(data[(data["UnitPrice"]==0) &
     (data["Description"].isnull()) & 
     (data["CustomerID"].isnull())
    ].index)
print(data.shape)


# In[46]:


data[(data["cancellation"]!=1) & (data["Quantity"]<0)]["CustomerID"].value_counts()


# In[47]:


data[(data["cancellation"]!=1) & (data["Quantity"]<0)]["Description"].value_counts().head(10)


# In[48]:


data = data.drop(data[(data["cancellation"]!=1) & (data["Quantity"]<0)].index)
print(data.shape)


# In[49]:


data[(data["cancellation"]!=1)]["Quantity"].describe()


# In[50]:


print(sorted(data["Description"].dropna().unique(), key=len)[:100])


# In[51]:


data = data.drop(data[(data["CustomerID"].isnull()) & (data["UnitPrice"]==0)].index)
print(data.shape)


# In[52]:


print(sorted(data["Description"].dropna().unique(), key=len)[:100])


# In[53]:


uniq_descr = data["Description"].unique()
uniq_descr


# In[54]:


single_symbols = [
    '', 'u','a','y','g','j','k','v','w','z','c','h',
    'r','f','p','o','m','t','l','b','i','e','s','d','x','n',
    'if', 'it', 'to', 'or', 'of', 'is', 'and', 'with'
                 ]
i = 0
uniq_words = []
uniq_descr_ = []
for descr in uniq_descr:
    descr_ = re.split('[^a-z]', descr.lower().strip())
    descr_ = [i for i in descr_ if i not in single_symbols]
    uniq_descr_.append(descr_)
    for word in descr_:
        uniq_words.append(word)
uniq_words = set(uniq_words)

# create zeros dataframe
full_vectorize = pd.DataFrame(
    np.zeros(shape=(len(uniq_descr_), len(uniq_words)))
)
full_vectorize.columns = uniq_words
full_vectorize.index = uniq_descr

# make correct vectors
for ind_desr, descr in tqdm(enumerate(uniq_descr_)):
    for word in descr:
        full_vectorize[word][ind_desr] = 1


# In[ ]:


full_vectorize.head(3)


# In[ ]:


plt.figure(figsize=(40, 10))
# fit hierarchical clustering from scipy
result_model = cluster_model.linkage(full_vectorize[:], method='complete', metric="cosine")
dend_cos = cluster_model.dendrogram(result_model, labels=full_vectorize.index[:])


# In[ ]:


t = 0.9
cluster_labels = fcluster(result_model, t,  criterion="inconsistent")


# In[ ]:


print(full_vectorize.shape[0], len(set(cluster_labels)))


# In[ ]:


cluster_dict = {
    i:j for i,j in zip(full_vectorize.index.tolist(), cluster_labels.tolist())
}
data["cluster_description"] = data["Description"].apply(lambda x: cluster_dict[x])


# In[ ]:


data[["Description", "cluster_description"]].sort_values(by="cluster_description").head()


# In[ ]:


data.head()


# In[ ]:


print(data["Country"].nunique())
data["Country"].value_counts().head(10)


# In[ ]:


piv_data = data[["Country", "cancellation"]].groupby("Country").agg({"cancellation": [sum, 'count']})
piv_data["cancellation", "perc"] = piv_data["cancellation"]["sum"] / piv_data["cancellation"]["count"]


# In[ ]:


piv_data.sort_values(by=[("cancellation", "perc")], ascending=False)


# In[ ]:


data.head()


# In[ ]:




