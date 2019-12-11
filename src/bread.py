import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.listdir("./input"))


# import the data
Bak=pd.read_csv('./input/BreadBasket_DMS.csv')
Bak.head()

# Inspect the data
Bak.loc[Bak['Item']=='NONE',:].head()
Bak.loc[Bak['Item']=='NONE',:].count()
# Drop none values from the dataset
Bak=Bak.drop(Bak.loc[Bak['Item']=='NONE'].index)

Bak['Year'] = Bak.Date.apply(lambda x:x.split('-')[0])
Bak['Month'] = Bak.Date.apply(lambda x:x.split('-')[1])
Bak['Day'] = Bak.Date.apply(lambda x:x.split('-')[2])
Bak['Hour'] =Bak.Time.apply(lambda x:int(x.split(':')[0]))
#df = df.drop(columns='Time')
Bak.head()


print('Total number of Items sold at the bakery is:',Bak['Item'].nunique())
# print('List of Items sold at the bakery:')
# Bak['Item'].unique()
# print('List of Items sold at the Bakery:\n')
# for item in set(Bak['Item']):
#     print(item)

print('Ten Most Sold Items At The Bakery')
print(Bak['Item'].value_counts().head(10))



fig, ax=plt.subplots(figsize=(16,7))
Bak['Item'].value_counts().sort_values(ascending=False).head(20).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)
plt.xlabel('Food Item',fontsize=20)
plt.ylabel('Number of transactions',fontsize=20)
ax.tick_params(labelsize=20)
plt.title('20 Most Sold Items at the Bakery',fontsize=25)
plt.grid()
plt.ioff()
plt.show()


print(Bak.groupby('Month').nunique())

# Transactions per hour
# use sns to plot the counts bar
# sns.set(style="darkgrid")
plt.subplots(figsize=(16,7))
plt.title('Transactions per hour',fontsize=25)
trans_hour = sns.countplot(x="Hour", data=Bak)
plt.show()


# Transactions per month
plt.subplots(figsize=(16,6))
plt.xlabel('Months',fontsize=18)
plt.ylabel('Number of transactions per month',fontsize=18)
plt.title('Transactions per month',fontsize=25)
Month_counts = sns.countplot(x="Month", data=Bak)
plt.show()

## caculate de transactions per day in every months
list_trans = Bak.groupby('Month')['Transaction'].nunique().tolist()
list_days = Bak.groupby('Month')['Day'].nunique().tolist()
list_trans_per_day = [x / y for x, y in zip(list_trans, list_days)]

print(list_trans_per_day)
months = Bak.groupby('Month').nunique().index.tolist()
fig, Month=plt.subplots(figsize=(16,7))
Month = plt.bar(months,list_trans_per_day, color=['c'], edgecolor='k')
plt.xlabel('Months',fontsize=16)
plt.ylabel('Number of transactions per day',fontsize=16)
plt.title('Number of transactions per day in different months',fontsize=25)
plt.grid()
plt.show()

# Sales on different days of the week
Bak1 = Bak.groupby(['Date']).size().reset_index(name='counts')
Bak1['Day'] = pd.to_datetime(Bak1['Date']).dt.day_name()
#Bak1


plt.figure(figsize=(20,15))
plt.subplot(2,2,1)
ax=sns.boxplot(x='Day',y='counts',data=Bak1,width=0.8,linewidth=2)
plt.xlabel('Day of the Week',fontsize=15)
plt.ylabel('Total Sales',fontsize=15)
plt.title('Sales on Different Days of Week',fontsize=20)
ax.tick_params(labelsize=10)
plt.grid()
plt.ioff()
# plt.show()

# apriori
from mlxtend.frequent_patterns import apriori, association_rules
# transfrom data to make items as columns and each transaction as a row and count same Items bought in one transaction but fill other cloumns of the row with 0 to represent item which are not bought.
hot_encoded_Bak=Bak.groupby(['Transaction','Item'])['Item'].count().unstack().reset_index().fillna(0).set_index('Transaction')
hot_encoded_Bak.head()
print(hot_encoded_Bak.head())

hot_encoded_Bak = hot_encoded_Bak.applymap(lambda x: 0 if x<=0 else 1)

# here we choose the min support as 1%
frequent_itemsets = apriori(hot_encoded_Bak, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print(rules.head(10))

rules[ (rules['lift'] >= 1.17) &
       (rules['confidence'] >= 0.5) ]

# print(rules[ (rules['lift'] >= 1.17) &
#        (rules['confidence'] >= 0.5) ])

# Support can be thought of as the percentage of the total amount of transactions relevant to an association. This is perhaps better understood by a simple equation:     
# Support(Item1) = (Transactions containing Item1) / (Total transactions) 
# Confidence tells us how likely it is that purchasing Item1 results in a purchase of Item2.
# Confidence(Item1 -> Item2) = (Transactions containing both Item1 and Item2) / (Transactions containing Item1)
# A Lift of 1 means there is no association between products A and B. Lift of greater than 1 means products A and B are more likely to be bought together. Finally, Lift of less than 1 refers to the case where two products are unlikely to be bought together.


Top_items=Bak['Item'].value_counts().head(10).index.tolist()
# print(Top_items)

print(type(Top_items))
print(Top_items)

Hour_by_Item=Bak[['Hour','Item','Transaction']].groupby(['Hour','Item'],as_index=False).sum()
print(Hour_by_Item.head())
plt.figure(figsize=[13,5])
plt.ticklabel_format(style='plain', axis='y')
plt.title('Sale by Hour for Top 10 Items')
sns.lineplot(x='Hour',y='Transaction',data=Hour_by_Item[Hour_by_Item['Item'].isin(Top_items)],hue='Item')
plt.show()