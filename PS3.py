#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Question 1
import pandas as pd

url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user'
users = pd.read_csv(url,delimiter = '|')

# Calculate mean age per occupation
mean_age = users.groupby('occupation')['age'].mean()

print(mean_age)


# In[20]:


# Calculate male ratio per occupation
male_ratio = users.groupby('occupation')['gender'].agg(lambda x: (x == 'M').sum() / len(x))

# Sort by male ratio in descending order
male_ratio = male_ratio.sort_values(ascending=False)

print(male_ratio)


# In[21]:


# Calculate minimum and maximum ages per occupation
min_max_age = users.groupby('occupation')['age'].agg(['min', 'max'])

print(min_max_age)


# In[22]:


# Calculate mean age per occupation and sex
mean_age_by_sex = users.groupby(['occupation', 'gender'])['age'].mean()

print(mean_age_by_sex)


# In[5]:


# Calculate percentage of women and men per occupation
total_by_occupation_gender = users.groupby(['occupation', 'gender']).agg({'gender': 'count'})
total_by_occupation = users.groupby('occupation').agg('count')
percentage_by_occupation_gender = (total_by_occupation_gender.div(total_by_occupation, level = "occupation") * 100).round(2)

print(percentage_by_occupation_gender)


# In[8]:


#question2
import pandas as pd

url = "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv"
euro12 = pd.read_csv(url)


# In[9]:


# Select only the "Goal" column
goals = euro12["Goals"]

print(goals.head())


# In[10]:


# Get the number of unique teams in the "Team" column
num_teams = euro12["Team"].nunique()

print("Number of teams: ", num_teams)


# In[11]:


# Get the number of columns in the Data
num_cols = euro12.shape[1]

print("Number of columns: ", num_cols)


# In[12]:


# Select only the "Team", "Yellow Cards", and "Red Cards" columns
discipline = euro12[["Team", "Yellow Cards", "Red Cards"]]

print(discipline.head())


# In[13]:


# Sort the teams by "Red Cards" first, then by "Yellow Cards"
discipline = euro12[["Team", "Yellow Cards", "Red Cards"]].sort_values(["Red Cards", "Yellow Cards"], ascending=[False, False])

print(discipline.head())


# In[14]:


# Calculate the mean "Yellow Cards" given per team
mean_yellow_cards = euro12["Yellow Cards"].mean()

print("Mean Yellow Cards per team:", mean_yellow_cards)


# In[15]:


# Select the teams that start with "G"
g_teams = euro12[euro12["Team"].str.startswith("G")]

print(g_teams)


# In[16]:


# Select the first 7 columns
first_7_cols = euro12.iloc[:, :7]

print(first_7_cols.head())


# In[23]:


# Select all columns except the last 3
all_except_last_3 = euro12.iloc[:, :-3]

print(all_except_last_3.head())


# In[18]:


# Select the Shooting Accuracy for England, Italy, and Russia
shooting_accuracy = euro12.loc[euro12['Team'].isin(['England', 'Italy', 'Russia']), ['Team', 'Shooting Accuracy']]

print(shooting_accuracy)


# In[24]:


#Question 3

import pandas as pd
import numpy as np
import random

#Create 3 differents Series, each of length 100\
first = pd.Series(np.random.randint(1,4,100))
second = pd.Series(np.random.randint(1,3,100))
third = pd.Series(np.random.randint(1000,30000,100))

#Create a DataFrame by joinning the Series by column
table = pd.concat([first,second,third],axis=1)
table


# In[26]:


#Change the name of the columns to bedrs, bathrs, price_sqr_meter
table.columns = ['bedrs','bathrs','price_sqr_meter']
table.head()


# In[27]:


#Create a one column DataFrame with the values of the 3 Series and assign it to 'bigcolumn'
bigcolumn = pd.concat([first,second,third],axis=0)
bigcolumn


# In[28]:


#Ops it seems it is going only until index 99. Is it true?
len(bigcolumn)


# In[29]:


#Reindex the DataFrame so it goes from 0 to 299
bigcolumn.reset_index(drop=True, inplace=True)
bigcolumn


# In[62]:


#Question 4
#Wind Statistics

#Step 1. Import the necessary libraries


import pandas as pd
import numpy as np

#Step 2. Import the dataset from the attached file wind.txt Step 3. Assign it to a variable called data and replace the first 3 columns by a proper datetime index

with open('/Users/Architgoel/Desktop/wind.txt', 'r') as file:
    content = file.read() 


# In[34]:


data


# In[39]:


#Question 5

import pandas as pd
url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv"
chipo = pd.read_csv(url, delimiter='\t')
chipo.head(10)


# In[40]:


#What is the number of observations in the dataset?
chipo.shape[0]


# In[41]:


#number of columns in the dataset
chipo.shape[1]


# In[42]:


#names of all the columns.
chipo.columns


# In[43]:


#How is the dataset indexed?
chipo.index


# In[44]:


#the most-ordered item
chipo.item_name.value_counts().head(1)


# In[45]:


#For the most-ordered item, how many items were ordered?
chipo.groupby('item_name')['quantity'].sum().max()


# In[46]:


#the most ordered item in the choice_description column
chipo.choice_description.value_counts().head()


# In[47]:


#How many items were orderd in total?
chipo.quantity.sum()


# In[50]:


#Question 6
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
USMD = pd.read_csv(r"/Users/Architgoel/Downloads/us-marriages-divorces-1867-2014.csv")
USMD.info()



# In[51]:


USMD.head()


# In[52]:


years = USMD['Year']
marriages = USMD['Marriages_per_1000']
divorces = USMD['Divorces_per_1000']
USMD = plt.figure(figsize=(16,8))
USMD = plt.plot(years, marriages, label='Marriages')
USMD = plt.plot(years, divorces, label='Divorces')
USMD = plt.title("Number of marriages and divorces per capita in the U.S. between 1867 and 2014")
USMD = plt.xlabel("Years",fontsize=14)
USMD = plt.legend(fontsize = 12, loc = "upper left")
USMD = plt.ylabel("Marriages",fontsize=14)
USMD = plt.grid(True)
USMD


# In[53]:


#Question 7

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
USMD = pd.read_csv(r"/Users/Architgoel/Downloads/us-marriages-divorces-1867-2014.csv")
Avar = USMD.loc[USMD.Year.isin([1900,1950,2000]),['Year','Marriages_per_1000','Divorces_per_1000']]
Avar


# In[54]:


Avar = Avar[Avar['Year'].apply(lambda x: x in [1900, 1950, 2000])]
years = Avar['Year']
marriages = Avar['Marriages_per_1000']
divorces = Avar['Divorces_per_1000']
Avar = plt.figure(figsize= (16,8))
Avar = plt.bar(years, marriages, label ='Marriages')
Avar = plt.bar(years, divorces, label = 'Divorces')
Avar = plt.title("Number of marriages and divorces per capita in the U.S. between 1990,1950 and 2000", fontsize=16)
Avar = plt.xlabel("Years", fontsize=14)
Avar = plt.legend(fontsize = 12, loc = "upper left")
Avar


# In[56]:


#Question 8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dead_actors = pd.read_csv(r"/Users/Architgoel/Downloads/actor_kill_counts.csv")
dead_actors


# In[57]:


actors = dead_actors['Actor']
killCount = dead_actors['Count']
dead_actors = plt.figure(figsize=(12,6))
dead_actors = plt.barh(actors, killCount, label='actor')
#dead_actors = plt.barh(killCount, width=0.5, labe;='kill Count')
dead_actors = plt.title("The deadliest actors in Hollywood", fontsize=18)
dead_actors = plt.xlabel("kill count", fontsize=14)
dead_actors = plt.legend(fontsize = 12, loc = "upper right")
dead_actors


# In[59]:


#Question 9 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r"/Users/Architgoel/Downloads/roman-emperor-reigns.csv")
data = data[data["Cause_of_Death"] == "Assassinated"]

# Create the pie chart
patches, texts = plt.pie(data.Length_of_Reign)

# Create a legend outside the pie chart
plt.legend(data.Emperor, loc="center left", bbox_to_anchor=(1, 0.5))

plt.show()



# In[60]:


#Question 10

data = pd.read_csv(r"/Users/Architgoel/Downloads/arcade-revenue-vs-cs-doctorates.csv")



# In[61]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

colours = ['#C4E57D','#2AC280','#FFAE39','#DC5B3B','#52E397','#C5CBA3','#9CD5F6','#6E50D9','#9A5E59','#9BC8F5']

data.plot.scatter(x='Total Arcade Revenue (billions)',

y='Computer Science Doctorates Awarded (US)', c=colours , s = 50, figsize = (5,4))

plt.title('Revenue Vs CS Doctorates', color = 'Blue', fontsize = 18)
plt.xlabel('Total Arcade Revenue (billions)' , color = 'Red', fontsize = 12)
plt.ylabel('CS Doctorates Awarded', color = 'Red', fontsize = 12)


# In[ ]:




