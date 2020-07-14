#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


insurance=pd.read_csv("/Users/apple/Downloads/DS_Projects/Insurance Project/Insurance Dataset Project.csv")
insurance.head()


# ## EDA

# In[3]:


insurance.shape


# In[4]:


insurance["Result"].hist(figsize=(20,8))
plt.show()


# In[5]:


insurance.info()


# In[6]:


insurance.isnull().sum()


# In[7]:


len(insurance)


# In[3]:


#finding the missing values in percentage
percent_missing=insurance.isnull().sum()*100/len(insurance)
percent_missing
#percent_missing_df=pd.DataFrame({"column_names":insurance.columns,"percent_missing":percent_missing})
#percent_missing_df


# In[4]:


percent_missing_df=pd.DataFrame({"column_names":insurance.columns,"percent_missing":percent_missing})
#percent_missing_df
percent_missing_df.sort_values("percent_missing",inplace=True)
percent_missing_df


# In[6]:


col=list(insurance.columns)
col


# In[7]:


#finding the number of duplicates rows
duplicate_rows=insurance[insurance.duplicated()]
duplicate_rows.shape


# In[5]:


#shape after removing duplicates values
insurance1=insurance.drop_duplicates(keep='first')
insurance1.shape


# In[13]:


insurance1.nunique()


# In[14]:


#check each row unquie value
for c in col:
    print(insurance1[c].unique())


# In[6]:


#year is same ,payment_typology_3 almost empty
insurance1.drop(columns=['year_discharge','payment_typology_3'],inplace=True)


# In[10]:


#Histogam for "Age"
insurance1["Age"].hist(figsize = (35,25))
plt.show()


# In[16]:


#Histogam for "zip_code_3_digits"
insurance1["zip_code_3_digits"].hist(figsize = (35,25))
plt.show()


# In[7]:


insurance1.loc[insurance1['zip_code_3_digits']=='OOS','zip_code_3_digits']=111


# In[8]:


insurance1['zip_code_3_digits']=insurance1['zip_code_3_digits'].astype(str)


# In[9]:


insurance1.loc[insurance1['zip_code_3_digits']=='nan','zip_code_3_digits']='102'
insurance1['zip_code_3_digits']=insurance1['zip_code_3_digits'].astype(int)
np.mean(insurance1['zip_code_3_digits'])


# In[10]:


insurance1.dropna(inplace=True)


# In[21]:


#insurance1.reset_index(drop=True).head()


# In[11]:


insurance1.isnull().sum()


# In[23]:


insurance1.shape


# In[12]:


#create a new dataframe which contain dummy variable 
dummy=pd.get_dummies(data=insurance1,columns=['Age','Gender','Cultural_group', 'ethnicity','Admission_type','Mortality risk', 'Surg_Description','Abortion', 'Emergency dept_yes/No'])
dummy.head()


# In[25]:


dummy.columns


# In[26]:


dummy.Days_spend_hsptl.unique()


# In[13]:


#since we have 120 + and the mean of rhe day spend is 5 so 120+5=125
dummy.loc[dummy['Days_spend_hsptl']=='120 +','Days_spend_hsptl']=125


# In[14]:


dummy['Days_spend_hsptl']=dummy['Days_spend_hsptl'].astype(int)


# In[18]:


np.mean(dummy['Days_spend_hsptl'])


# In[19]:


#checking the each area with Result variable (percentage value)
ar=pd.crosstab(dummy.Area_Service,dummy.Result,normalize='index').round(4)*100
ar


# In[20]:


#graph for area of service
plt.figure(figsize=(15,8))
ar.plot.bar()
plt.ylabel('percentage')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[21]:


ahr=pd.crosstab(index=[dummy['Area_Service'],dummy['Hospital County']],columns=dummy.Result,normalize='index').round(4)*100
ahr


# In[22]:


#graph for area of service
plt.figure(figsize=(55,25))
ahr.plot.bar()
plt.ylabel('percentage')
plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))


# In[15]:


insurance1.rename(columns={"Mortality risk":"Mortality_risk"},inplace=True)


# In[24]:



mrr=pd.crosstab(insurance1.Mortality_risk,insurance1.Result,normalize='index')*round(4)/100
mrr


# In[25]:


plt.figure(figsize=(35,8))
mrr.plot.bar()
plt.ylabel("percentage")
plt.legend(loc="upper left",bbox_to_anchor=(1,0.8))


# In[26]:


sdr=pd.crosstab(insurance1.Surg_Description,insurance1.Result,normalize='index')*round(4)/100
mrr


# In[27]:


plt.figure(figsize=(35,8))
sdr.plot.bar()
plt.ylabel("percentage")
plt.legend(loc="upper left",bbox_to_anchor=(1,0.8))


# In[16]:


#lets create a range of amount for hosipital cost 
#its a new coloumn we have generated based on given coloumn
bins = [0, 1000, 10000, 100000, 1000000,10000000]
names = ['T', '10T', '100T', '1M', '10M','100M']


# In[17]:


d=dict(enumerate(names,1))   #enumerate gives index to values
dummy['CostRange']=np.vectorize(d.get)(np.digitize(dummy["Tot_charg"],bins))
dummy["CostRange"]


# In[31]:


cr=pd.crosstab(dummy.CostRange,dummy.Result)
cr


# In[32]:


plt.figure(figsize=(35,8))
cr.plot.bar()
plt.ylabel('count')
plt.legend(loc="upper left",bbox_to_anchor=(1,0.5))


# In[33]:


pr=pd.crosstab(dummy.Payment_typology_1 ,dummy.Result,normalize='index')*round(4)/100
pr


# In[34]:


plt.figure(figsize=(35,8))
pr.plot.bar()
plt.ylabel("percentage")
plt.legend(loc='upper left',bbox_to_anchor=(1,0.8))


# In[35]:


clar=pd.crosstab(index=[dummy['CostRange'],insurance1['Age']],columns=dummy['Result'])
clar


# In[36]:


plt.figure(figsize=(35,8))
clar.plot.bar()
plt.ylabel('count')
plt.legend(loc='upper left',bbox_to_anchor=(1,0.8))


# In[18]:


bins = [0, 10, 25, 50, 75,100]
names = ['D0', 'D10', 'D25', 'D75', 'D100','G100']

d = dict(enumerate(names, 1))
dummy['Days_spend_range']=np.vectorize(d.get)(np.digitize(dummy['Days_spend_hsptl'], bins))


# In[19]:


dar=pd.crosstab(dummy['Days_spend_range'],dummy['Result'],normalize='index').round(4)*100
dar


# In[39]:


plt.figure(figsize=(15,8))
dar.plot.bar()
plt.ylabel('percentage')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# In[20]:


dummy["Area_Service"]=dummy["Area_Service"].astype(str)
dummy["Hospital County"]=dummy["Hospital County"].astype(str)
dummy["Hospital Name"]=dummy["Hospital Name"].astype(str)
dummy["Home or self care,"]=dummy["Home or self care,"].astype(str)
dummy["ccs_diagnosis_description"]=dummy["ccs_diagnosis_description"].astype(str)
dummy["ccs_procedure_description"]=dummy["ccs_procedure_description"].astype(str)
dummy["apr_drg_description"]=dummy["apr_drg_description"].astype(str)
dummy["apr_mdc_description"]=dummy["apr_mdc_description"].astype(str)
dummy["Description_illness"]=dummy["Description_illness"].astype(str)
dummy["Payment_typology_1"]=dummy["Payment_typology_1"].astype(str)
dummy["payment_typology_2"]=dummy["payment_typology_2"].astype(str)


# In[21]:


string_names=( 'Area_Service', 'Hospital County',  'Hospital Name',
        'Home or self care,',
        'ccs_diagnosis_description',
       'ccs_procedure_description', 'apr_drg_description',
       'apr_mdc_description',  'Description_illness',
       'Payment_typology_1','payment_typology_2','CostRange','Days_spend_range','Result')


# In[22]:


from sklearn.preprocessing import LabelEncoder
LB=LabelEncoder()
for i in string_names:
    dummy[i]=LB.fit_transform(dummy[i])


# In[23]:


dummy.head()


# In[54]:


dummy.corr()['Result']


# In[55]:


#correlation map
ax=plt.subplots(figsize=(30,20))
corr=dummy.corr()

sns.heatmap(corr)

plt.show()


# In[24]:


x=dummy.drop("Result",axis=1)


# In[25]:


Y=dummy["Result"]


# In[26]:


from sklearn import preprocessing
X=preprocessing.scale(x)
X


# In[27]:


from sklearn.decomposition import PCA


# In[28]:


pca=PCA(n_components=2)
pca


# In[29]:


PCA=pca.fit_transform(X)
PCA


# In[30]:


PCA.shape


# In[31]:


PCA=pd.DataFrame(PCA)
PCA


# In[32]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(PCA,Y,random_state=1,test_size=0.3)


# In[33]:


from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(train_x,train_y)


# In[34]:


y_pred=LR.predict(test_x)


# In[35]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_pred,test_y)


# In[36]:


from sklearn.metrics import accuracy_score
LR_score=accuracy_score(y_pred,test_y)
LR_score


# In[37]:


from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB


# In[38]:


GNB=GaussianNB()
GNB.fit(train_x,train_y)


# In[39]:


GNB_y_pred=GNB.predict(test_x)


# In[40]:


confusion_matrix(GNB_y_pred,test_y)


# In[41]:


GNB_score=accuracy_score(GNB_y_pred,test_y)
GNB_score


# In[42]:


BNB=BernoulliNB()
BNB.fit(train_x,train_y)
BNB_y_pred=BNB.predict(test_x)
confusion_matrix(BNB_y_pred,test_y)
BNB_score=accuracy_score(BNB_y_pred,test_y)
print(BNB_score)


# In[63]:


from sklearn.tree import DecisionTreeClassifier
DTC=DecisionTreeClassifier()
DTC.fit(train_x,train_y)
DTC_y_pred=DTC.predict(test_x)
confusion_matrix(DTC_y_pred,test_y)


# In[64]:


DTC_score=accuracy_score(DTC_y_pred,test_y)
DTC_score


# In[65]:


from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(n_jobs=3,n_estimators=100,criterion="entropy")
RF.fit(train_x,train_y)
RF_y_pred=RF.predict(test_x)
confusion_matrix(RF_y_pred,test_y)


# In[66]:


RF_score=accuracy_score(RF_y_pred,test_y)
RF_score


# In[ ]:


from sklearn.ensemble import BaggingClassifier
BC=BaggingClassifier(base_estimator=RF,n_estimators=100)
BC.fit(train_x,train_y)
BC_y_pred=BC.predict(test_x)
confusion_matrix(BC_y_pred,test_y)


# In[ ]:


BC_score=accuracy_score(BC_y_pred,test_y)
BC_score


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
ABC=AdaBoostClassifier(base_estimator=RF,n_estimators=100)
ABC.fit(train_x,train_y)


# In[ ]:


ABC_y_pred=ABC.predict(test_x)
confusion_matrics(ABC_y_pred,test_y)


# In[ ]:


ABC_score=accuracy_score(ABC_y_pred,test_y)
ABC_score


# In[ ]:


from sklearn.svm import SVC
SVC=SVC(kernel='linear')
SVC.fit(train_x,train_y)


# In[ ]:




