#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 17:44:21 2022

@author: chloenam
"""
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from scipy import stats 
from sklearn.decomposition import PCA 
from PIL import Image 
import requests 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from scipy.stats import ttest_ind
from scipy.stats import t 
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 

#%% seeding random number generator to my N number 
mySeed = 12314682
random.seed(mySeed)

#%% Question #1 

theArt = pd.read_csv('theArt.csv', header = None)
data = np.genfromtxt('theData.csv', delimiter = ',')

classicalArt = data[:, 0:35]
modernArt = data[:, 35:70]
print(classicalArt.shape) #300 rows, 35 columns
print(modernArt.shape) #300 rows, 35 columns 
modernArtNaNCheck = print(np.any(np.isnan(classicalArt))) #No NaN = False
classicalArtCheck = print(np.any(np.isnan(modernArt)))
classicalArtMean = print(np.mean(classicalArt)) #4.742
modernArtMean = print(np.mean(modernArt)) #4.257
print()


#making plots 
numBins = 7 
plt.hist(classicalArt, bins = numBins )
plt.title("Distribution of the Classical Art Ratings")
plt.show()
plt.hist(modernArt, bins = numBins)
plt.title("Distribution of the Modern Art Ratings")
plt.show() 

#mann whitney test
from scipy.stats import mannwhitneyu 

u_statistic, p_value = mannwhitneyu(classicalArt, modernArt)
u_statistic = u_statistic.mean()
p_value = p_value.mean()

print(p_value)
print(u_statistic)
if p_value < 0.05:
    print("The p-value is less than 0.05; there is a statistically significant difference between the two samples.")
    if np.mean(classicalArt) > np.mean(modernArt):
        print("Classical art is more liked than modern art.")
    else: 
        print("Modern art is more liked than classical art.")
else:
    print("The p-value is higher than 0.05; there is not a statistically significant difference between the two samples.")
print()

#%% Question #2

nonHumanArt = data[:, 70:92]
nonHumanArtNaNCheck = print(np.any(np.isnan(nonHumanArt))) # No NaN = False 
nonHumanArtMean = print(np.mean(nonHumanArt)) #3.333
modernArtMean = print(np.mean(modernArt)) #4.257
print()

if np.mean(nonHumanArt) > np.mean(modernArt):
    print("Animal and computer generated art is more liked than modern art.")
else:
    print("Modern art is more liked than animal and computer generated art.")

#mann whitney test 
from scipy.stats import mannwhitneyu

modernArt_flat = modernArt.flatten()
nonHumanArt_flat = nonHumanArt.flatten()

u_statistic, p_value = mannwhitneyu(modernArt_flat, nonHumanArt_flat)
u_statistic = u_statistic.mean()
p_value = p_value.mean()


print("The p-value is: ", p_value)
if p_value < 0.05:
    print("The p-value is less than 0.05; there is a statistically significant difference between the two samples.")
print()

#making plots 
numBins = 7 
plt.hist(nonHumanArt, bins = numBins )
plt.title("Distribution of the Non-Human Art Ratings")
plt.show()
plt.hist(modernArt, bins = numBins)
plt.title("Distribution of the Modern Art Ratings")
plt.show() 


#%% Question #3
#Do women give higher preference ratings than men?
#Column 217 (Column "HI"): User gender (1 = male, 2 = female, 3 = non-binary)
#Columns 1-91 preference ratings of the art 

gender = data[:,216]
art_ratings = data[:,0:91]

#put variables in a dataframe to drop nan values
df_gender = pd.DataFrame(gender)
df_art_ratings = pd.DataFrame(art_ratings)

#join both dataframes together
df_art_ratings[91] = df_gender[0]

#drop nan values
gender_ratings_data = df_art_ratings.dropna()

#locate 1 for male, and 2 for female:
male_ratings = gender_ratings_data.loc[gender_ratings_data[91] == 1]
female_ratings = gender_ratings_data.loc[gender_ratings_data[91] == 2] 

#delete values from dataframe:
del male_ratings[91]
del female_ratings[91]

#return to array:
female_ratings = female_ratings.to_numpy()
male_ratings = male_ratings.to_numpy()

#calculate the median values:
median_male = np.median(male_ratings) #4.0
median_female = np.median(female_ratings) #4.0

#Conduct the Mann-Whitney U test
U3, p3 = mannwhitneyu(female_ratings, male_ratings)
U3 = U3.mean() 
p3 = p3.mean() 

# Print the results of the test
print("U-statistic: ", U3) #8563.423076923076
print("p-value: ", p3) #0.37475428900506297

if p3 < 0.05:
    print("There is a statistically significant difference between male and female ratings.")
else:
    print("There is not a statistically significant difference between male and female ratings.")
    
numBins = 7
plt.hist(female_ratings, bins = numBins)
plt.title("Distribution of Female Art Ratings")
plt.show()
plt.hist(male_ratings, bins = numBins)
plt.title("Distribution of Male Art Ratings")
plt.show()


#%% Question #4 
#Is there a difference in preference rating of users with art education vs. none?
#Column 219 (Column "HK") shows art education

art_education = data[:, 218]
df_art_education = pd.DataFrame(art_education)

df_art_ratings[91] = df_art_education[0]

education_ratings_data = df_art_ratings.dropna()

no_education = education_ratings_data.loc[education_ratings_data[91] == 0]

del no_education[91]
some_education = education_ratings_data.loc[education_ratings_data[91].isin([1, 2, 3])]

del some_education[91]

no_education = no_education.to_numpy()
some_education = some_education.to_numpy()

median_some_education = np.median(some_education) #4.0
print(median_some_education)
median_no_education = np.median(no_education) #4.0
print(median_no_education)

U4, p4 = mannwhitneyu(no_education, some_education)
U4 = U4.mean()
p4 = p4.mean()

print("U-statistic: ", U4) #9051.4395
print("p-value: ", p4) #0.454690589

if p4 < 0.05:
    print("There is a statistically significant difference between users with no art education and some education.")
else:
    print("There is not a statistically significant difference between users with no art education and some education.")
    
numBins = 7
plt.hist(no_education, bins = numBins)
plt.title("Distribution of Art Ratings with No Art Background")
plt.show()
plt.hist(some_education, bins = numBins)
plt.title("Distribiton of Art Ratings with Some Art Background")
plt.show()



#%% Question #5
# Build a linear regression model (+ridge?) to predict art preference ratings from energy ratings only.
import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


energy_ratings = data[:, 91: 182]
art_ratings = data[:,0:91]

data5= np.column_stack((energy_ratings, art_ratings))
varCorrs = np.corrcoef(data5[:,0],data5[:,1])

energy_median = np.median(energy_ratings, axis = 0).reshape(-1,1)
art_median = np.median(art_ratings, axis = 0).reshape(-1,1)

energy_mean = np.mean(energy_ratings, axis = 0).reshape(-1,1)
art_mean = np.mean(art_ratings, axis = 0).reshape(-1,1)

data_mean = np.column_stack((energy_median, art_median))
data_median = np.column_stack((energy_median, art_median))

artSD = np.std(art_ratings, axis = 0).reshape(-1,1)
energySD = np.std(energy_ratings, axis = 0).reshape(-1,1)
data_SD = np.column_stack((energySD, artSD))

#calculating correlation matrix (pearsons r) between the variables
#correlation heat map

plt.imshow(varCorrs)
plt.colorbar()
plt.xlabel("Energy Ratings")
plt.ylabel("Art Preference Ratings")
plt.title("Pearson Correlation Matrix Heat Map")
plt.show()

#splitting mean into train and test sets 
X_train, X_test, y_train, y_test = train_test_split(energy_mean, art_mean, test_size = 0.5, random_state = 1)

#plot original data (the means)
plt.plot(energy_mean, art_mean, 'o', markersize=5, color='black')

#fit the train and test set 
ourModel = LinearRegression().fit(X_train, y_train)

#calculating r^2 = score
rSq = ourModel.score(energy_mean, art_mean) #0.11164551

#coefficient (b1)
slope = ourModel.coef_ #-0.59723

#intercept (b0)
intercept = ourModel.intercept_ #6.79912

#regression line : y= mx + b
yHat = slope * energy_mean + intercept

#Predictions
y_pred = ourModel.predict(X_test)

#Mean absolute error 
mae = mean_absolute_error(y_test, y_pred) #0.50602185

#mean squared error 
mse = mean_squared_error(y_test, y_pred) #0.469584

#root mean squared error 
rmse = np.sqrt(mse) #0.68526269

#residuals 
residuals = art_mean - yHat.flatten()

#plot graph 
plt.xlabel("Energy Ratings")
plt.ylabel("Art Ratings")

#plot regression line 
plt.plot(energy_mean, yHat, color = 'red', linewidth = 1)
plt.title("Using scikit-learnL R^2 = {:.3f}".format(rSq))

#%% Question #6 ridge regression
#Build a regression model to predict art preference ratings from energy ratings and demographic information

energy_ratings = data[:, 91: 182]
demographic = data[:, 215:220]
demographic_NaNCheck = print(np.isnan(demographic))
energy_ratings_NaNCheck = print(np.isnan(energy_ratings))
energy_ratings_and_demographic = np.column_stack((energy_ratings, demographic))
energy_ratings_and_demographic_mean = np.mean(energy_ratings_and_demographic, axis = 0).reshape(-1,1)
art_mean = np.mean(art_ratings, axis = 0).reshape(-1,1)


X = energy_ratings_and_demographic_mean
y = art_mean

X = X[~np.isnan(X).any(axis = 1)]
y = y[~np.isnan(y).any(axis = 1)]

from sklearn.model_selection import train_test_split, cross_val_score
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = mySeed)

from sklearn.linear_model import Ridge 
model=Ridge()
model.fit(XTrain, yTrain)

y_pred = model.predict(XTest)
plt.scatter(yTest, y_pred)
plt.plot([yTest.min(), yTest.max()], [yTest.min(), yTest.max()], 'k--', lw=4)

plt.xlabel("Energy Ratings and Demographic Info")
plt.ylabel("Predicted Art Preference Ratings")
plt.title("Ridge Regression")
plt.show()
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(yTest, y_pred)
print(mse)

#%% Question #7 
#kNN means

ratings_mean = art_ratings.mean(axis = 0).reshape(-1,1)

energy_means = energy_ratings.mean(axis = 0).reshape(-1,1)

art_types = theArt.iloc[:,5]

art_types = art_types.drop(0)

combineddata = np.column_stack((energy_means, ratings_mean, art_types))

combineddata = combineddata.astype(int) 

inertias = [] 

for i in range(1,11):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(combineddata)
    inertias.append(kmeans.inertia_)


plt.plot(range(1,11), inertias, marker = 'o')
plt.title('Elbow Method Plot')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show() 

zscoredData = stats.zscore(combineddata)

pca = PCA().fit(zscoredData)

eigVals = pca.explained_variance_ 
loadings = pca.components_*-1 
orig_data_new_coordinates = pca.fit_transform(zscoredData)*-1

x = np.column_stack((orig_data_new_coordinates[:,0], orig_data_new_coordinates[:,1]))

numClusters = 9 
sSum = np.empty([numClusters, 1])*np.NaN

for ii in range(2, numClusters +2):
    kMeans = KMeans(n_clusters = int(ii)).fit(x)
    cId =  kMeans.labels_
    cCoords = kMeans.cluster_centers_
    s = silhouette_samples(x, cId).reshape(-1,1)
    sSum[ii-2] = sum(s)
    
plt.subplot(3,3,ii-1)
plt.hist(s, bins = 20)

plt.xlim(-0.2, 1)
plt.ylim(0,25) 
plt.xlabel('Silhouette Score')
plt.ylabel('count')
plt.title("sum: {}".format(int(sSum[ii-2])))
plt.tight_layout() 

plt.plot(np.linspace(2, numClusters, 9), sSum)
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of silhouette scores")
plt.show() 

# 3 KMeans graph 
numClusters = 3 
kMeans = KMeans(n_clusters = numClusters).fit(x) 
cId = kMeans.labels_ 
cCoords = kMeans.cluster_centers_ 

for ii in range(numClusters):
    plotIndex = np.argwhere(cId == int(ii))
    plt.plot(x[plotIndex, 0], x[plotIndex, 1], 'o', markersize = 1 )
    plt.plot(cCoords[int(ii-1),0], cCoords[int(ii-1),1], 'o', markersize = 5, color = 'black' )
    plt.xlabel("preference art ratings")
    plt.ylabel("energy ratings")
    plt.title("k means")
    
    
kMeans.fit(combineddata) 
labels = kMeans.labels_
clusters = labels 
clusters[clusters == 0] = 3 
predicted_types = np.where(clusters == 0,3, clusters) 

kmeans_results = pd.concat([pd.DataFrame(combineddata), pd.DataFrame(predicted_types)], axis = 1)

r = np.corrcoef(kmeans_results, rowvar = False)

plt.imshow(r) 
plt.colorbar() 
plt.show()

#%% 8) Considering only the first principal component of the self-image ratings as inputs to a regression 
#model – how well can you predict art preference ratings from that factor alone?

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from scipy.special import expit 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import math 

self_image = data[:, 205:214]
self_image_NaNCheck = np.isnan(self_image) 
art = data[:,0:91]
art_mean = np.mean(art, axis = 1).reshape(-1,1)
self_image_art = np.column_stack((art_mean, self_image))

self_image_art_NaNCheck = np.isnan(self_image_art)
self_image_art_NaNrows =np.where(self_image_art_NaNCheck.any(axis=1))[0]
self_image_art_noNaN = np.delete(self_image_art, self_image_art_NaNrows , axis=0)


#correlation heatmap
corrMatrix = np.corrcoef(self_image_art_noNaN,rowvar=False)
plt.xlabel("Self Image")
plt.ylabel("Art")
plt.title("Heat Map")
plt.imshow(r) 
plt.colorbar()
plt.show()

zscoredData = stats.zscore(self_image_art_noNaN)

pca = PCA().fit(zscoredData)

eigVals = pca.explained_variance_

loadings = pca.components_*-1

origDataNewCoordinates = pca.fit_transform(zscoredData)*-1
varExplained = eigVals/sum(eigVals)*100

kaiserThreshold = 1
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigVals > kaiserThreshold)) #This outputs 2

kairserThresholdPassers = eigVals > kaiserThreshold
newData = pca.transform(zscoredData)[:, kairserThresholdPassers]


from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

X = newData
y = self_image_art_noNaN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=mySeed)
model = Ridge()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.scatter(y_pred, y_test)
plt.plot(y_pred, y_pred, color = 'red')
plt.xlabel('Self-Image Ratings')
plt.ylabel('Art Preference Ratings')
plt.title('Ridge Regression on Preference Ratings R^2 = 0.52664')

from sklearn.metrics import mean_squared_error

rsquared = model.score(X, y) 
print(rsquared) #0.52664
mse = mean_squared_error(y_test, y_pred)
print(mse) 
math.sqrt(mse) #0.55321
              
#%% Quesion #9
#Consider the first 3 principal components of the “dark personality” traits
#use these traits as inputs to a regression model to predict art preference ratings. 
#Which of these components significantly predict art preference ratings?
#Comment on the likely identity of these factors (e.g. narcissism, manipulativeness, callousness, etc.). 



dark_personality = np.array(data[:, 183:195])
art = np.array(data[:, 0:91])

art_mean = art.mean(axis = 1)

dark_personality_NaNCheck = np.any(np.isnan(dark_personality))
art_NaNCheck = np.any(np.isnan(art_mean))

dark_personality_art = np.column_stack((dark_personality, art_mean))
dark_personality_art_NaN = np.isnan(dark_personality_art)
dark_personality_art_NaNrows = np.where(dark_personality_art_NaN.any(axis = 1))[0]
dark_personality_art_noNaN = np.delete(dark_personality_art, dark_personality_art_NaNrows, axis = 0)

dark_personality_art_noNaN = (pd.DataFrame(dark_personality_art_noNaN)).iloc[:,0:12]
art_mean_noNANs = (pd.DataFrame(dark_personality_art_noNaN)).iloc[:,12]

#self_image_art_corr = np.corrcoef(self_image_art_noNaN, art_mean_noNANs)
#demographic_art_corr = np.corrcoef(demographic, art_mean_noNANs)

corrMatrix = np.corrcoef(dark_personality_art_noNaN, rowvar = False)
plt.imshow(corrMatrix)
plt.xlabel("Dark-personality")
plt.ylabel("Dark-personality")
plt.colorbar()
plt.show()

#PCA
zscoredData = stats.zscore(dark_personality_art_noNaN)
pca = PCA().fit(zscoredData)
eigVals = pca.explained_variance_
loadings = pca.components_*-1
rotatedData = pca.fit_transform(zscoredData)*-1
varExplained = eigVals/sum(eigVals)*100

kaiserThreshold = 1
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigVals > kaiserThreshold)) #This outputs 2
# 3 Factors selected for the critera 


kairserThresholdPassers = eigVals > kaiserThreshold
PCA1 = pca.transform(zscoredData)[:, kairserThresholdPassers]

#PCA Number 2
pca2 = PCA()
X_pca2 = pca2.fit_transform(PCA1)
n_components2 = sum(pca2.explained_variance_ > 1)
X_pca2 = X_pca2[:, :n_components2]
#PCA Number 3
pca3 = PCA()
X_pca3 = pca3.fit_transform(X_pca2)
n_components3 = sum(pca3.explained_variance_ > 1)
X_pca3 = X_pca3[:, :n_components3]


#Making the regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Load your data
X = X_pca3
y = combinedMatrixDarkPersonalityAndArt
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=mySeed)
model = Ridge()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

#Plotting the results for PCA #1
plt.scatter(y_pred, y_test)
plt.plot(y_pred, y_pred, color = 'red')
plt.xlabel('Dark Personality Traits')
plt.ylabel('Art Preference Ratings')
plt.title('Ride Regression on Preference Ratings with 3 PCAs r^2 = .583)


from sklearn.metrics import mean_squared_error

rsquared = model.score(X, y)
print(rsquared) #.497133
mse = mean_squared_error(y_test, y_pred)
print(mse) #.6438


#%% Question #10 
#Can you determine the political orientation of the users from all the other information available?
#Column 216: user ages
#1 = progressive, 2 = liberal, 3 = moderate, 4=conservative, 5=libertarian, 6 = independent)

age = np.array(data[:,215])
poli_orient = np.array(data[:,217])

age_NaN = np.any(np.isnan(age))
poli_orient_NaN = np.any(np.isnan(poli_orient))

age_political = np.column_stack((age, poli_orient))
age_political_NaN = np.isnan(age_political)
age_political_NaNrows = np.where(age_political_NaN.any(axis = 1))[0]
age_political_noNaN = np.delete(age_political, age_political_NaNrows, axis = 0)
age_noNaN = (pd.DataFrame(age_political_noNaN)).iloc[:,0:1]
political_NaN = (pd.DataFrame(age_political_noNaN)).iloc[:,1]

print("Total Number of Users: ", len(political_NaN))

left = political_NaN.isin([1, 2]).sum()
left_avg_age = np.mean(age_noNaN[political_NaN.isin([1,2])])
left_age_std = np.std(age_noNaN[political_NaN.isin([1,2])])

print("Number of users who lean left: ", left)
print("Average age of users who lean left: ", left)
print("STD of age of users who lean left: ", left)

non_left = political_NaN.isin([3,4,5,6]).sum()
non_left_avg_age = np.mean(age_noNaN[political_NaN.isin([3,4,5,6])])
non_left_age_std = np.std(age_noNaN[political_NaN.isin([3,4,5,6])])

print("Number of users who are non-left leaning: ", non_left)
print("Average age of users who are non-left leaning: ", non_left_avg_age)
print("STD of age of users who are non-left leaning:  ", non_left_age_std)

#plot
plt.scatter(age_noNaN, political_NaN, color = 'blue')
plt.xlabel("Age")
plt.xlim([16,26])
plt.ylabel("Political Orientation")
plt.yticks(np.array([0,6]))
plt.show()

#forest 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

x = (age_noNaN).values.reshape(-1,1).astype(np.float64)
y = (political_NaN).values.reshape(-1,1).astype(np.float64)
model = LogisticRegression().fit(x,y)
numTrees = 100
clf = RandomForestClassifier(n_estimators=numTrees).fit(x,y)
pred = clf.predict(x)
model_accuracy = accuracy_score(y, pred)
print("Random Forest Model Accuracy: ", model_accuracy)

clf = LogisticRegression(multi_class='ovo', random_state=mySeed)
auc = metrics.roc_auc_score(left, non_left)
print(auc)


                       
























