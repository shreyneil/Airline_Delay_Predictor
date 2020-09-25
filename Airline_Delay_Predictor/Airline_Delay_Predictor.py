import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sklearn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cross_validation  import train_test_split, cross_val_score, ShuffleSplit
import scipy
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn import linear_model, cross_validation, metrics, svm, ensemble
import seaborn as sns

df = pd.read_csv("2007.csv")

#data exploration
df1 = df[df['Origin']=='ORD'].dropna(subset=['DepDelay'])
df1['DepDelayed'] = df1['DepDelay'].apply(lambda x:int(x>=15))
print "total flights: " + str(df1.shape[0])
print "total delays: " + str(df1['DepDelayed'].sum())

grouped = df1[['DepDelayed', 'Month']].groupby('Month').mean()

grouped.plot(kind='bar')

df1['hour'] = df1['CRSDepTime'].map(lambda x: int(str(int(x)).zfill(4)[:2]))

grouped = df1[['DepDelayed', 'hour']].groupby('hour').mean()

grouped.plot(kind='bar')

grouped1 = df1[['DepDelayed', 'UniqueCarrier']].groupby('UniqueCarrier').filter(lambda x: len(x)>10)
grouped2 = grouped1.groupby('UniqueCarrier').mean()
carrier = grouped2.sort_values(['DepDelayed'], ascending=False)

carrier[:15].plot(kind='bar')

grouped = df1[['DepDelayed', 'DayofMonth']].groupby('DayofMonth').mean()

grouped.plot(kind='bar')

grouped = df1[['DepDelayed', 'DayOfWeek']].groupby('DayOfWeek').mean()

grouped.plot(kind='bar')


grouped1 = df1[['DepDelayed', 'Dest']].groupby('Dest').filter(lambda x: len(x)>10)
grouped2 = grouped1.groupby('Dest').mean()
grouped2[:15].plot(kind='bar')


grouped1 = df1[['DepDelayed', 'Distance']].groupby('Distance').filter(lambda x: len(x)>600)
grouped2 = grouped1.groupby('Distance').mean()
grouped2[70:].plot(kind='bar')

del df1['TailNum']

df1['Origin'] = (df1['Origin']!='n').astype(int)

names = pd.unique(df1[['UniqueCarrier']].values.ravel())
names = pd.Series(np.arange(len(names)),names)

names1 = pd.unique(df1[['Dest']].values.ravel())
names1 = pd.Series(np.arange(len(names1)),names1)

df1['UniqueCarrier'] = df1['UniqueCarrier'].map({'XE': 0,'YV': 1,'OH': 2,'OO': 3,'UA': 4,
   'US': 5,'DL': 6,'MQ': 7,'NW': 8,'AA': 9,'AS': 10,'B6': 11,'CO': 12,'EV': 13})

df1['Dest'] = df1['Dest'].map({'EWR': 0,'IAH': 60,'CLE': 37,'ATL': 1,'BDL': 2,'BHM': 3,'BMI': 4,'BNA': 5,   
'BUF': 6,'CAE': 7,'CAK': 8,'CHS': 9, 'CID': 10, 'CLT': 11, 'CWA': 12, 'DSM': 13, 'FWA': 14, 'GRB': 15,   
'GSO': 16, 'GSP': 17, 'JAX': 18, 'LAN': 19, 'MBS': 20,'MDT': 21,'MSN': 22,'OKC': 23,'ORF': 24,   
'PIT': 25, 'RDU': 26,'ROA': 27,'ROC': 28,'SAV': 29,'SBN': 30,'ATW': 31,'FSD': 32,'GRR': 33,
'MLI': 34,'ABE': 35,'AUS': 36,'DAB': 38,'MCI': 39,'AVP': 40,'MHT': 41,'CVG': 42,'JFK': 43,
'ASE': 44,'MEM': 45,'COS': 46,'BTV': 47,'AZO': 48,'ICT': 49,'MKE': 50,'TVC': 51,'TUL': 52,
'SGF': 53,'TYS': 54,'LEX': 55,'OMA': 56,'DFW': 57,'FAR': 58,'CRW': 59,'PIA': 61,'SYR': 62,
'LNK': 63,'DAY': 64,'SPI': 65,'XNA': 66,'SDF': 67,'SLC': 68,'RAP': 69,'BZN': 70,'DTW': 71,
'IND': 72,'STL': 73,'HNL': 74,'OGG': 75,'SFO': 76,'PVD': 77,'LAX': 78,'IAD': 79,'SEA': 80,
'MSP': 81,'PHL': 82,'SAN': 83,'DEN': 84,'BOI': 85,'GEG': 86,'SNA': 87,'SMF': 88,'SJC': 89,
'PDX': 90,'RIC': 91,'ALB': 92,'BWI': 93,'SAT': 94,'OAK': 95,'CMH': 96,'BOS': 97,'DCA': 98,
'MCO': 99,'LGA': 100,'FLL': 101,'PHX': 102,'STT': 103,'LAS': 104,'MIA': 105,'TPA': 106,'PBI': 107,
'SJU': 108,'CMI': 109,'LIT': 110,'EVV': 111,'RST': 112,'MQT': 113,'CHA': 114,'SWF': 115,
'LSE': 116,'FNT': 117,'DBQ': 118,'JAN': 119,'BTR': 120,'HSV': 121,'TOL': 122,'HPN': 123,   
'PNS': 124, 'PSP': 125,'RNO': 126,'MSY': 127,'ABQ': 128, 'ELP': 129, 'TUS': 130,'RSW': 131,    
'JAC': 132,'EGE': 133, 'HDN': 134,'MTJ': 135,'ANC': 136,'LGB': 137,'ONT': 138,'PWM': 139
,'MOB': 140,'SHV': 141,'MSO': 142,'BIL': 143,'FCA': 144,'ROW': 145,'BGM': 146,'CPR': 147,
'PIH': 148,'VPS': 149,'RFD': 150,'GJT': 151,'GPT': 152})

dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843),
                (0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843)]




#building models
Ycol = 'DepDelay'
delay_threshold = 15

Xcols = ['Month','DayOfWeek','DayofMonth','Dest','UniqueCarrier','Distance']

scaler = StandardScaler()
X_values = df[df['Origin']=='ORD'][Xcols]
Y_values = df[df['Origin']=='ORD'][Ycol]

X_values['UniqueCarrier'] = pd.factorize(X_values['UniqueCarrier'])[0]
X_values['Dest'] = pd.factorize(X_values['Dest'])[0]

rows = np.random.choice(X_values.index.values, 20000) 
sampled_X = X_values.ix[rows]
sampled_Y = Y_values.ix[rows]


TrainX, TestX, TrainY, TestY = train_test_split(sampled_X, sampled_Y, 
test_size=0.50, random_state=0)

TrainX_scl = scaler.fit_transform(TrainX)
TestX_scl = scaler.transform(TestX)

print TestX[pd.isnull(TestX).any(axis=1)].T
print TrainX[pd.isnull(TrainX).any(axis=1)].T
             
"""
Minimize chartjunk by stripping out unnecessary plot borders and axis ticks
    
The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
"""

def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    
    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    
    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()
        
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)



#Logistic regresion

def show_confusion_matrix(cm):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title('Confusion matrix')
    plt.set_cmap('Blues')
    plt.colorbar()

    target_names = ['Not Delayed', 'Delayed']

    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=60)
    plt.yticks(tick_marks, target_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Convenience function to adjust plot parameters for a clear layout.
    plt.show()
    
    
clf_lr = sklearn.linear_model.LogisticRegression(penalty='l2', class_weight='balanced')
logistic_fit=clf_lr.fit(TrainX, np.where(TrainY >= delay_threshold,1,0))
pred = clf_lr.predict(TestX)

cm_lr = confusion_matrix(np.where(TestY >= delay_threshold,1,0), pred)
print("Confusion matrix")
print(pd.DataFrame(cm_lr))
report_lr = precision_recall_fscore_support(list(np.where(TestY >= delay_threshold,1,0)), list(pred), average='micro')
print "\nprecision = %0.2f, recall = %0.2f, F1 = %0.2f, accuracy = %0.2f\n" % \
        (report_lr[0], report_lr[1], report_lr[2]/2, accuracy_score(list(np.where(TestY >= delay_threshold,1,0)), list(pred)))
print(pd.DataFrame(cm_lr.astype(np.float64) / cm_lr.sum(axis=1)))
    
show_confusion_matrix(cm_lr)

#random forest classifier

# Create Random Forest classifier with 50 trees
clf_rf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
clf_rf.fit(TrainX, np.where(TrainY >= delay_threshold,1,0))

# Evaluate on test set
pred = clf_rf.predict(TestX)

# print results
cm_rf = confusion_matrix(np.where(TestY >= delay_threshold,1,0), pred)
print("Confusion matrix")
print(pd.DataFrame(cm_rf))
report_rf = precision_recall_fscore_support(list(np.where(TestY >= delay_threshold,1,0)), list(pred), average='micro')
print "\nprecision = %0.2f, recall = %0.2f, F1 = %0.2f, accuracy = %0.2f\n" % \
        (report_rf[0], report_rf[1], report_rf[2]/2, accuracy_score(list(np.where(TestY >= delay_threshold,1,0)), list(pred)))
print(pd.DataFrame(cm_rf.astype(np.float64) / cm_rf.sum(axis=1)))
    
show_confusion_matrix(cm_rf)

#feature importance
importances = pd.Series(clf_rf.feature_importances_, index=Xcols)
importances.sort_values()

plt.barh(np.arange(len(importances)), importances, alpha=0.7)
plt.yticks(np.arange(.5,len(importances),1), importances.index)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature importance')
remove_border()
plt.show()
 
#cross validating to check the maximum accuracy we can get
clf_rf = RandomForestClassifier(n_estimators=45, n_jobs=-1)
RF_scores = cross_validation.cross_val_score(clf_rf, TrainX, np.where(TrainY >= delay_threshold,1,0), cv=10, scoring='accuracy')
RF_scores

RF_scores.min(), RF_scores.mean(), RF_scores.max()

#cross validation using SVM
from sklearn.svm import SVC
svc =  SVC(kernel='rbf', C=100, gamma=0.001).fit(TrainX_scl, np.where(TrainY >= delay_threshold,1,0))
SVC_scores = cross_val_score(svc, TrainX_scl, np.where(TrainY >= delay_threshold,1,0), cv=10, scoring='accuracy', n_jobs=-1) 
SVC_scores

SVC_scores.min(), SVC_scores.mean(), SVC_scores.max()
'''
The SVM requires a much longer time to compute. Given the small performance gain 
we try to tune the SVM parameters, concentrating on the gamma parameter. We use 
a validation curve plotter on the results from the SciKit ShuffleSplit routine 
using 10 values of gamma ranged from 10e-5 to 10e5.
'''
from sklearn.learning_curve import validation_curve
n_samples, n_features = TrainX_scl.shape
cv = ShuffleSplit(n_samples, n_iter=10, train_size=500, test_size=500, random_state=0)
n_Cs = 10
Cs = np.logspace(-5, 5, n_Cs)

SVC_train_scores, SVC_test_scores = validation_curve(
    SVC(gamma=0.0001), TrainX_scl,np.where(TrainY >= delay_threshold,1,0), 'C', Cs, cv=cv)

def plot_validation_curves(param_values, train_scores, test_scores):
    for i in range(train_scores.shape[1]):
        plt.semilogx(param_values, train_scores[:, i], alpha=0.4, lw=2, c='b')
        plt.semilogx(param_values, test_scores[:, i], alpha=0.4, lw=2, c='g')

plot_validation_curves(Cs, SVC_train_scores, SVC_test_scores)
plt.ylabel("score for SVC(C=C, gamma=0.001)")
plt.xlabel("C")
#plt.text(1e-3, 0.5, "Underfitting", fontsize=16, ha='center', va='bottom')
#plt.text(1e3, 0.5, "Few Overfitting", fontsize=16, ha='center', va='bottom')
plt.title('Validation curves for the C parameter');
         
'''
Based on the results and the validation curves, we don't find that varying the gamma 
parameter will improve the model performance much. Given the results, we find that 
look at improving the RF classifier model is the best path to follow, next by encoding 
the qualitative variables into dummy variables that we had factorized previously into 
category numbers. SciKit Learn has a OneHotEncoder routine that we can use for this. 
The advantage is that we can quickly transform the test variables based on the same 
encoding. By encoding the variables as dummies, the number of variables increases 
substantially.
'''

# Use SciKitLearn OneHotEncoder to see if the model is improved
from sklearn.preprocessing import OneHotEncoder
categ = [Xcols.index(x) for x in 'Month', 'DayofMonth', 'Distance', 'Dest']
encoder = OneHotEncoder(categorical_features = categ)
TrainXenc = encoder.fit_transform(TrainX).toarray()
TestXenc = encoder.transform(TestX).toarray()

# Create Random Forest classifier with 50 trees
clf_rf = RandomForestClassifier(n_estimators=50, n_jobs=-1)
clf_rf.fit(TrainXenc, np.where(TrainY >= delay_threshold,1,0))

# Evaluate on test set
pred = clf_rf.predict(TestXenc)

# print results
cm_rf = confusion_matrix(np.where(TestY >= delay_threshold,1,0), pred)
print("Confusion matrix")
print(pd.DataFrame(cm_rf))
report_rf = precision_recall_fscore_support(list(np.where(TestY >= delay_threshold,1,0)), list(pred), average='micro')
print "\nprecision = %0.2f, recall = %0.2f, F1 = %0.2f, accuracy = %0.2f\n" % \
        (report_rf[0], report_rf[1], report_rf[2]/2, accuracy_score(list(np.where(TestY >= delay_threshold,1,0)), list(pred)))
print(pd.DataFrame(cm_rf.astype(np.float64) / cm_rf.sum(axis=1)))
    
show_confusion_matrix(cm_rf)

#Random forest exploration

Ntrees = 50
Trees = np.arange(Ntrees)+1
m = np.sqrt(TrainX.shape[1]).astype(int)
cv = 10

clf_scores = np.zeros((Ntrees,cv))

for tree in Trees:
    cols = (tree - 1)
    clf = ensemble.RandomForestClassifier(n_estimators=tree, max_features=m, random_state=0,n_jobs=-1)
    clf_scores[cols,:] = cross_validation.cross_val_score(clf, TrainX, np.where(TrainY >= delay_threshold,1,0), cv=cv, scoring = 'accuracy' , n_jobs = -1)


plt.subplots(figsize=(10,8))
score_means = np.mean(clf_scores, axis=1)
score_std = np.std(clf_scores, axis=1)
score_medians = np.median(clf_scores, axis=1)
plt.scatter(Trees,score_means, c='k', zorder=3, label= 'Mean of accuracy scores')
plt.errorbar(Trees, score_means, yerr = 2*score_std,color='#31a354', alpha =0.7, capsize=20, elinewidth=4, linestyle="None", zorder = 1, label= 'SE of accuracy scores')
plt.title('Accuracy by choice of number of trees')
plt.legend(frameon=False, loc='lower right')
plt.ylabel('Accuracy Scores')
plt.xlabel('Number of trees in the Random Forest')
plt.xticks(rotation=90)
remove_border()
plt.show



plt.subplots(figsize=(10,8))
plt.hlines(np.max(score_means),0, 51, linestyle='--', color='red', linewidth=2, alpha=0.7,  zorder = 2, label= 'Maximum of means')
plt.hlines(np.max(score_medians),0, 51, linestyle='--',color='blue', linewidth=2, alpha=0.7,  zorder = 2, label= 'Maximum of medians')
plt.scatter((np.argmax(score_means)+1),np.max(score_means), s=50, c='red', marker='o', zorder=3)
plt.scatter((np.argmax(score_medians)+1),np.max(score_medians), s=50, c='blue', marker='o', zorder=3)
plt.plot(Trees,score_means, zorder=3, c= 'k', label= 'Mean of accuracy scores')
plt.errorbar(Trees, score_means, yerr = 2*score_std,color='#31a354', alpha =0.7, capsize=20, elinewidth=4, linestyle="None", zorder = 1, label= 'SE of accuracy scores')
plt.annotate((np.argmax(score_medians)+1), 
    xy = ((np.argmax(score_medians)+1), np.max(score_medians)), 
    xytext = (5, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'blue', alpha = 0.5))
plt.annotate((np.argmax(score_means)+1), 
    xy = ((np.argmax(score_means)+1), np.max(score_means)), 
    xytext = (5, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'red', alpha = 0.5))
plt.title('Accuracy by choice of the number of trees')
plt.legend(frameon=False, loc='lower right')
plt.ylabel('Mean accuracy scores')
plt.xlabel('Number of trees in the Random Forest')
plt.xlim(0, 41)
plt.xticks(Trees,rotation=90)
remove_border()
plt.show

Ntrees2 = 50
Trees2 = np.arange(Ntrees2)+1
clf_OOBscores = np.zeros((Ntrees2))

for tree in Trees2:
    cols = (tree - 1)
    clf = ensemble.RandomForestClassifier(n_estimators=tree, oob_score=True, max_features=m, random_state=0, n_jobs=-1)
    clf.fit(TrainX, np.where(TrainY >= delay_threshold,1,0))
    clf_OOBscores[cols] = clf.oob_score_
                 
plt.subplots(figsize=(10,8))
plt.hlines(np.max(clf_OOBscores),0, 51, linestyle='--',color='blue', linewidth=2, alpha=0.7,  zorder = 2, label= 'Maximum of OOB scores')
plt.scatter((np.argmax(clf_OOBscores)+1),np.max(clf_OOBscores), s=50, c='blue', marker='o', zorder=3)
plt.plot(Trees2,clf_OOBscores, zorder=3, c= 'k', label= 'OOB scores')
plt.annotate((np.argmax(clf_OOBscores)+1), 
    xy = ((np.argmax(clf_OOBscores)+1), np.max(clf_OOBscores)), 
    xytext = (5, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'blue', alpha = 0.5))
plt.title('OOB score by choice of the number of trees')
plt.legend(frameon=False, loc='lower right')
plt.ylabel('OOB scores')
plt.xlabel('Number of trees in the Random Forest')
plt.xlim(0, 51)
plt.xticks(rotation=90)

remove_border()
plt.show
