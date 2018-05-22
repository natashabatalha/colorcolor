import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB,MultinomialNB 
from sklearn.svm import SVC
import pickle as pk
import numpy as np
from colorcolor import compute_colors as c
from itertools import combinations as comb
import operator 
import pickle as pk
from sklearn.preprocessing import scale
import pandas as pd
import seaborn as sns


def get_best_filter(all_data, independent_variable,nfilter,filter_set):
    """
    Returns the highest accuracy filters given how many filter observations there are available
    returns one accuracy/filter combo per algorithm

    Parameters
    ----------
    all_data : pandas.DataFrame 
        This is a dataframe that contains columns: planet properties and filter observations 
        and rows: every single model in the dataset. Consider playing around with what subset of data you provide. 
        For example, a cloud free sample, a sample with a narrower range of phase angles. 
    independent_variable : str
        The independent variable will define what you want to classify by. I.e. 'metallicity','clouds','distance'
    nfilter : int
        This is the number of filters you want to assume you get during the observation. IN our paper we test 2-6. 
    filter_set : str
        We tested a variety of different filter sets defined by the reference files. Currently our options are: WFIRST,VPL,and 
        fake for the fake filter set that was tested. 

    Returns
    -------
    dict, dict
        A list of dictionaries. The first is a dictionary of the optimized filter set and the second is 
        a dictionary of floats which descries the % accuracy for each algorithm being tested
    """
    #get all filteres 
    all_filters = list(c.print_filters(filter_set))
    LDA = {}
    KNN ={}
    CART ={}
    SVM={}
    accuracies={}
    #run through nfilter options 
    for fs in comb(all_filters,nfilter): #all_filters[1:]:#

        #can only ever look at difference between two filters 
        fcomb = [ii[0]+ii[1] for ii in comb(fs,2)] #['575',fs,'575'+fs]#
        fcomb += fs #plus aboluste colors
        print(fcomb)
        X = all_data.loc[:,fcomb]
        #must make label a string
        Y =  all_data[independent_variable].astype(str)
        X = X.values
        Y = Y.values
        #get accuracies of models
        name = ''.join(fs)
        result =compare_models(X,Y)
        accuracies[name]= result

        #get maximum for each filter
        LDA[name] = result['LDA'][0]
        KNN[name] = result['KNN'][0]
        CART[name] = result['CART'][0]
        SVM[name] = result['SVM'][0]

    #get absolute maximum for each ML method
    maxkey = {}
    maxval ={}
    maxkey['LDA'] = max(LDA.items(), key=operator.itemgetter(1))[0]
    maxval['LDA'] = LDA[maxkey['LDA']]
    maxkey['KNN'] = max(KNN.items(), key=operator.itemgetter(1))[0]
    maxval['KNN'] = KNN[maxkey['KNN']]
    maxkey['CART'] = max(CART.items(), key=operator.itemgetter(1))[0]
    maxval['CART'] = CART[maxkey['CART']]
    maxkey['SVM'] = max(SVM.items(), key=operator.itemgetter(1))[0]
    maxval['SVM'] = SVM[maxkey['SVM']]
    return maxkey,maxval

def compare_models(x,y,plot=False,seed = 7):
    """
    This function uses a Kfold analysis to compare several different machine learning algorithms 
    Code adapted from https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/
    Used for classifying planets based on color-color plots

    Parameters
    ----------
    x : dict 
        This is a large dictionary of observations for every planet in the training set 
    y : str 
        This string defines the labels (i.e. metallicity or clouds or distance)
    plot : bool 
        (Optional) Default=False. This will plot out the comparison of each of the models. 
        If you are running a ton of parameter space it is recommended to turn this to False 
    seed : int 
        (Optional) Default=7 this is to ensure that the seed for each of our random states is the same

    Returns
    -------
    dict
        A dictionary with an entry for each algorithm that was tested. [mean accuracy, standard devaition of accuracy]
    """

    # prepare configuration for cross validation test harness
    # random seed must be the same for everything


    # prepare models
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))              
    models.append(('SVM', SVC()))

    # evaluate each model in turn with the same number of splits and the same random seed
    results = []
    names = []
    scoring = 'accuracy'
    out = {}
    for name, model in models:
        kfold = model_selection.KFold(n_splits=75, random_state=seed)
        cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        out[name] = [cv_results.mean(), cv_results.std()]
        print([cv_results.mean(), cv_results.std()])
    # boxplot algorithm comparison
    if plot==True:
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.show()

    return out

#SECTION 4.1 -- Classifying by metallicity with several different number of filters
#data = pk.load(open('/Volumes/MyPassportforMac/WFIRST/FluxDataFrameWFIRST.pk','rb'))
def optimize(data,classify_by,filter_set,output_file='results.pk'):
    """Reproduces results in section 4.1
    This is the code to produce the results from section 4.1 on finding the best filters for 
    classifying metallicities, distance, cloud parameter, etc. 

    Specifically, it runs through all combinations of 2,3,4,5 and 6 filter combinations using all 
    four algorithms. For each combination it returns:
    - the best filter combination for 2, 3, 4, 5, and 5 filter combinations
    - the classification accuracy for that combination for each ML algorithm tested

    Parameters: 
    -----------
    output_file : str 
        (Optional) This is a pickle where the output gets dumped to. If you do not wnat to save results: 'output_file=None'
    data : pandas.DataFrame 
        This is a dataframe that contains columns: planet properties and filter observations 
        and rows: every single model in the dataset
    classify_by : str
        In section 4.1 we explore classifying by 'metallicity', 'clouds' and 'distance'
    filter_set : str
        In section 4.1 we explore  several different filter sets. Current files exist for 'wfirst','vpl' and 'fake'

    Returns
    -------
    dict 
        We tested 2,3,4,5 and 6 filter combinations. The top level key of the dictionary is '2','3','4' etc. 
        Then within each one of those keys is a list [dict_filters, dict_accuracy]. dict_filters contains one key per 
        algorithm you tested, along with the optimized filter for that algorithm. dict_accuracy contains the % accuracy for 
        each of those algorithms

    Example
    -------
    >>> data = pk.load(open('/Volumes/MyPassportforMac/WFIRST/FluxDataFrameWFIRST.pk','rb'))
    >>> #select subset of dataset for testing
    >>> subset = data.loc[data['cloud']==0] #cloud free
    >>> optimize('test.html', subset, 'metallicity','wfirst')

    """
    et= data
    everything=et.dropna()[~et.dropna().isin([np.inf, -np.inf])]
    #define subset of everything you want to test 
    everything = everything.dropna()

    #run through parameter space for 2-6 filter options
    ALL = {}
    for i in [2,3,4,5,6]:
        a = get_best_filter(everything,classify_by,i,filter_set)
        ALL[str(i)] = a 

    if output_file!=None: pk.dump(ALL,open(output_file,'wb'))

    return ALL

#################### DISCUSSION-- testing VPL filters

"""
import pickle as pk

et= pk.load(open('/Volumes/MyPassportforMac/WFIRST/FluxDataFrameVPL.pk','rb'))
everything=et.dropna()[~et.dropna().isin([np.inf, -np.inf])]
#define subset of everything you want to test 
everything = everything.dropna()

ALL = {}
for i in [3,4]:
    print (i)
    a = get_best_filter(everything,'metallicity',i)
    print(a)
    ALL[str(i)] = a 
pk.dump(ALL,open('/Volumes/MyPassportforMac/WFIRST/VPLfilters_cloudfreesample.pk','wb'))
#({'LDA': '354556698799', 'CART': '354543537789'}, {'LDA': 0.14050690567146637, 'CART': 0.49909638554216879})
"""


#################### DISCUSSION-- Optimizing our own filters 
"""
import pickle as pk

et= pk.load(open('/Users/batalha/Documents/WFIRST/FluxDataFrameFAKE_5pct_575.pk','rb'))
everything=et.dropna()[~et.dropna().isin([np.inf, -np.inf])]
#define subset of everything you want to test 
everything = everything.loc[(everything['distance']>=3) & (everything['cloud']>=1)].dropna()# .loc[(everything['cloud']==0)] & (everything['phase']==90)

ALL = {}
for i in [2]:
    print (i)
    a = get_best_filter(everything,'metallicity',i,'fake')
    print(a)
    ALL[str(i)] = a 
#pk.dump(ALL,open('/Users/batalha/Documents/WFIRST/FAKEfilters_bymetal_highFsed.pk','wb'))

ALL = {}
for i in [2]:
    print (i)
    a = get_best_filter(everything,'cloud',i,'fake')
    print(a)
    ALL[str(i)] = a 
#pk.dump(ALL,open('/Users/batalha/Documents/WFIRST/FAKEfilters_bycloud_highFsed.pk','wb'))
"""


def lda_analysis(X,y):
    """
    This contains functionality to do a linear discriminant analysis as the 
    
    Parameters
    ----------

    x : pandas.DataFrame
        Dataframe with all filter observations 
    y : pandas.DataFrame 
        Dataframe with the parameter you wish to classify by

    Returns
    -------
    dict 
        Return dictionary with components of LDA analysis

    """
    #SCALE x incase you have absolute filters and units are different
    standardised_X = scale(X)
    standardised_X = pd.DataFrame(standardised_X, index=X.index, columns=X.columns)

    lda = LinearDiscriminantAnalysis().fit(X,y)

    lda_values = rpredict(lda, standardised_X.reset_index(drop=True), y.reset_index(drop=True), True)

    ldahist(lda_values["x"].LD1, y.reset_index(drop=True), 1)
    ldahist(lda_values["x"].LD2, y.reset_index(drop=True), 2)

    return lda_values



#ALL OF THESE FUNCTIONS ARE MODIFIED TAKEN FROM 
#http://python-for-multivariate-analysis.readthedocs.io/a_little_book_of_python_for_multivariate_analysis.html
def ldahist(data, g ,lda_num,sep=False):
    if lda_num==1:
        xlabel = 'First Linear Discriminant'
    elif lda_num==2:
        xlabel = 'Second Linear Discriminant'
    xmin = np.trunc(np.min(data)) - 1
    xmax = np.trunc(np.max(data)) + 1
    ncol = len(set(g))
    binwidth = 0.5
    bins=np.arange(xmin, xmax + binwidth, binwidth)
    if sep:
        fig, axl = plt.subplots(ncol, 1, sharey=True, sharex=True)
    else:
        fig, axl = plt.subplots(1, 1, sharey=True, sharex=True)
        axl = [axl]*ncol
    for ax, (group, gdata) in zip(axl, data.groupby(g)):
        sns.distplot(gdata.values, bins, ax=ax, label="group "+str(group))
        ax.set_xlim([xmin, xmax])
        if sep:
            ax.set_xlabel("group"+str(group))
        else:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_xlabel(xlabel)
    plt.tight_layout()


def rpredict(lda, X, y, out=False):
    ret = {"class": lda.predict(X),
           "posterior": pd.DataFrame(lda.predict_proba(X), columns=lda.classes_)}
    ret["x"] = pd.DataFrame(lda.fit_transform(X, y))
    ret["x"].columns = ["LD"+str(i+1) for i in range(ret["x"].shape[1])]
    return ret

def calcWithinGroupsVariance(variable, groupvariable):
    # find out how many values the group variable can take
    levels = sorted(set(groupvariable))
    numlevels = len(levels)
    # get the mean and standard deviation for each group:
    numtotal = 0
    denomtotal = 0
    for leveli in levels:
        levelidata = variable[groupvariable==leveli]
        levelilength = len(levelidata)
        # get the standard deviation for group i:
        sdi = np.std(levelidata)
        numi = (levelilength)*sdi**2
        denomi = levelilength
        numtotal = numtotal + numi
        denomtotal = denomtotal + denomi
    # calculate the within-groups variance
    Vw = numtotal / (denomtotal - numlevels)
    return Vw
def calcBetweenGroupsVariance(variable, groupvariable):
    # find out how many values the group variable can take
    levels = sorted(set((groupvariable)))
    numlevels = len(levels)
    # calculate the overall grand mean:
    grandmean = np.mean(variable)
    # get the mean and standard deviation for each group:
    numtotal = 0
    denomtotal = 0
    for leveli in levels:
        levelidata = variable[groupvariable==leveli]
        levelilength = len(levelidata)
        # get the mean and standard deviation for group i:
        meani = np.mean(levelidata)
        sdi = np.std(levelidata)
        numi = levelilength * ((meani - grandmean)**2)
        denomi = levelilength
        numtotal = numtotal + numi
        denomtotal = denomtotal + denomi
    # calculate the between-groups variance
    Vb = numtotal / (numlevels - 1)
    return(Vb)
def calclda(variables, loadings):
    # find the number of samples in the data set and the number of variables
    numsamples, numvariables = variables.shape
    # make a vector to store the discriminant function
    ld = np.zeros(numsamples)
    # calculate the value of the discriminant function for each sample
    for i in range(numsamples):
        valuei = 0
        for j in range(numvariables):
            valueij = variables.iloc[i, j]
            loadingj = loadings[j]
            valuei = valuei + (valueij * loadingj)
        ld[i] = valuei
    # standardise the discriminant function so that its mean value is 0:
    ld = scale(ld, with_std=False)
    return ld
def groupStandardise(variables, groupvariable):
    # find the number of samples in the data set and the number of variables
    numsamples, numvariables = variables.shape
    # find the variable names
    variablenames = variables.columns
    # calculate the group-standardised version of each variable
    variables_new = pd.DataFrame()
    for i in range(numvariables):
        variable_name = variablenames[i]
        variablei = variables[variable_name]
        variablei_Vw = calcWithinGroupsVariance(variablei, groupvariable)
        variablei_mean = np.mean(variablei)
        variablei_new = (variablei - variablei_mean)/(np.sqrt(variablei_Vw))
        variables_new[variable_name] = variablei_new
    return variables_new
def calcSeparations(variables, groupvariable):
    # calculate the separation for each variable
    for variablename in variables:
        variablei = variables[variablename]
        Vw = calcWithinGroupsVariance(variablei, groupvariable)
        Vb = calcBetweenGroupsVariance(variablei, groupvariable)
        sep = Vb/Vw
        print("variable", variablename, "Vw=", Vw, "Vb=", Vb, "separation=", sep)

