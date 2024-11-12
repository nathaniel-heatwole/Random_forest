# RANDOM_FOREST.PY
# Nathaniel Heatwole, PhD (heatwolen@gmail.com)
# Uses a random forest to predict passenger survival in the Titanic disaster
# Training data: 891 Titanic passengers ('Titanic dataset') (from https://www.kaggle.com/competitions/titanic/data)

import time
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore, Style
from sklearn.ensemble import RandomForestClassifier
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.feature_selection import SelectKBest, chi2

time0 = time.time()
ver = ''  # version (empty or integer)

topic = 'Random forest'
topic_underscore = topic.replace(' ','_')

#--------------#
#  PARAMETERS  #
#--------------#

# random forest
n_estimators = 100      # total trees in random forest (default = 100)
criterion = 'gini'      # split quality measure ('gini', 'entropy', 'log_loss') (default = 'gini')
max_depth = None        # maximum depth of tree (default = None)
max_features = 'sqrt'   # number of features to consider when looking for best split ('sqrt', 'log2', 'None') (default = 'sqrt')
bootstrap = True        # whether bootstrap samples are used when building trees (if False, the entire dataset is used to build each tree) (default = True)
oob_score = False       # whether to use out-of-bag samples to estimate the generalization score (only available if bootstrap = True) (default = False)
max_samples = None      # if bootstrap is True, number of samples to draw from X to train each base estimator (if None, samples = obs) (default = None)
random_state = 0        # controls randomness of samples bootstrapping and feature sampling (default = None)

# variables
y_var = 'survived'
embark_vars = ['embarked c', 'embarked q', 'embarked s', 'embarked missing']
economic_vars = ['fare', 'fare zero', 'pclass 1', 'pclass 2', 'pclass 3']
other_vars = ['age recoded', 'age missing', 'age estimated', 'male', 'sibsp', 'parch']
covars = embark_vars + economic_vars + other_vars

#-----------------#
#  DATA CLEANING  #
#-----------------#

total_covars = len(covars)

# import data
titanic = pd.read_csv('Titanic.csv')

# column names -> lowercase
for col in titanic.columns:
    titanic[col.lower()] = titanic[col]
    titanic.drop(columns=col, axis=1, inplace=True)
del col

# remove unneeded columns
titanic.drop(columns=['passengerid'], inplace=True)  # unneeded because this variable duplicates the index
titanic.drop(columns=['name', 'ticket', 'cabin'], inplace=True)  # these variables are too varied/granular to be useful for modeling purposes

# port of embarkation
for embarked in ['C', 'Q', 'S']:
    titanic['embarked ' + embarked.lower()] = titanic['embarked'].apply(lambda e: 1 if e == embarked else 0)
titanic['embarked missing'] = titanic['embarked'].apply(lambda e: 1 if pd.isna(e) else 0)  # because even missing values convey some information

# socioeconomic class
for passenger in range(titanic['pclass'].min(), titanic['pclass'].max() + 1):
    titanic['pclass ' + str(passenger)] = titanic['pclass'].apply(lambda p: 1 if p == passenger else 0)

# age related
titanic['age recoded'] = titanic['age'].apply(lambda a: 0 if np.isnan(a) else a)  # recodes missing age values to zero
titanic['age missing'] = titanic['age'].apply(lambda a: 1 if np.isnan(a) else 0)  # because even missing values convey some information
titanic['age estimated'] = titanic['age'].apply(lambda a: 1 if a > 1 and (a - math.floor(a) == 0.5) else 0)  # ages ending in one half are estimated

# other
titanic['male'] = titanic['sex'].apply(lambda s: 1 if s == 'male' else 0)
titanic['fare zero'] = titanic['fare'].apply(lambda f: 1 if f == 0 else 0)

#-------------#
#  ALGORITHM  #
#-------------#

fmt = '.2f'

# model data
y = titanic[y_var]
x = titanic[covars]

# define forest
eqn = RandomForestClassifier(
                             n_estimators = n_estimators,
                             criterion = criterion,
                             max_depth = max_depth,
                             max_features = max_features,
                             bootstrap = bootstrap,
                             max_samples = max_samples,
                             oob_score = oob_score,
                             random_state = random_state
                            )

# fit forest
fit = eqn.fit(x, y)
accuracy = round(fit.score(x, y), 4)

# generate predictions
titanic['pred_class'] = fit.predict(x)
titanic[['prob_died', 'prob_survived']] = fit.predict_proba(x)

# summary stats
stats = pd.DataFrame()
df = titanic[['survived', 'prob_survived']]
stats['mean'] = df.groupby(['survived']).mean()
stats['p25'] = df.groupby(['survived']).quantile(0.25)
stats['p75'] = df.groupby(['survived']).quantile(0.75)
stats = round(stats, 2)
stats['iqr'] = [str(format(stats['p25'][i], fmt)) + ' - ' + str(format(stats['p75'][i], fmt)) for i in stats.index]
del df

#----------------------#
#  FEATURE IMPORTANCE  #
#----------------------#

# correlation matrix
corr_matrix = titanic[itertools.chain([y_var], covars)].corr(method='pearson')

# chi-squared
select_k_best = SelectKBest(score_func=chi2, k=total_covars)
select_k_best.fit_transform(x, y)
chisq_best_features = x.columns[select_k_best.get_support()]

# coefficient of variation
coef_variation = pd.DataFrame()
coef_variation['stdev'] = titanic[covars].std()
coef_variation['mean'] = titanic[covars].mean()
coef_variation['cov'] = coef_variation['stdev'] / coef_variation['mean']
coef_variation.sort_values(by='cov', ascending=False, inplace=True)

# sklearn
feature_imp = pd.DataFrame(fit.feature_importances_, index=x.columns)
feature_imp = round(feature_imp, 3)
feature_imp.rename(columns={0:'value'}, inplace=True)
feature_imp.sort_values(by='value', ascending=False, inplace=True)

#--------#
#  PLOT  #
#--------#

# parameters
title_size = 11
axis_labels_size = 8
axis_ticks_size = 8
point_size = 8
text_size = 8

# spatial buffers (zero = no buffer)
x_buffer = 0.02
y_buffer = 0.075

# passenger counts
total_survivors = len(titanic.loc[titanic['survived'] == 1])
total_nonsurvivors = len(titanic.loc[titanic['survived'] == 0])

# labels
survived1_label = 'Survivors (n = ' + str(total_survivors) + ')' + '\n' + 'mean: ' + str(format(stats['mean'][1], fmt)) + '\n' + 'IQR: ' + str(stats['iqr'][1])
survived0_label = 'Non-survivors (n = ' + str(total_nonsurvivors) + ')' + '\n' + 'mean: ' + str(format(stats['mean'][0], fmt)) + '\n' + 'IQR: ' + str(stats['iqr'][0])

# generate plot
fig1 = plt.figure()
plt.title(topic + ' classifier - Titanic survival', fontsize=title_size, fontweight='bold')
plt.scatter(titanic['prob_survived'], titanic['survived'], s=point_size)
plt.text(x_buffer, y_buffer, survived0_label, va='center', ha='left', fontsize=text_size)
plt.text(1 - x_buffer, 1 - y_buffer, survived1_label, va='center', ha='right', fontsize=text_size)
plt.xlabel('Survival probability (predicted)', fontsize=axis_labels_size)
plt.ylabel('Survival (actual)', fontsize=axis_labels_size)
plt.xticks(fontsize=axis_ticks_size)
plt.yticks(fontsize=axis_ticks_size)
plt.grid(True, alpha=0.5, zorder=0)
plt.show(True)

#----------#
#  EXPORT  #
#----------#

# functions
def console_print(title, df):
    print(Fore.GREEN + '\033[1m' + '\n' + title + Style.RESET_ALL)
    print(df)
def txt_export(title, df, f):
    print(title, file=f)
    print(df, file=f)

# export summary (console, txt)
with open(topic_underscore + '_summary' + ver + '.txt', 'w') as f:
    df = feature_imp
    title = topic.upper() + ' - FEATURE IMPORTANCE'
    console_print(title, df)
    txt_export(title, df, f)
del title, f, df

# export plot (pdf)
pdf = PdfPages(topic_underscore + '_plots' + ver + '.pdf')
pdf.savefig(fig1)
pdf.close()
del pdf

###

# runtime
runtime_sec = round(time.time() - time0, 2)
if runtime_sec < 60:
    print('\n' + 'runtime: ' + str(runtime_sec) + ' sec')
else:
    runtime_min_sec = str(int(np.floor(runtime_sec / 60))) + ' min ' + str(round(runtime_sec % 60, 2)) + ' sec'
    print('\n' + 'runtime: ' + str(runtime_sec) + ' sec (' + runtime_min_sec + ')')
del time0


