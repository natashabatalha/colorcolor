"""
Script to compute all color colors for all filter combinations included in Filter database 
"""
import pandas as pd 
import numpy as np 
import json 
import os
from sqlalchemy import *
import pysynphot as psyn 
from itertools import combinations as comb
from colorcolor import compute_colors as  c
import pickle as pk
import multiprocessing
engine = create_engine('sqlite:///' + os.path.join(os.getenv('ALBEDO_DB'),'AlbedoModels_2015.db'))

header = pd.read_sql_table('header',engine)

star = {'temp':5800, 'metal':0.0, 'logg':4.0}

#Fake SET
filters = list(c.print_filters('wfirst'))
ccdf = pd.DataFrame({'modelid':[], 'cloud':[], 'metallicity':[], 'distance':[], 'phase':[]}) 
fdf = pd.DataFrame({i:[] for i in filters})
ccdf = ccdf.append(fdf,ignore_index=True)
"""
def runPS(headerindex):
	#for i in header.index:

	global ccdf 

	planet_dict = header.loc[headerindex]
	planet = c.select_model(planet_dict)

	print(planet_dict['index'])
	#print(planet)

	six_colors = []

	for ff in range(int(len(filters)/3+1)):
		three = filters[ff*3:(ff+1)*3]
		if len(three) ==2:
			three +=[filters[0]] #place holder 
		elif len(three) ==0:
			continue
		cc123 = c.color_color(planet, star, three[0],three[1] ,three[2],'wfirst')
		six_colors += [cc123[2]]
		six_colors += [cc123[3]]
		six_colors += [cc123[4]]
	#twentysix_colors = twentysix_colors[:-1] #chop off the last since its a repeat


	newdf = pd.DataFrame( {'modelid':headerindex, 'cloud':[planet_dict['cloud']], 'metallicity':[planet_dict['metallicity']], 'distance':[planet_dict['distance']], 
							'phase':[planet_dict['phase']]},index=[0]) 

	addcols = pd.DataFrame({ii:[jj] for ii,jj in zip(filters, six_colors)},index=[0])

	newdf = newdf.join(addcols)
	return newdf
	#ccdf=ccdf.append(newdf,ignore_index = True)



dfs = multiprocessing.Pool().map(runPS, header.index)
ccdf = pd.concat(dfs)

pk.dump(ccdf,open('ccdfmultiprocessing_test.pk','wb'))
"""
ccdf = pk.load(open('ccdfmultiprocessing_test.pk','rb'))

i = 0 
for f in comb(filters,2):#f in filters[2:]:

	#make dataframe on first iteration
	if i ==0:
		f1f2 = -2.5*np.log10(ccdf[f[0]]/ccdf[f[1]])
		f1f2=f1f2.rename(f[0]+f[1])  
		test=pd.concat([ccdf,f1f2],axis=1) 
	else:
		f1f2 = -2.5*np.log10(ccdf[f[0]]/ccdf[f[1]])
		f1f2=f1f2.rename(f[0]+f[1])  
		test = pd.concat([test,f1f2],axis=1)
	i +=1
test.index = test.modelid
pk.dump(test,open('wfirst_parallel_cutofftop.pk','wb'))