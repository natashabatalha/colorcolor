"""
Script to compute a crap ton of filter combinations. Does it in parallel. 
In this example we are specifically testing the very last thing described in the paper 

That is: we take a bunch of 5% bandpass filters from 0.5-1 micron (in fake.json) and then 
try and find a set of filters that optimizes classification when it is added to the 575 nm filter

"""
import pandas as pd 
import numpy as np 
import json 
import os
from sqlalchemy import *
import pysynphot as psyn 
from itertools import combinations as comb
import colorcolor as c
import pickle as pk
import multiprocessing
#planet_dict = {"cloud": None,"display_string": None,"distance": None,"gravity": 25,"metallicity": None,"phase": None,"temp": 150}

engine = create_engine('sqlite:///' + os.path.join(os.getenv('ALBEDO_DB'),'AlbedoModels.db'))

header = pd.read_sql_table('header',engine)

star = {'temp':5800, 'metal':0.0, 'logg':4.0}

#In the paper this was done for the Fake set of filters 

#load in filter set 
filters = list(c.print_filters('fake'))

#build the initial dataframe with physical planet parameters
ccdf = pd.DataFrame({'modelid':[], 'cloud':[], 'metallicity':[], 'distance':[], 'phase':[]}) 
#add the rest of the columns for each filter
fdf = pd.DataFrame({i:[] for i in filters})
ccdf = ccdf.append(fdf,ignore_index=True)

#this is the parameter space run that calls colorcolor a bunch to compute the individual colors 

def runPS(headerindex):
	#for i in header.index:

	global ccdf 

	planet_dict = header.loc[headerindex]
	planet = c.select_model(planet_dict)

	print(planet_dict['index'])
	#print(planet)

	twentysix_colors = []

	for ff in range(int(len(filters)/3+1)):
		three = filters[ff*3:(ff+1)*3]
		if len(three) ==2:
			three +=['575']
		cc123 = c.color_color(planet, star, three[0],three[1] ,three[2],'fake')
		twentysix_colors += [cc123[2]]
		twentysix_colors += [cc123[3]]
		twentysix_colors += [cc123[4]]
	twentysix_colors = twentysix_colors[:-1] #chop off the last since its a repeat
	#cc456 = c.color_color(planet, star, filters[3],filters[4] ,filters[5],'fake')
	#cc789 = c.color_color(planet, star, filters[6],filters[7] ,filters[8],'fake')
	#cc101112 = c.color_color(planet, star, filters[9],filters[10] ,filters[11],'fake')

	newdf = pd.DataFrame( {'modelid':headerindex, 'cloud':[planet_dict['cloud']], 'metallicity':[planet_dict['metallicity']], 'distance':[planet_dict['distance']], 
							'phase':[planet_dict['phase']]},index=[0]) 

	addcols = pd.DataFrame({ii:[jj] for ii,jj in zip(filters, twentysix_colors)},index=[0])

	newdf = newdf.join(addcols)
	return newdf
	#ccdf=ccdf.append(newdf,ignore_index = True)



dfs = multiprocessing.Pool().map(runPS, header.index)
ccdf = pd.concat(dfs)

#pk.dump(ccdf,open('ccdfmultiprocessing.pk','wb'))

f = '600'
f1f2 = -2.5*np.log10(ccdf['575']/ccdf[f])
f1f2=f1f2.rename('575600')   
test=pd.concat([ccdf,f1f2],axis=1)   
for f in filters[2:]:
	f1f2 = -2.5*np.log10(ccdf['575']/ccdf[f])
	f1f2=f1f2.rename('575'+f)
	test = pd.concat([test,f1f2],axis=1)
pk.dump(test,open('FluxDataFrameFAKE_5pct_575.pk','wb'))