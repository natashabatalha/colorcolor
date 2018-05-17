"""
Script to insert the corresponding models into the SQLite database 
"""
import os 
import pandas as pd
from sqlalchemy import *
from sqlalchemy import Table, MetaData
import colcol 

#specify where the grid lives 
f_alb = '/Users/batalha/Documents/atmosphere_models/fortney_grid/albspec_2015'

#function to skip over hidden files
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

#define your database name 
db = create_engine('sqlite:///' + os.path.join(os.getenv('ALBEDO_DB'),'AlbedoModels_2015.db'))

#metalicities
for m in ['m0.0','m0.5','m1.0','m1.5','m1.7','m2.0']:
	#distances
	for d in listdir_nohidden(os.path.join(f_alb,m)):
		#access all files
		for readin in listdir_nohidden(os.path.join(f_alb,m,d)):

			#if it doesnt have an fsed then its cloud free.. 
			if readin.find('_f')==-1:
				table_name = readin[0:readin.find('_p')]+'_NC'+readin[readin.find('_p'):-4]
				print(table_name)

			else: 
				table_name = readin[:-4]
				print(table_name)
			#read the file and pull out what we need
			spec = pd.read_csv(os.path.join(f_alb,m,d, readin), delim_whitespace=True, skip_blank_lines=True, header=1)
			spec = spec[['WAVELN', 'GEOMALB']]
			try:
				spec = spec[spec['GEOMALB']!='***********']
				spec = spec.apply(pd.to_numeric)
			except:
				spec = spec

			#remove spikes
			spec = colcol.remove_out(spec)
			#stick into database
			spec.to_sql(table_name, db, if_exists='replace')
