import os 
import shutil
import glob
import pandas as pd
import numpy as np

from sqlalchemy import *

#create empty dataframe where model info will go 

df= pd.DataFrame({'display_string':[], 'gravity':[], 'tint':[], 'metallicity':[],'distance':[], 'cloud':[], 'phase':[]})

phang = np.linspace(0,180,19)

#function to ignore hidden files when listing files in a directory
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

#file paths 
f_cld = '/Users/batalha/Documents/atmosphere_models/fortney_grid/jfort_cld'
f_ind = '/Users/batalha/Documents/atmosphere_models/fortney_grid/jfort_ind'
f_pt = '/Users/batalha/Documents/atmosphere_models/fortney_grid/jfort_pt'
f_alb = '/Users/batalha/Documents/atmosphere_models/fortney_grid/albspec_2015'

#metalicities 
for m in listdir_nohidden(f_cld): 

	#distances 
	for d_str in listdir_nohidden(os.path.join(f_ind,m)):
		
		ind_file = os.path.join(f_ind,m,d_str)
		pt_file = os.path.join(f_pt,m,d_str[:-4]+'.pt')

		d = d_str[d_str.find('d'):d_str.find('.ind')]
		
		#add no cloud case to clouds
		nc_and_c = list(listdir_nohidden(os.path.join(f_cld,m,d)))
		nc_and_c += ['NC']

		for c_str in nc_and_c:

			#cloud type
			if c_str =='NC':
				c = c_str
			else:
				c = c_str[c_str.find('-f')+1:c_str.find('-d')]

			for p in phang:

				#cld_file = os.path.join(f_cld,m,d,c_str)

				display_string = 'Jupiter '+str(int(10.0**float(m[1:])))+'x '+d[1:]+'AU '+c+', '+str(int(p))

				gravity = float(d_str[1:d_str.find('_t')])

				temp = float(d_str[d_str.find('_t')+2:d_str.find('_m')])

				metallicity = float(m[1:])

				distance = float(d[1:])

				try:
					#if there are clouds
					cloud = float(c[1:])
				except:
					cloud = 0.0

				ind_p = list('000')
				ind_p[3-len(str(int(p))):] = str(int(p))

				index = d_str[:-4]+'_'+c+'_phang'+''.join(ind_p)

				df_new = pd.DataFrame({'display_string':display_string, 'gravity':gravity,
								 'temp':temp, 'metallicity':metallicity,'distance':distance, 
								 				'cloud':cloud, 'phase':p}, index = [index])
				df = df.append(df_new)


#export to json 
db = create_engine('sqlite:///reference/AlbedoModels_2015.db')
df.to_sql('header',db, if_exists='replace')
print(df)


