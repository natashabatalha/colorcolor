from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.io import output_file, show
from bokeh.layouts import gridplot, column, row
import numpy as np 
import pandas as pd 
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import pysynphot as psyn
import json 
import os
from sqlalchemy import *
import scipy.signal as scisig
from itertools import combinations as comb


from bokeh.layouts import gridplot
from bokeh.palettes import Spectral10, Colorblind6, Category10
Spectral10 = Spectral10[::-1]
#mdoel database

engine = create_engine('sqlite:///' + os.path.join(os.getenv('ALBEDO_DB'),'AlbedoModels_2015.db'))
header = pd.read_sql_table('header',engine)


def print_available(key):
	"""Return available models for certain parameter space 

	Parameters
	----------
	key : str 
		available parameters are [cloud, distance, gravity, metallicity, phase, temp]

	Returns
	-------
	str, list
		input key and list of available values
	"""
	return key, header[key].unique()



def color_color(planet, star, filter1, filter2, filter3, set, plotstuff=None): 
	"""Returns color color magnitude based on signle planet and star properties 
	
	Function to compute color-color magnitude of WFIRST filters. Takes in three filter options 
	and computes [filter1-filter2, filter2-filter3]. 

	Parameters
	----------
	planet : dict 
		planet dictionary containing albedo spectrum {'WAVELN':[],'GEOMALB':[]} 

	star : dict 
		star dictionary containing following format {'temp':5000, 'metal':0.0, 'logg':4.0}

	filter1 : str
		calls to wfirst.json which contains necessary filter info 

	set : str 
		set which describes which bundle to grab wfirst filters from (avialable: wfirst, vpl)

	(Optional)plotstuff : str
		Default to None but if equal to 'CheckIntegral' it will plot all the color-color integral compoenents

	Returns
	-------
	float , float
		[filter1-filter2, filter2-filter3]

	TODO
	----
	- add wavelength dependence to filters
	"""
	#get filter specs 
	#TODO replace with wavelength dependence 
	if set.lower()=='wfirst':
		with open(os.path.join(os.getenv('ALBEDO_DB'),"WFIRST.json")) as filters: 
			fdata = json.load(filters)
	elif set.lower()=='vpl':
		with open(os.path.join(os.getenv('ALBEDO_DB'),"VPL.json")) as filters: 
			fdata = json.load(filters)
	elif set.lower()=='fake':
		with open(os.path.join(os.getenv('ALBEDO_DB'),"fake.json")) as filters: 
			fdata = json.load(filters)			
	#get stellar sed 
	sed = psyn.Icat("phoenix", star['temp'], star['metal'],star['logg'])
	sed.convert('microns')
	sed.convert('flam')
	swave = sed.wave 
	sflux = sed.flux * 1e4 #(ergs cm-2 s -1 um-1)
	
	#planet properties 
	pwave = planet['WAVELN']  #(micron)
	palb = planet['GEOMALB']  #Albedo

	#select only relevant wavelength 
	minwave = np.min([fdata[filter1]['minwave'], fdata[filter2]['minwave'],fdata[filter3]['minwave']])
	maxwave = np.max([fdata[filter1]['maxwave'], fdata[filter2]['maxwave'],fdata[filter3]['maxwave']])
	palb = palb[(pwave>=minwave) & (pwave<=maxwave)]
	pwave = pwave[(pwave>=minwave) & (pwave<=maxwave)]

	#get planet and star on planet wave grid 
	sflux = np.interp(pwave, swave, sflux)
	
	f1 = pwave*0
	f1[(pwave>fdata[filter1]['minwave']) & (pwave<fdata[filter1]['maxwave'])] = fdata[filter1]['thru']
	f2 = pwave*0
	f2[(pwave>fdata[filter2]['minwave']) & (pwave<fdata[filter2]['maxwave'])] = fdata[filter2]['thru']
	f3 = pwave*0
	f3[(pwave>fdata[filter3]['minwave']) & (pwave<fdata[filter3]['maxwave'])] = fdata[filter3]['thru']

	if (plotstuff == 'CheckIntegral'):# & (~os.path.isfile('CheckIntegral.html')):
		output_file('CheckIntegral.html')
		fil = figure(title='filter', x_axis_label='micron', y_axis_label='Throughput')
		fsed = figure(title='star', x_axis_label='micron', y_axis_label='cgs/micron')
		fpla = figure(title='planet', x_axis_label='micron', y_axis_label='Albedo')
		fcon = figure(title='Convolution', x_axis_label='micron', y_axis_label='Multiplier Eqn.A1 Cahoy 2010')

		fil.line(pwave, f1, legend='f1', color='red')
		fil.line(pwave,f2, legend='f2', color='blue')
		fil.line(pwave, f3, legend='f3', color='green')

		fpla.line(pwave, palb)

		fsed.line(pwave, sflux)

		fcon.line(pwave, f1*palb*sflux, legend='f1', color='red')
		fcon.line(pwave, f2*palb*sflux, legend='f2', color='blue')
		fcon.line(pwave, f3*palb*sflux, legend='f3', color='green')

		show(gridplot( [[fil, fcon],[fpla, fsed]] , merge_tools=False))

	#compute integrals         filter *  albedo  *  sed     * Dlambda 
	f1int = np.sum( f1[:-1] * palb[:-1] * sflux[:-1] * np.diff(pwave))
	f2int = np.sum( f2[:-1] * palb[:-1] * sflux[:-1] * np.diff(pwave))
	f3int = np.sum( f3[:-1] * palb[:-1] * sflux[:-1] * np.diff(pwave) )

	try: 
		f1_f2 = -2.5*np.log10(  f1int / f2int )

		f2_f3 = -2.5*np.log10(  f2int / f3int )
	except: 
		f1_f2 = np.nan
		f2_f3 = np.nan

	return  f1_f2,f2_f3, f1int, f2int, f3int

def remove_out(model, kernel_size= 3):
	"""function to remove crappy spikes 

	Parameters
	----------
	model : dataframe
		dataframe with GEOMALB and WAVELN 
	std : float
		scale for outlier remover 

	Returns
	-------
	dataframe
		same dataframe without outliers in albedo	

	"""
	#remove NaN
	model = model.dropna()
	#for i in model['index'][1:-1]:
	#	vals = model['GEOMALB'][i-1:i+2]
	#	med = np.median(vals) 
	#	x = model['GEOMALB'][i]
	#	if ((x < med - std*np.std(vals)) | (x > med + std*np.std(vals))): 
	#		print("removing", x, i, vals, )
	#		model['GEOMALB'][i] = np.mean(vals.pop(1))
	model['GEOMALB'] = scisig.medfilt(model['GEOMALB'],kernel_size=kernel_size)
	
	return model	

def select_model(planet_dict,kernel_size=3):
	"""given dictionary of planet value return model from SQLite database 

	Parameters
	----------
	planet_dict : dict 
		dictionary with gravity, temp, metallicity, cloud, distance, phase 

	Returns
	-------
	DataFrame 
		Dataframe with two keys GEOMALB and WAVELN
	"""
	gravity_val = float(planet_dict['gravity'])
	temp_val = float(planet_dict['temp'])
	met_val = float(planet_dict['metallicity'])
	cloud_val = float(planet_dict['cloud'])
	dist_val = float(planet_dict['distance'])
	phase_val=float(planet_dict['phase'])
    
	row = header.loc[(header.gravity==gravity_val) & (header.temp==temp_val)
     & (header.metallicity==met_val) & (header.distance==dist_val) & 
     (header.cloud==cloud_val) & (header.phase==phase_val)]
    
	model = pd.read_sql_table(row['index'].values[0],engine)
	a = remove_out(model,kernel_size)
	
	return a


def print_filters(set):
	"""Prints all filter options contained in database..
    
    Parameters
    ----------
    set : str
        set that defines filters bundle to choose. Currently availalbe are: WFIRST VPL fake
	"""
	if set.lower() == 'wfirst':
		with open(os.path.join(os.getenv('ALBEDO_DB'),"WFIRST.json")) as filters:
			data = json.load(filters)
		return data.keys()
	elif set.lower() == 'vpl':
		with open(os.path.join(os.getenv('ALBEDO_DB'),"VPL.json")) as filters:
			data = json.load(filters)
		return data.keys()
	elif set.lower() == 'fake':
		with open(os.path.join(os.getenv('ALBEDO_DB'),"fake.json")) as filters:
			data = json.load(filters)
		return data.keys()

def get_filter(filter, set):
	"""get filter info
	
	Parameters
	----------
	filter : str
		Name of the filter (see print_filters) for options 
	set : str
		Set that defines filter bundle to choose from. Currently availalbe are WFIRST, VPL, fake
	"""
	if set.lower() =='wfirst':
		with open(os.path.join(os.getenv('ALBEDO_DB'),"WFIRST.json")) as filters:
			data = json.load(filters)
		return data[filter]
	elif set.lower() == 'vpl':
		with open(os.path.join(os.getenv('ALBEDO_DB'),"VPL.json")) as filters:
			data = json.load(filters)
		return data[filter]
	elif set.lower() == 'fake':
		with open(os.path.join(os.getenv('ALBEDO_DB'),"fake.json")) as filters:
			data = json.load(filters)
		return data[filter]

def cahoy_fig18(metallicity, distance, clouds, filter1, filter2, filter3, 
			star={'temp':5800, 'metal':0.0, 'logg':4.0}, bokeh_file='color-color.html', plotstuff=None):
	"""Plots 4 c-c diagrams similar to Cahoy+2010 
	
	FUnction to plot figure similar to Cahoy 2010 Figure 18. This figure is a 4 panel figure 
	different metallicities, distnaces, clouds. Distances are plotted as different plots, metal and clouds 
	are overplotted on same figure. NOTE: Number of metallicities * number of clouds <=6. This calles function 
	color_color instead of using precomputed colors. 

	Parameters
	----------
	metallicity : list 
		list of floats in accordance with Albedo Spec Database [0.0, 1.5, 2.0], Number of metallicities * number of clouds <=6

	clouds : list 
		list of floats in accordance with Albedo Spec Database [0.0, 0.3], Number of metallicities * number of clouds <=6

	distance : list 
		list of floats in accordance with albedo spec database [0.85, 2.0,5.0] (only 3 allowed)

	filter1 : str
		first filter see print_filters for options 

	filter2 : str 
		second filter see print_filters for options 

	filter 3 : str 
		third filter see print_filteres for options 

	star : dict 
		star dictionary containing following format {'temp':5800, 'metal':0.0, 'logg':4.0}

	plotstuff : str
		(Optional) Default is none. There are certain plotting checks throughout, i.e .'CheckIntegral'
	"""

	#since there is only one gravity and temperature, these are not allowed to be 
	#explored in parameter space
	planet_dict = {
      "cloud": None,
      "display_string": None,
      "distance": None,
      "gravity": 25,
      "metallicity": None,
      "phase": None,
      "temp": 150}

	plots = {str(distance[0]) : figure(plot_width = 500, plot_height=500, title = str(distance[0])+' AU'),
   			str(distance[1]) : figure(plot_width = 500, plot_height=500,title = str(distance[1])+' AU'),
   			str(distance[2]) : figure(plot_width = 500, plot_height=500,title = str(distance[2])+' AU'), 
   			'all' : figure(plot_width = 500, plot_height=500,title = 'ALL AU') }
	allx = []
	ally = [] 
	allc = []
	alla = []

	
	for d in distance: 
		dx = []
		dy = []
		dc = []	
		da = []
		col_ind = -1
		for m in metallicity: 
	   		for c in clouds:
	   			col_ind+=1 
	   			print(m,c,col_ind)
	   			for p,a in zip([100],[100]):#zip(np.linspace(0,180,10),np.linspace(.1,1,10)):
   					planet_dict["metallicity"] = m
   					planet_dict['distance'] = d
   					planet_dict['cloud'] = c 
   					planet_dict['phase'] = p 

   					model = select_model(planet_dict)
   					print('model')
   					cc = color_color(model, star,filter1, filter2, filter3, plotstuff=plotstuff)
   					
   					dx += [cc[0]]
   					dy += [cc[1]]
   					dc += [Category10[6][col_ind]]
   					da += [a]

   					allx += [cc[0]]
   					ally += [cc[1]]
   					allc += [Category10[6][col_ind]]
   					alla += [a]
   					
		plots[str(d)].circle(dx,dy, color=dc, size = 10, alpha = da )
	plots['all'].circle(allx,ally, color=allc, size = 10 , alpha = alla )

	p1 = plots[str(distance[0])]
	p1.title.text_font_size='18pt'
	p1.xaxis.major_label_text_font_size='14pt'
	p1.yaxis.major_label_text_font_size='14pt'
	p2 = plots[str(distance[1])]
	p2.title.text_font_size='18pt'
	p2.xaxis.major_label_text_font_size='14pt'
	p2.yaxis.major_label_text_font_size='14pt'
	p3 = plots[str(distance[2])]
	p3.xaxis.major_label_text_font_size='14pt'
	p3.yaxis.major_label_text_font_size='14pt'
	p3.title.text_font_size='18pt'
	p4 = plots['all']		
	p4.title.text_font_size='18pt'	
	p4.xaxis.major_label_text_font_size='14pt'	
	p4.yaxis.major_label_text_font_size='14pt'	

	output_file(bokeh_file)
	show(gridplot([[p1, p2],[p3, p4]], merge_tools=False))

	return dx, dy, dc, da, allx, ally, allc, alla

def three_filter_fig(metallicity, distance, clouds, phase, filter1, filter2, filter3, 
			bokeh_file='cc-all.html'):
	"""Plot single plot colored only by distance
	
	FUnction to plot figure similar to Cahoy 2010 Figure 18. This figure is a 4 panel figure 
	different metallicities, distnaces, clouds. Distances are plotted as different plots, metal and clouds 
	are overplotted on same figure. NOTE: Number of metallicities * number of clouds <=6

	Parameters
	----------
	metallicity : list 
		list of floats in accordance with Albedo Spec Database [0.0, 1.5, 2.0]

	clouds : list 
		list of floats in accordance with Albedo Spec Database [0.0, 0.3]

	distance : list 
		list of floats in accordance with albedo spec database [0.85, 2.0,5.0] 

	phase : list 
		list of floats in accordance with albedo spec database [0,60] 

	filter1 : str
		first filter see print_filters for options 

	filter2 : str 
		second filter see print_filters for options 

	filter 3 : str 
		third filter see print_filteres for options 

	star : dict 
		star dictionary containing following format {'temp':5800, 'metal':0.0, 'logg':4.0}

	plotstuff : str
		(Optional) Default is none. There are certain plotting checks throughout, i.e .'CheckIntegral'
	"""

	#since there is only one gravity and temperature, these are not allowed to be 
	#explored in parameter space



	allx = []
	ally = [] 
	allc = []
	allm = []
	allcld = []
	allp=[]
	alld=[]

	ds = np.array(print_available('distance')[1])
	
	df = allcolors.dropna()[~allcolors.dropna().isin([np.inf, -np.inf]).any(1)]
	
	for i in df.index: 
		
		planet_dict = header.loc[header['index'] == allcolors['modelid'][i]]

		#check for right planet parameters
		if float(planet_dict['phase']) not in phase: 
			continue
		if float(planet_dict['metallicity']) not in metallicity: 
			continue
		if float(planet_dict['cloud']) not in clouds: 
			continue 

		if float(planet_dict['distance']) not in list(distance): 
			continue 
		#check for right filters 
		filters = [str(allcolors['f1'][i]),str(allcolors['f2'][i]),str(allcolors['f3'][i])]
		if (filter1 in filters) and (filter2 in filters) and (filter3 in filters): 
			print(filters)
		else:
			continue 	

		allx += [allcolors['f2_f3'][i]]
		ally += [allcolors['f1_f2'][i]]
		print(allcolors['f1_f2'][i], allcolors['f2_f3'][i])
		#color by distance 
		d =float( planet_dict['distance'])

		allc += [Spectral10[np.where(ds == d)[0][0]]]
		allm +=[float(planet_dict['metallicity'])]
		allcld +=[float(planet_dict['cloud'])]
		allp +=[float(planet_dict['phase'])]
		alld +=[float(planet_dict['distance'])]

	source = ColumnDataSource(data = dict(
		x = allx , 
		y = ally ,
		color = allc , 
		m = allm, 
		c = allcld, 
		p = allp,
		d = alld))

	hover = HoverTool(tooltips=[
    ("Metallicity", "@m"),
    ("Cloud f", "@c"),
    ("Phase", "@p"),
	("Distance", "@d")])

	plot = figure(plot_width = 700, plot_height=700,y_axis_label = filter1+'-'+filter2
				,x_axis_label = filter2+'-'+filter3, tools=[hover,'resize','box_zoom','save','pan','wheel_zoom','reset'])	

	plot.circle('x','y', color='color', source =source,size = 10, alpha = 0.5)

	p1 = plot
	p1.title.text_font_size='18pt'
	p1.xaxis.major_label_text_font_size='14pt'
	p1.yaxis.major_label_text_font_size='14pt'
	

	output_file(bokeh_file)
	show(p1)

	return allx, ally, allc

def four_filter_fig(metallicity, distance, clouds, phase,filter1, filter2, filter3, filter4,
			bokeh_file='cc-4-all.html'):
	"""Plot single plot colored only by distance
	
	Function to plot all models colord by distance for 4 filters. 

	Parameters
	----------
	metallicity : list 
		list of floats in accordance with Albedo Spec Database [0.0, 1.5, 2.0]

	clouds : list 
		list of floats in accordance with Albedo Spec Database [0.0, 0.3]

	distance : list 
		list of floats in accordance with albedo spec database [0.85, 2.0,5.0] 

	phase : list 
		list of floats in accordance with albedo spec database [0,60] 

	filter1 : str
		first filter see print_filters for options 

	filter2 : str 
		second filter see print_filters for options 

	filter 3 : str 
		third filter see print_filteres for options
	filter 4 : str 
		fourth filter see print_filteres for options 
	star : dict 
		star dictionary containing following format {'temp':5800, 'metal':0.0, 'logg':4.0}

	plotstuff : str
		(Optional) Default is none. There are certain plotting checks throughout, i.e .'CheckIntegral'
	"""

	allx = []
	ally = [] 
	allc = []
	allm = []
	allcld = []
	allp=[]
	alld=[]

	ds = np.array(print_available('distance')[1])
	
	for i in allfluxes.index: 

		planet_dict = header.loc[header['index'] == allfluxes['modelid'][i]]

		#check for right planet parameters
		if float(planet_dict['phase']) not in phase: 
			continue
		if float(planet_dict['metallicity']) not in metallicity: 
			continue
		if float(planet_dict['cloud']) not in clouds: 
			continue 

		if float(planet_dict['distance']) not in list(distance): 
			continue 
		
		#check for right filters 
		f1 = allfluxes[filter1][i]
		f2 = allfluxes[filter2][i]
		f3 = allfluxes[filter3][i]
		f4 = allfluxes[filter4][i]

		if 0 in [f1 , f2 , f3 , f4 ]: 
			continue

		try:
			allx += [-2.5*np.log10(  f3 / f4 )]
			ally += [-2.5*np.log10(  f1 / f2 )]
		except:
			continue

		#color by distance 
		d =float( planet_dict['distance'])
		allc += [Spectral10[np.where(ds == d)[0][0]]]
		allm +=[float(planet_dict['metallicity'])]
		allcld +=[float(planet_dict['cloud'])]
		allp +=[float(planet_dict['phase'])]
		alld +=[float(planet_dict['distance'])]

	source = ColumnDataSource(data = dict(
		x = allx , 
		y = ally ,
		color = allc , 
		m = allm, 
		c = allcld, 
		p = allp,
		d = alld))

	hover = HoverTool(tooltips=[
    ("Metallicity", "@m"),
    ("Cloud f", "@c"),
    ("Phase", "@p"),
	("Distance", "@d")])

	plot = figure(plot_width = 700, plot_height=700,y_axis_label = filter1+'-'+filter2
				,x_axis_label = filter3+'-'+filter4, tools=[hover,'resize','box_zoom','save','pan','wheel_zoom','reset'])	

	plot.circle('x','y', color='color', source =source,size = 10, alpha = 0.5)

	p1 = plot
	p1.title.text_font_size='18pt'
	p1.xaxis.major_label_text_font_size='14pt'
	p1.yaxis.major_label_text_font_size='14pt'
	

	output_file(bokeh_file)
	show(p1)

	return allx, ally, allc

#WFIRST
def WFIRST_colors(output_file,star={'temp':5800, 'metal':0.0, 'logg':4.0}):
	"""
	Creates a dataframe with all possible absolute maginitudes and colors for the Albedo Model Grid 
	This does not require anything but an output filename but it assumes that you have already 
	created a pointer to the AlbedoModel database via the environment variable "Albedo_DB"

	Parameters 
	----------
	output_file : str
		Output pickle file where the dataframe will get stored. 
	star : dict 
		(Optional) Default is Sun-like. You can feel free to change the temp, metal and logg of the star but note that 
		this will still pull the original model set that was computed for a sun like star
	"""
	filters = list(c.print_filters('wfirst'))

	ccdf = pd.DataFrame({'modelid':[], filters[0]:[], filters[1]:[], filters[2]:[], filters[3]:[], filters[4]:[], filters[5]:[],
		                'cloud':[], 'metallicity':[], 'distance':[], 'phase':[]}) 

	for i in header.index:

		planet_dict = header.loc[i]
		planet = c.select_model(planet_dict)

		print(planet_dict['index'])

		cc123 = c.color_color(planet, star, filters[0],filters[1] ,filters[2],'wfirst')
		cc456 = c.color_color(planet, star, filters[3],filters[4] ,filters[5],'wfirst')

		newdf = pd.DataFrame( {'modelid':planet_dict['index'], filters[0]:cc123[2], filters[1]:cc123[3], filters[2]:cc123[4], 
								filters[3]:cc456[2], filters[4]:cc456[3], filters[5]:cc456[4],
								'cloud':[planet_dict['cloud']], 'metallicity':[planet_dict['metallicity']], 'distance':[planet_dict['distance']], 'phase':[planet_dict['phase']]}, index = [0]) 
		
		ccdf=ccdf.append(newdf,ignore_index = True )

	for f in comb(filters,2):
	    f1f2 = -2.5*np.log10(ccdf[f[0]]/ccdf[f[1]])
	    ccdf = ccdf.join(pd.DataFrame({f[0]+f[1]:f1f2}))

	pk.dump(ccdf,open(output_file,'wb'))



#VPL
def VPL_colors(output_file,star={'temp':5800, 'metal':0.0, 'logg':4.0}):
	"""
	Creates a dataframe with all possible absolute maginitudes and colors for the Albedo Model Grid 
	and the VPL colors from Krissansen-Totten 2016
	This does not require anything but an output filename but it assumes that you have already 
	created a pointer to the AlbedoModel database via the environment variable "Albedo_DB"

	Parameters 
	----------
	output_file : str
		Output pickle file where the dataframe will get stored. 
	star : dict 
		(Optional) Default is Sun-like. You can feel free to change the temp, metal and logg of the star but note that 
		this will still pull the original model set that was computed for a sun like star
	"""
	filters = list(c.print_filters('vpl'))
	ccdf = pd.DataFrame({'modelid':[], filters[0]:[], filters[1]:[], filters[2]:[], filters[3]:[], filters[4]:[],
		                'cloud':[], 'metallicity':[], 'distance':[], 'phase':[]}) 
	for i in header.index:

		planet_dict = header.loc[i]
		planet = c.select_model(planet_dict)

		print(planet_dict['index'])
		cc123 = c.color_color(planet, star, filters[0],filters[1] ,filters[2],'vpl')
		cc456 = c.color_color(planet, star, filters[3],filters[4] ,filters[1],'vpl')

		newdf = pd.DataFrame( {'modelid':i, filters[0]:cc123[2], filters[1]:cc123[3], filters[2]:cc123[4], 
	            filters[3]:cc456[2], filters[4]:cc456[3],
								'cloud':[planet_dict['cloud']], 'metallicity':[planet_dict['metallicity']], 'distance':[planet_dict['distance']], 
								'phase':[planet_dict['phase']]}, index = [0]) 
		ccdf=ccdf.append(newdf,ignore_index = True )
	for f in comb(filters,2):
		f1f2 = -2.5*np.log10(ccdf[f[0]]/ccdf[f[1]])
		ccdf = ccdf.join(pd.DataFrame({f[0]+f[1]:f1f2}))

	pk.dump(ccdf,open(output_file,'wb'))



#Fake SET
filters = list(c.print_filters('fake'))
ccdf = pd.DataFrame({'modelid':[], 'cloud':[], 'metallicity':[], 'distance':[], 'phase':[]}) 
fdf = pd.DataFrame({i:[] for i in filters})
ccdf = ccdf.append(fdf,ignore_index=True)



