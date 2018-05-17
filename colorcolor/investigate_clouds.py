import pandas as pd
from bokeh.plotting import figure, show, output_file 
from bokeh.layouts import column,row
import numpy as np
from bokeh.palettes import magma as colfun1
from bokeh.palettes import viridis as colfun2
from bokeh.palettes import Spectral11
from bokeh.models import HoverTool
from bokeh.models import LinearColorMapper, LogTicker, ColorBar,LogColorMapper
import os 


#direc = '/Users/batalha/Documents/atmosphere_models/fortney_grid/jfort_cld'
#mh = 'm0.0'
#dis = 'd1.0'
#fsed = ['f0.01','f0.3','f1']


def investigate_fsed(directory, metal, distance, fsed, plot_file = 'plot.html'):
      """
      This functionw as created to investigate CLD output from the grid created in Batalha+2018. 
      The CLD files are not included in the SQLite database so these file names refer to the 
      specific naming function and directory. These files are available upon request. 
      Email natasha.e.batalha@gmail.com 

      The plot itselfs plots maps of the wavelength dependent single scattering albedo 
      and cloud opacity for up to three different fseds at a single metallicty and distance. 
      It was created becaues there are several times when fsed does funky numerical things 
      in the upper layers of the atmosphere. 

      Parameters
      ----------
      directory : str 
            Directory which points to the fortney grid of cloud output from Ackerman Marley code 
      metal : str 
            Metallicity: Options are m0.0 m0.5  m1.0  m1.5  m1.7  m2.0
      distance : str 
            Distance to G-type star. Options are d0.5 d0.6  d0.7  d0.85 d1.0  d1.5  d2.0  d3.0  d4.0  d5.0
      fsed : list of str 
            Up to three sedimentation efficiencies. Options are f0.01 f0.03 f0.3 f1 f3 f6
      plot_file : str 
            (Optional)This is the html plotting file Default = 'plot.html'
      """
      cols = colfun1(200)
      color_mapper = LinearColorMapper(palette=cols, low=0, high=1)

      dat01 = pd.read_csv(os.path.join(directory, metal, distance,metal+'x_rfacv0.5-nc_tint150-'+fsed[0]+'-'+distance+'.cld'), header=None,delim_whitespace=True)
      dat1 = pd.read_csv(os.path.join(directory, metal, distance,metal+'x_rfacv0.5-nc_tint150-'+fsed[1]+'-'+distance+'.cld'), header=None,delim_whitespace=True)
      dat6 = pd.read_csv(os.path.join(directory, metal, distance,metal+'x_rfacv0.5-nc_tint150-'+fsed[2]+'-'+distance+'.cld'), header=None,delim_whitespace=True)


      scat01 = np.flip(np.reshape(dat01[4],(60,196)),0)#[0:10,:]
      scat1 = np.flip(np.reshape(dat1[4],(60,196)),0)#[0:10,:]
      scat6 = np.flip(np.reshape(dat6[4],(60,196)),0)#[0:10,:]


      xr, yr = scat01.shape

      f01a = figure(x_range=[150, yr], y_range=[0,xr],
                               x_axis_label='Wavelength Grid', y_axis_label='Pressure Grid, TOA ->',
                               title="Scattering Fsed = 0.01",
                              plot_width=300, plot_height=300)

      f1a = figure(x_range=[150, yr], y_range=[0,xr],
                               x_axis_label='Wavelength Grid', y_axis_label='Pressure Grid, TOA ->',
                               title="Scattering Fsed = 0.3",
                              plot_width=300, plot_height=300)


      f6a =figure(x_range=[150, yr], y_range=[0,xr],
                               x_axis_label='Wavelength Grid', y_axis_label='Pressure Grid, TOA ->',
                               title="Scattering Fsed = 1",
                              plot_width=300, plot_height=300)

      f01a.image(image=[scat01],  color_mapper=color_mapper, x=0,y=0,dh=xr,dw = yr)
      f1a.image(image=[scat1],  color_mapper=color_mapper, x=0,y=0,dh=xr,dw = yr)
      f6a.image(image=[scat6], color_mapper=color_mapper, x=0,y=0,dh=xr,dw = yr)

      color_bar = ColorBar(color_mapper=color_mapper, #ticker=LogTicker(),
                           label_standoff=12, border_line_color=None, location=(0,0))

      f01a.add_layout(color_bar, 'right')
 

      #PLOT OPD
      scat01 = np.flip(np.reshape(dat01[2]+1e-60,(60,196)),0)
      scat1 = np.flip(np.reshape(dat1[2]+1e-60,(60,196)),0)
      scat6 = np.flip(np.reshape(dat6[2]+1e-60,(60,196)),0)
      xr, yr = scat01.shape
      cols = colfun2(200)[::-1]
      color_mapper = LogColorMapper(palette=cols, low=1e-3, high=10)


      f01 = figure(x_range=[150, yr], y_range=[0,xr],
                               x_axis_label='Wavelength Grid', y_axis_label='Pressure Grid, TOA ->',
                               title="Optical Depth Fsed = 0.01",
                              plot_width=300, plot_height=300)

      f1 = figure(x_range=[150, yr], y_range=[0,xr],
                               x_axis_label='Wavelength Grid', y_axis_label='Pressure Grid, TOA ->',
                               title="Optical Depth Fsed = 0.3",
                              plot_width=300, plot_height=300)


      f6 =figure(x_range=[150, yr], y_range=[0,xr],
                               x_axis_label='Wavelength Grid', y_axis_label='Pressure Grid, TOA ->',
                               title="Optical Depth Fsed = 1",
                              plot_width=300, plot_height=300)

      f01.image(image=[scat01],  color_mapper=color_mapper, x=0,y=0,dh=xr,dw = yr)
      f1.image(image=[scat1],  color_mapper=color_mapper, x=0,y=0,dh=xr,dw = yr)
      f6.image(image=[scat6], color_mapper=color_mapper, x=0,y=0,dh=xr,dw = yr)

      color_bar = ColorBar(color_mapper=color_mapper, ticker=LogTicker(),
                           label_standoff=12, border_line_color=None, location=(0,0))

      f01.add_layout(color_bar, 'right')
      output_file(plot_file)
      show(row(column(f01a,f1a,f6a),column(f01,f1,f6)))
