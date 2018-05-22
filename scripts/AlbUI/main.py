from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox
from bokeh.models.widgets import Slider, Select, Button
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, CustomJS#, HoverTool, Div
import numpy as np
import os 
import pandas as pd
from sqlalchemy import *
from bokeh.palettes import Spectral6
import json
from colorcolor import compute_colors as c

#establish sql connection to albedos database
engine = create_engine('sqlite:///' + os.path.join(os.getenv('ALBEDO_DB'),'AlbedoModels_2015.db'))
meta_alb = pd.read_sql_table('header',engine)


axis_map = {
    "Wavelength": "WAVELN",
    "Planetary Albedo": "GEOMALB"
}

# Create Input controls
gravity = Select(title="Gravity (m/s^2)", value='25.0', options=list(meta_alb.gravity.unique().astype(np.str)))
temp = Select(title="Temperature (K)", value='150', options=list(meta_alb.temp.unique().astype(np.str)))
metallicity = Slider(title="Metallicity", start=0.0, end=2, value=0.0, step=0.5)
distance = Select(title="Distance (AU)", value='0.5', options=list(meta_alb.distance.unique().astype(np.str)))
cloud = Select(title="Cloud Type", value='0.0', options=list(meta_alb.cloud.unique().astype(np.str)))
phase = Slider(title="Phase Angle", start=0.0, end=180, value=0, step=10)
x_axis = Select(title="X Axis", options=sorted(axis_map.keys()), value="Wavelength")
y_axis = Select(title="Y Axis", options=sorted(axis_map.keys()), value="Planetary Albedo")

button = Button(label="Download", button_type="success")

# Create Column Data Source that will be used by the plot (add hovers here if wanted)
#source = ColumnDataSource(data=dict(x=[], y=[], color=[], title=[], hover1=[], hover2=[], alpha=[]))
source = ColumnDataSource(data=dict(x=[], y=[]))

'''
hover = HoverTool(tooltips=[
    ("Title", "@title"),
    ("Year", "@year"),
    ("$", "@revenue")
])
'''
filters =  list(c.print_filters('wfirst'))
left = [c.get_filter(i, 'wfirst')['minwave'] for i in filters]
right = [c.get_filter(i, 'wfirst')['maxwave'] for i in filters]

p = figure(plot_height=600, plot_width=700, title="",y_range=[0,1], tools='pan,wheel_zoom,box_zoom,save,reset')#toolbar_location=None, tools=[hover])
p.quad(top = [1,1,1,1,1,1], bottom=[0.,0.,0.,0.,0.,0.], left=left, right=right, color =Spectral6 ,alpha = 0.2)

p.line(x="x", y="y", source=source, color="blue", alpha=0.5)

def select_model():
    gravity_val = float(gravity.value)
    temp_val = float(temp.value)
    met_val = float(metallicity.value)
    cloud_val = float(cloud.value)
    dist_val = float(distance.value)
    phase_val=float(phase.value)
    row = meta_alb.loc[(meta_alb.gravity==gravity_val) & (meta_alb.temp==temp_val)
     & (meta_alb.metallicity==met_val) & (meta_alb.distance==dist_val) & 
     (meta_alb.cloud==cloud_val) & (meta_alb.phase==phase_val)]
    return row


def update():
    df = select_model()
    x_name = axis_map[x_axis.value]
    y_name = axis_map[y_axis.value]

    p.xaxis.axis_label = x_axis.value
    p.yaxis.axis_label = y_axis.value
    p.title.text = df['display_string'].values[0]
    source.data = dict(
        x=pd.read_sql_table(df['index'].values[0],engine)[x_name],
        y=pd.read_sql_table(df['index'].values[0],engine)[y_name]
    )

def download():
    df = select_model()
    x_name = axis_map[x_axis.value]
    y_name = axis_map[y_axis.value]
    filename = os.path.expanduser('~')+'/Downloads/'+df['display_string'].values[0] + '.txt'
    np.savetxt(filename, np.c_[pd.read_sql_table(df['index'].values[0],engine)[x_name], 
        pd.read_sql_table(df['index'].values[0],engine)[y_name]])

button.on_click(download)

controls = [gravity, temp, metallicity, distance, cloud, phase, x_axis, y_axis]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

sizing_mode = 'fixed'  # 'scale_width' also looks nice with this example

inputs = widgetbox(*controls, sizing_mode=sizing_mode)
l = layout([
    [inputs, p],
    [button]
], sizing_mode=sizing_mode)

update()  # initial load of the data

curdoc().add_root(l)
curdoc().title = "Albedo Models"


