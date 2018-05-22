import numpy as np

from bokeh.layouts import row, widgetbox, column
from bokeh.models import CustomJS, Slider,Select
from bokeh.plotting import figure, output_file, show, ColumnDataSource
import pickle as pk 
from colorcolor import compute_colors as c
from bokeh.palettes import Spectral10
from bokeh.palettes import  inferno, viridis,grey
from itertools import combinations as comb


DATA = pk.load(open('../notebooks/wfirst_colors_dataframe.pk','rb')).dropna()
DATA = DATA.dropna()[~DATA.dropna().isin([np.inf, -np.inf])]
DATA = DATA.dropna()
DATA=DATA.reset_index(drop=True)

fs = c.print_filters('wfirst')
fcomb = [i[0]+i[1] for i in comb(fs,2)] 
fcomb += [ '506', '575', '661', '721', '883', '940']

x = DATA['575883'].values
y = DATA['506575'].values

metal={}
ii=0
for i in np.unique(DATA['metallicity']):
    metal[str(i)] = inferno(len(np.unique(DATA['metallicity'])))[ii]
    ii+=1 #ColumnDataSource(data=dict( metal = np.unique(DATA['metallicity']),color =inferno(len(np.unique(DATA['metallicity'])))))

phase={}
ii=0
for i in np.unique(DATA['phase']):
    phase[str(i)] =viridis(len(np.unique(DATA['phase'])))[ii]#ColumnDataSource(data=dict( phase = np.unique(DATA['phase']), color = viridis(len(np.unique(DATA['phase'])))))
    ii+=1

dist = {}
ii=0
for i in np.unique(DATA['distance'])[::-1]:
    dist[str(i)] =Spectral10[ii]#ColumnDataSource(data=dict( dist = np.unique(DATA['distance']), color = Spectral10))
    ii+=1

cloud = {}
ii=0
for i in np.unique(DATA['cloud']):
    cloud[str(i)] = grey(len(np.unique(DATA['cloud'])))[ii]
    if i == 0: cloud[str(i)] = 'blue'#ColumnDataSource(data=dict( dist = np.unique(DATA['cloud']), color =grey(len(p.unique(DATA['cloud'])))))
    ii+=1

mc,pc,dc, cc,mn,pn,dn, cn =[],[],[],[],[],[],[],[]
for i in DATA.index: 
    mc += [metal[str(DATA['metallicity'][i])]]
    pc += [phase[str(DATA['phase'][i])]]
    dc += [dist[str(DATA['distance'][i])]]
    cc += [cloud[str(DATA['cloud'][i])]]    

    mn +=[str(DATA['metallicity'][i])]
    pn +=[str(DATA['phase'][i])]
    dn +=[str(DATA['distance'][i])]

    c = DATA['cloud'][i]
    if c == 0: ccc = 'Cloud free' 
    else: ccc = "F_sed= "+str(c)
    cn +=[ccc]

source = ColumnDataSource(data=dict(x=x, y=y, metallicity=mc, phase = pc, distance=dc, cloud=cc,color=dc,legend=dn))
leg_source = ColumnDataSource(data=dict(  metallicity=mn, phase=pn, distance=dn, cloud=cn ))
og_source = ColumnDataSource(data=dict(DATA))

plot = figure(plot_width=1000, plot_height=800, x_axis_label = 'Filter Combination Chosen from Dropdown',
    y_axis_label='Filter Combination Chosen From Dropdown')

plot.circle('x', 'y', color='color',legend='legend',source=source, size=10)

callback = CustomJS(args=dict(source=source, og_source=og_source,leg_source=leg_source), code="""
    var data = source.data;
    var og_data = og_source.data;
    var pick_x = xaxis.value;
    var pick_y = yaxis.value;
    var color_by = color_by.value;
    var leg_data = leg_source.data;
    
    x = data['x'];
    y = data['y'];
    color = data['color'];
    legend = data['legend'];

    for (i = 0; i < x.length; i++) {
        x[i] = og_data[pick_x][i];
        y[i] = og_data[pick_y][i];
        color[i] = data[color_by][i];
        legend[i] = leg_data[color_by][i];
    }
    source.change.emit();
""")



pick_x = Select(title="Pick X Axis Filters:", value="575883", options=fcomb,callback=callback)
callback.args["xaxis"] = pick_x

pick_y = Select(title="Pick Y Axis Filters:", value="506575", options=fcomb,callback=callback)
callback.args["yaxis"] = pick_y

color_by = Select(title="Pick Property to Color By:", value="distance", options=['metallicity','phase','cloud','distance'],callback=callback)
callback.args["color_by"] = color_by


layout = column(
    plot,
    widgetbox(pick_x, pick_y, color_by),
)

#plot.title.text_font_size='18pt'
plot.xaxis.major_label_text_font_size='14pt'
plot.yaxis.major_label_text_font_size='14pt'

output_file("color-color-notitle.html", title="Colors with WFIRST")

"""
        <h1>Planets in WFIRST Color-Color Space</h1>
        <p></p>
        <p></p>
        <h3>Select X and Y filters. Six numbers correspond to ABCDEF: Filter at ABC nanometers - Filter at DEF nanometers </h3>
"""

show(layout)