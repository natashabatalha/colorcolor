# colorcolor : Color Classification of Extrasolar Giant Planets

1. Installation can be done via pip, but the notebooks included in the github repository will be helpful in running the code. The github also includes `json` files for the filers used in the paper. You can use these or make your own. 

`pip install colorcolor` 

or 

`git clone https://github.com/natashabatalha/colorcolor.git`


2. Download Database from zendo and place in the `reference` folder with your filter json files. 

https://zenodo.org/record/2003949#.XAlz6i3MxIU

3. Export database path: 

export ALBEDO_DB="/path/to/directory/with/reference/"


`Notebooks/` folder contains step-by-step instructions to reproduce results of Batalha+2018. 


[![Build Status](https://travis-ci.org/natashabatalha/colorcolor.svg?branch=master)](https://travis-ci.org/natashabatalha/colorcolor)
