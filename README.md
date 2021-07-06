# vector-engine #
Vector-valued PSHA calculations, based on the Openquake Engine Python library (https://github.com/gem/oq-engine)

## Installation ##
Set-up a dedicated __vpsha__ virtual environment:
`$ python3 -m venv /path/to/virtual/env/vpsha`

Activate the __vpsha__ virtual environment:
`$ source /path/to/virtual/env/vpsha/bin/activate`

Install the Openquake Engine Python library and dependencies:
`(vpsha) $ pip install openquake.engine`


## Run program ##
Print help:
 $ python3 vectorengine/v-engine.py -h

Full (N-D) hazard matrix calculation:
 $ python3 vectorengine/v-engine.py AreaSourceClassicalPSHA/test_job.ini full --imcm BakerCornell2006

Find N-D (correlated) intensity-measure samples matching POE target value:
 $ python3 vectorengine/v-engine.py AreaSourceClassicalPSHA/test_job.ini optim -n 1 
(Note: POE target value must be specified in the configuration file)

Compute full (N-D) hazard matrix and all N 1-D marginal hazard curve:
 $ python3 vectorengine/v-engine.py AreaSourceClassicalPSHA/test_job.ini calc-marginals

Compute full hazard matrix and make N plots of the comparison between the associated 1-D marginal and the 1-D hazard curve:
 $ python3 vectorengine/v-engine.py AreaSourceClassicalPSHA/test_job.ini plot-marginals
  
