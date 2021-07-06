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
Full (N-D) hazard matrix calculation:
 $ python3 vectorengine/v-engine.py AreaSourceClassicalPSHA/test_job.ini full --imcm BakerCornell2006

