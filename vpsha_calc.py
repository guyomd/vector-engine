#!/usr/bin/env python

import os
from openquake.hazardlib.sourceconverter import SourceConverter
from openquake.commonlib.readinput import get_oqparam, get_source_model_lt
from openquake.hazardlib.nrml import to_python


# Set the name of the .ini file:
fname = '/Users/mpagani/Repos/venv/src/oq-engine/demos/hazard/LogicTreeCase2ClassicalPSHA/job.ini'

# Read the .ini file
oqparam = get_oqparam(fname)

# Read the ssc logic tree
ssc_lt = get_source_model_lt(oqparam)

# Creating a source converter
conv = SourceConverter(oqparam.investigation_time, 
                       oqparam.rupture_mesh_spacing, 
                       oqparam.complex_fault_mesh_spacing,
                       oqparam.width_of_mfd_bin, 
                       oqparam.area_source_discretization)
#
# Loop over the realisations and parse the SSMs
for rlz in ssc_lt:
    #
    # Set the name of the model
    ssm_fname = os.path.join(oqparam.base_path, rlz.value)
    #
    # Read the source model
    ssm = to_python(ssm_fname, conv)
    #
    # Info
    print(rlz)
    #
    # Loop over groups included in the source model
    for grp in ssm:
        #
        # Get the update source group for the current realisation
        updated_group = ssc_lt.apply_uncertainties(rlz.lt_path, grp)
        #
        # Loop over sources 
        for src in updated_group:
            #
            # Check if the source is an Area Source
            if type(src).__name__ is 'AreaSource':
                #
                # Loop over the point sources in the area source
                for x in src:
                    #
                    # x is a point source. Now, you can do whatever you want with it.
                    pass

