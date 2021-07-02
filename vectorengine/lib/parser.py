import os
import h5py
import numpy as np

from openquake.commonlib.readinput 
import (get_oqparam,
                                  
                                   get_source_model_lt,
                                           get_gsim_lt,
                                           get_imts)

from openquake.hazardlib.sourceconverter import SourceConverter
from openquake.hazardlib.nrml import to_python
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.geo import Polygon, Point
from openquake.hazardlib.calc.filters import SourceFilter


def parse_openquake_ini(job_ini):
    """
    Parse an XML-formatted Seismic Source Model (SSM) for the Openquake Engine.

    Acknowledgement: M. Pagani

    :param job_ini: str, Path to the Openquake Engine configuration file.

    :return a 2-element tuple containing an instance of the class
    "openquake.commonlib.oqvalidation.OqParam" and another instance of the class
    "openquake.commonlib.logictree.SourceModelLogicTreeCollection"
    """
    # Read the .ini file
    oqparam = readinput.get_oqparam(job_ini)
    # Read model:
    csm = readinput.get_composite_source_model(oqparam)
    # Logic tree for the seismic source model:
    ssm_lt = csm.full_lt.source_model_lt
    # Logic tree for the ground shaking intensity model:
    gsim_lt = csm.full_lt.gsim_lt
    return oqparam, ssm_lt, gsim_lt

def get_value_and_weight_from_rlz(rlz):
    return rlz.value, rlz.weight

def get_value_and_weight_from_gsim_rlz(rlz):
    return rlz.value, rlz.weight['weight']

def parse_sites(oqparam):
    
    """
    if (oqparam.region is not None):
        assert oqparam.region.startswith('POLYGON')
        # Convert region specifications to polygon:
        reg_lons = [float(x.split(' ')[0]) for x in oqparam.region[9:-2].split(', ')]
        reg_lats = [float(x.split(' ')[1]) for x in oqparam.region[9:-2].split(', ')]
        pts = [Point(lon, lat) for lon, lat in zip(reg_lons, reg_lats)]
        poly = Polygon(pts)
        # Convert polygon to sitecolllection:
        mesh = poly.discretize(oqparam.region_grid_spacing)
        sites = [Site(Point(lon, lat, depth),
                      vs30=oqparam.reference_vs30_value,
                      z1pt0=oqparam.reference_depth_to_1pt0km_per_sec,
                      z2pt5=oqparam.reference_depth_to_2pt5km_per_sec)
                 for lon, lat, depth in zip(mesh.lons, mesh.lats, mesh.depths)]
        sitecol = SiteCollection(sites)
    elif isinstance(oqparam.sites, list):
        sites = [Site(Point(s[0], s[1], s[2]),
                      vs30=oqparam.reference_vs30_value,
                      z1pt0=oqparam.reference_depth_to_1pt0km_per_sec,
                      z2pt5=oqparam.reference_depth_to_2pt5km_per_sec)
                 for s in oqparam.sites]
        sitecol = SiteCollection(sites)
    else:
        assert isinstance(oqparam.sites, SiteCollection)
        sitecol = oqparam.sites
    return sitecol
    """
    return readinput.get_site_collection(oqparam)


def load_dataset_from_hdf5(hdf5file, label='hazard_matrix', num_sites=1):
    """
       This function returns a N-D matrix, and a dictionary of IM values
       from a HDF5 archive.
    """
    with h5py.File(hdf5file, 'r') as f:
        dset = f.get(label)
        data = dset[()]
        imtls = dict()
        for k in dset.attrs.keys():
            imtls.update({k: dset.attrs[k]})
    assert data.shape[0] == num_sites  # Default:Only one site permitted
    return np.squeeze(data), imtls


def get_matrix_values_and_axes(hdf5file, label='hazard_matrix'):
    mat, imtls = load_dataset_from_hdf5(hdf5file, label=label)
    per = imtls.keys()
    logx = np.array([np.log(imtls[p]) for p in per])
    x = np.log(np.array([imtls[p] for p in per]))
    return mat, imtls, logx, x
