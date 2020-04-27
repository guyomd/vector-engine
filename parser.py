import os
from openquake.hazardlib.sourceconverter import SourceConverter
from openquake.commonlib.readinput import (get_oqparam,
                                           get_source_model_lt,
                                           get_gsim_lt)
from openquake.hazardlib.nrml import to_python
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.geo import Polygon, Point

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
    oqparam = get_oqparam(job_ini)
    # Read the ssc logic tree
    ssm_lt = get_source_model_lt(oqparam)
    # Reag the gsim logic tree
    gsim_lt = get_gsim_lt(oqparam)
    return oqparam, ssm_lt, gsim_lt


def get_sources_from_rlz(rlz, oqparam, ssm_lt):
    """
    :param rlz: "openquake.commonlib.logictree.Realization" instance
    :param oqparam: "openquake.commonlib.oqvalidation.OqParam" instance
    :param ssm_lt: instance of class
                  "openquake.commonlib.logictree.SourceModelLogicTreeCollection"

    :return : a list of seismic sources instances, e.g.
                  "openquake.hazardlib.source.area.AreaSource"
    """
    # Creating a source converter
    conv = SourceConverter(oqparam.investigation_time,
                           oqparam.rupture_mesh_spacing,
                           oqparam.complex_fault_mesh_spacing,
                           oqparam.width_of_mfd_bin,
                           oqparam.area_source_discretization)
    # Set the name of the model
    ssm_fname = os.path.join(oqparam.base_path, rlz.value)
    # Read the source model
    ssm = to_python(ssm_fname, conv)
    # Loop over groups included in the source model
    sources = []
    for grp in ssm:
        # Update source group for the current realisation:
        updated_group = ssm_lt.apply_uncertainties(rlz.lt_path, grp)
        # Loop over sources:
        for src in updated_group:
            sources.append(src)
    return sources

def get_value_and_weight_from_rlz(rlz):
    return rlz.value, rlz.weight

def get_value_and_weight_from_gsim_rlz(rlz):
    return rlz.value, rlz.weight['weight']

def parse_sites(oqparam):
    if (oqparam.region is not None):
        assert oqparam.region.startswith('POLYGON')
        """
        raise ValueError('More than one site is specified (N={len(site_ctx)}). Although '
                     'technically faesible using OQ library, this is not reasonable '
                     'for VPSHA calculations')
        """
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
        sites_col = SiteCollection(sites)
    elif isinstance(oqparam.sites, list):
        sites = [Site(Point(s[0], s[1], s[2]),
                      vs30=oqparam.reference_vs30_value,
                      z1pt0=oqparam.reference_depth_to_1pt0km_per_sec,
                      z2pt5=oqparam.reference_depth_to_2pt5km_per_sec)
                 for s in oqparam.sites]
        sites_col = SiteCollection(sites)
    else:
        assert isinstance(oqparam.sites, SiteCollection)
        sites_col = oqparam.sites
    return sites_col