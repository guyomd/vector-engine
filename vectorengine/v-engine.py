import time
import sys
import logging
from datetime import datetime

from openquake.baselib import parallel, config, sap
from openquake.commonlib.oqvalidation import OqParam

from vectorengine.lib.main import run_job
from vectorengine.lib import imcm


#TERMINATE = config.distribution.terminate_workers_on_revoke
OQ_DISTRIBUTE = parallel.oq_distribute()

logging.info('OQ_DISTRIBUTE set to "{}"'.format(OQ_DISTRIBUTE))
if OQ_DISTRIBUTE.startswith('celery'):
    import celery.task.control
    
    def set_concurrent_tasks_default():
        stats = celery.task.control.inspect(timeout=1).stats()
        if not stats:
            logging.critical('No live computation nodes, aborting calculation')
            sys.exit(1)
        ncores = sum(stats[k]['pool']['max-concurrency'] for k in stats)
        parallel.CT = ncores * 2
        OqParam.concurrent_tasks.default = ncores * 2
        logging.warning('Using %s, %d cores',','.join(sorted(stats)), ncores)

    def celery_cleanup(terminate):
        """
        Release the resources used by an openquake job.
        In particular revoke the running tasks (if any).
        :param bool terminate: the celery revoke command terminate flag
        :param tasks: celery tasks
        """
        # Using the celery API, terminate and revoke and terminate any running
        # tasks associated with the current job.
        tasks = parallel.Starmap.running_tasks
        if tasks:
            logging.warning('Revoking %d tasks', len(tasks))
        else:  # this is normal when OQ_DISTRIBUTE=no
            logging.debug('No task to revoke')
        while tasks:
            task = tasks.pop()
            tid = task.task_id
            celery.task.control.revoke(tid, terminate=terminate)
            logging.debug('Revoked task %s', tid)
else:

    def set_concurrent_tasks_default():
        pass


if __name__ == "__main__":

    def _parse_inputs(ini_file,  mode, *, nb_runs=1, quantity='POE', 
                      imcm='BakerJayaram2008', log='INFO'):
        return ini_file, mode, nb_runs, quantity.lower(), imcm, log
    
    # Set-up argument parser:
    _parse_inputs.ini_file = 'configuration file, e.g. job.ini'
    _parse_inputs.mode = 'calculation mode: "full", "optim", "return-period", "calc-marginals", "plot-marginals"'
    _parse_inputs.nb_runs = 'number of N-D hazard samples matching target POE in "optim" mode'
    _parse_inputs.quantity = 'hazard curve unit: POE or ARE'
    _parse_inputs.imcm = 'inter-IM correlation model, e.g. "BakerJayaram2008"'
    _parse_inputs.log = 'verbosity level: DEBUG, INFO, WARNING, ERROR, CRITICAL'
    job_ini, calc_mode, n_runs, hc_unit, imcm_class, loglevel = \
            sap.run(_parse_inputs, prog="v-engine.py")

    # Set logging level:
    logging.basicConfig(level=getattr(logging,loglevel),
            format='[%(asctime)s] %(message)s') 

    # Initialize parallelization:
    parallel.Starmap.init()

    try:
        logging.info('Setting up default settings for concurrent tasks')
        set_concurrent_tasks_default()
        t0 = time.time()
        logging.info('Starting VPSHA computation run on {}'.format(datetime.now()))
        job_ini = sys.argv[1]
        run_job(job_ini, quantity=hc_unit, calc_mode=calc_mode, nb_runs=n_runs, 
                cm=getattr(imcm,imcm_class)())
        logging.warning('Calculation finished correctly in {:.1f} seconds'.format(time.time()-t0))
    finally:
        parallel.Starmap.shutdown()
        # Deactivate the following two lines due to incompatibility version 3.11 : 
        #if OQ_DISTRIBUTE.startswith('celery'):
        #    celery_cleanup(TERMINATE)

