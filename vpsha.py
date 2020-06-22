import time
import sys
import logging
from datetime import datetime

from openquake.baselib import parallel, config 
from openquake.commonlib.oqvalidation import OqParam

from lib.main import run_job

# HELP:
# To launch the script use the following command:
#   python3 vpsha.py "AreaSourceClassicalPSHA/job.ini"
#
# Below the following parameters can be changed by the user:
#  quantity = "poe" or "are"

TERMINATE = config.distribution.terminate_workers_on_revoke
OQ_DISTRIBUTE = parallel.oq_distribute()
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(message)s') # In command-line "--log=INFO", other levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

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
    parallel.Starmap.init()

    if len(sys.argv)>2:
        calc_mode = sys.argv[2]
    else:
        calc_mode = "full"

    if len(sys.argv)>3 and calc_mode=='optim':
        n_runs = int(sys.argv[3])
    else:
        n_runs = 1

    try:
        logging.info('Setting up default settings for concurrent tasks')
        set_concurrent_tasks_default()
        t0 = time.time()
        logging.info('Starting VPSHA computation run on {}'.format(datetime.now()))
        job_ini = sys.argv[1]
        run_job(job_ini, quantity='poe', calc_mode=calc_mode, nb_runs=n_runs)
        logging.warning('Calculation finished correctly in {:.1f} seconds'.format(time.time()-t0))
    finally:
        parallel.Starmap.shutdown()
        if OQ_DISTRIBUTE.startswith('celery'):
            celery_cleanup(TERMINATE)

