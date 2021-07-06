## Installation instructions for EDF users ##
1. Activate proxy authentication (use NNI and Sesame password):
`$ edf-proxy-cli.py`

2. Install the Openquake Engine for development.
Detailed instructions are provided here: https://github.com/gem/oq-engine/blob/master/doc/installing/development.md#installing-the-openquake-engine-for-development

3. Clone the __vector-engine__ repository in your `src` directory (where you already cloned the Openquake Engine, according to the instructions above):
`$ cd src`
`$ https_proxy=https://vip-users.proxy.edf.fr:3128 git clone https://github.com/guyomd/vector-engine`

4. Add path to __vector-engine__ in the `PYTHONPATH` environment variable. Add the following line at the bottom of your `~/.bashrc` file:
`$ export PYTHONPATH="$PATH_TO_VECTORENGINE_PARENT_DIRECTORY:$PYTHONPATH"`

5. Activate Openquake Environment 
`$ source openquake/bin/activate`

6. Run program help:
`(openquake) $ python3 v-engine.py -h`

7. Run demo:
`(openquake) $ python3 v-engine.py AreaSourceClassicalPSHA/test_job.ini`
