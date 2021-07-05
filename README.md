#L1 Triggering

Instructions and scripts for CMS L1 Triggering using Machine Learning.

## Instructions to set things up:

### Pre-requisites:

You'll need the `anaconda` or `miniconda` environment manager. This will ensure you have the correct Python (`Python > 3`) and pip (`pip > 8`) installed.

### Installing

#### Step 1: Make a directory for this work:

```bash
LOC=/software/$USER/triggering_software  # set this to wherever you like
TRIGGER_SOFTWARE=${LOC}/trigger
mkdir -p $LOC && cd $LOC
```

#### Step 2: Clone this repository

Note that you might have to create ssh keys and add them to your GitHub account for this to work (see [here](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh)). Or you can clone with https if you prefer.

```bash
git clone git@github.com:m-glowacki/CMS_trigger.git CMS_trigger
NOTE: use SSH and NOT https protocol.
```

```bash
PythonVer=3  # do PythonVer=2 for Python 2, but Python 3 is recommended
MINICONDA_DIR=/software/$USER/miniconda3  # change to wherever you like (/software/ is Bristol-specific)
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $MINICONDA_DIR
rm Miniconda3-latest-Linux-x86_64.sh

source $MINICONDA_DIR/etc/profile.d/conda.sh
conda update -y -n base -c defaults conda
export PYTHONNOUSERSITE=true

conda env create -f requirements.yaml
conda activate trigger_env

cd $CHIP_SOFTWARE
export PYTHONPATH="$PWD:$PYTHONPATH"
```

Then you can do the following to create a bash script for easy initialisation of the environment in the future:
run by doing: `source start_trigger_env.sh`

```bash
init_script="$HOME/start_chip_env.sh"  # change to whatever you want
echo "MINICONDA_DIR=$MINICONDA_DIR
TRIGGER_SOFTWARE=$TRIGGER_SOFTWARE
source \$MINICONDA_DIR/etc/profile.d/conda.sh
conda activate trigger_env
cd \$TRIGGER_SOFTWARE
export PYTHONNOUSERSITE=true
export PYTHONPATH=\"\$PWD:\$PYTHONPATH\"" > $init_script
```