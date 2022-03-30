# L1 Triggering

Instructions and scripts for CMS Machine Learning workflows.

## Instructions to set things up:

### Pre-requisites:

You'll need the `anaconda` or `miniconda` environment manager. This will ensure you have the correct Python (`Python > 3`) and pip (`pip > 8`) installed.

### Installing

#### Step 1: Make a directory for this work:

```bash
LOC=/software/$USER/ML_software 
ML_SOFTWARE=${LOC}/software
mkdir -p $LOC && cd $LOC
```

#### Step 2: Clone this repository

Note that you might have to create ssh keys and add them to your GitHub account for this to work (see [here](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh)).

```bash
git clone <your repo>
NOTE: make sure your repo contains an "environment.yaml" file which contains all the packages you wish to install.
```

#### Step 3: Set up environment

```bash
PythonVer=3  
MINICONDA_DIR=/software/$USER/miniconda3  # change to wherever you like 
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $MINICONDA_DIR
rm Miniconda3-latest-Linux-x86_64.sh

source $MINICONDA_DIR/etc/profile.d/conda.sh
conda update -y -n base -c defaults conda
export PYTHONNOUSERSITE=true

cd ML_software
conda env create -f requirements.yaml
conda activate ml_env

export PYTHONPATH="$PWD:$PYTHONPATH"
```

Then you can do the following to create a bash script for easy initialisation of the environment in the future:
run by doing: `source start_ml_env.sh`

```bash
init_script="$HOME/start_ml_env.sh"  
echo "MINICONDA_DIR=$MINICONDA_DIR
ML_SOFTWARE=$ML_SOFTWARE
source \$MINICONDA_DIR/etc/profile.d/conda.sh
conda activate ml_env
cd \$ML_SOFTWARE
export PYTHONNOUSERSITE=true
export PYTHONPATH=\"\$PWD:\$PYTHONPATH\"" > $init_script
```

on the DICE network a training set is already prepared here: `/software/ys20884/training_data`
