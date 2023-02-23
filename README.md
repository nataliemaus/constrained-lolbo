# Local Latent Bayesian Optimization (LOLBO)
Official implementation of Local Latent Space Bayesian Optimization over Structured Inputs https://arxiv.org/abs/2201.11872, leveraging 
Scalable Constrained BO (https://arxiv.org/abs/2002.08526) to allow for constraints to be added. 

All constraints in form c(x) <= 0.
Surrogate model used to model constraint function. 

This repository includes base code to run LOLBO, along with full implementation for GuacaMol and Penalized logP Molecular Optimization tasks to replicate results in Figure 1 left and Figure 2 in the original paper. The base code is also set up to allow LOLBO to be run on any other latent space optimization task. Please contact the authors if you need any help with applying the LOLBO to run the other two tasks from the paper (Arithmetic Expressions, DRD3 Receptor Docking Affinity), or any other latent space optimization task. 

## Weights and Biases (wandb) tracking
This repo it set up to automatically track optimization progress using the Weights and Biases (wandb) API. Wandb stores and updates data during optimization and automatically generates live plots of progress. If you are unfamiliar with wandb, we recommend creating a free account here:
https://wandb.ai/site
Otherwise, the code can also be run without wandb tracking by simply setting the argument `--track_with_wandb False` (see example commands below). 

## Getting Started

### Cloning the Repo (Git Lfs)
This repository uses git lfs to store larger data files and model checkpoints. Git lfs must therefore be installed before cloning the repository. 

```Bash
conda install -c conda-forge git-lfs
```

### Environment Setup (Conda)
Follow the steps below to install all dependencies to run LOLBO. Execute in the repository ROOT.

```Bash
conda env create -f lolbo_env.yml
conda activate lolbo_conda_env1
pip install molsets==0.3.1 --no-deps
```

The resultant environment will have all imports necessary to run LOLBO on the example GuacaMol and Penalized Log P molecular optimization tasks in this repo.

## How to Run LOLBO on Our Example Molecular Optimization Tasks

In this section we provide commands that can be used to start a LOLBO optimization after the environment has been set up. 

### Args:

To start an molecule optimization run, run `scripts/molecule_optimization.py` with desired command line arguments. To get a list of command line args specifically for the molecule optimization tasks with the SELFIES VAE, run the following: 

```Bash
cd scripts/

python3 molecule_optimization.py -- --help
```

For a list of the remaining possible args that are the more general LOLBO args (not specific to molecule tasks) run the following:

```Bash
cd scripts/

python3 optimize.py -- --help
```

The above commands will give defaults for each arg and a description of each. The only required argument is `--task_id`, which is the string id that determines the optimization task. 

### Task IDs
#### GuacaMol Task IDs
This code base provides support for the following 12 GuacaMol Optimization Tasks:

| task_id | Full Task Name     |
|---------|--------------------|
|  med1   | Median molecules 1 |
|  med2   | Median molecules 2 |
|  pdop   | Perindopril MPO    |
|  osmb   | Osimertinib MPO    |
|  adip   | Amlodipine MPO     |
|  siga   | Sitagliptin MPO    |
|  zale   | Zaleplon MPO       |
|  valt   | Valsartan SMARTS   |
|  dhop   | Deco Hop           |
|  shop   | Scaffold Hop       |
|  rano   | Ranolazine MPO     |
|  fexo   | Fexofenadine MPO   |

The original LOLBO paper features results on zale, pdop, and rano. For descriptions of these and the other GuacaMol objectives listed, as well as a leaderboard for each of these tasks, see https://www.benevolent.com/guacamol

#### Penalized Log P Task ID
To run on Penalized Log P instead of one of the above GuacaMOl tasks, use the following four-letter task id:

```
logp --> Penalized Log P
```

### Example Command to Optimize Penalized Log P with LOLBO 
##### (Replicates Result in Figure 1 Left of Paper)

```Bash
cd scripts/

CUDA_VISIBLE_DEVICES=0 python3 molecule_optimization.py --task_id logp --max_n_oracle_calls 500 --bsz 1 --k 10 - run_lolbo - done 
```
#### Command Modified to Run With Weights and Biases (wandb) Tracking
```Bash
CUDA_VISIBLE_DEVICES=0 python3 molecule_optimization.py --task_id logp --max_n_oracle_calls 500 --bsz 1 --k 10 --track_with_wandb True --wandb_entity $YOUR_WANDB_USERNAME - run_lolbo - done 
```

### Example Command to GuacaMol Objectives with LOLBO
To replicate the result in Figure 2 of the paper, simply run the below command three times - once each with `--task_id pdop`, `--task_id rano`, and `--task_id zale` (modify as above to run with wandb tracking):

```Bash
cd scripts/

CUDA_VISIBLE_DEVICES=1 python3 molecule_optimization.py --task_id zale --max_string_length 400 --max_n_oracle_calls 120000 --bsz 10 - run_lolbo - done 
```

runai submit valt2 -v /home/nmaus/:/workspace/ --working-dir /workspace/constrained-lolbo/scripts -i nmaus/lolbo -g 1 \
--command -- python3 molecule_optimization.py --task_id valt --max_string_length 400 --max_n_oracle_calls 120000 --bsz 10 --recenter_only False - run_lolbo - done 

### Example Command to run Docking w/ DRD3

#### DOCKSTRING VERSION 
conda activate dockstring 

CUDA_VISIBLE_DEVICES=8 python3 molecule_optimization.py --task_id dock_drd3 --track_with_wandb True --wandb_entity nmaus --max_string_length 400 --max_n_oracle_calls 2500000000000 --bsz 1 - run_lolbo - done 

#### TDC VERSION 
conda activate tdc_env

CUDA_VISIBLE_DEVICES=0 python3 molecule_optimization.py --task_id tdc_drd3 --track_with_wandb True --wandb_entity nmaus --max_string_length 400 --max_n_oracle_calls 10000 --bsz 1 --k 10 --num_initialization_points 100 - run_lolbo - done 

## How to Run LOLBO on Other Tasks
To run LOLBO on other tasks, you'll need to write two new classes: 

1. Objective Class

Create a new child class of LatentSpaceObjective (see `lolbo/latent_space_objective.py`) which all outlined methods defined. 

See example objective class for molecule tasks: 
`lolbo/molecule_objective.py `

2. Top Level Optimization Class

Create a new child class of Optimize (see `scripts/optimize.py`) which defines the two methods specific to the optimization task. The first is the `initialize_objective` method, which defines the variable `self.objective` to be an object of the new objective class created in step 1. The second is the `load_train_data` method which loads in the available training data that will be used to initialize the optimization run. 

See example optimization class for molecule tasks: 
`scripts/molecule_optimization.py`



# For testing DCSVGP: 
CUDA_VISIBLE_DEVICES=0 python3 molecule_optimization.py --task_id logp --update_e2e False --max_n_oracle_calls 500 --bsz 10 --k 10 --track_with_wandb True --init_n_update_epochs 80 --num_update_epochs 5 --wandb_entity nmaus --surrogate_model_type DCSVGP --mll_type PPGPR - run_lolbo - done 

surrogate_model_type in ["DCSVGP", "ApproximateGP", "ApproximateGP_DKL"]
mll_type in ["ELBO", "PPGPR"]

# GAUSS NODE 1, uai0, uai1, ... 
conda activate lolbo_mols

# Variations: 

# NNSVGP
CUDA_VISIBLE_DEVICES=4 python3 molecule_optimization.py --task_id pdop --update_e2e False --max_n_oracle_calls 100000 --bsz 10 --k 10 --track_with_wandb True --init_n_update_epochs 60 --num_update_epochs 2 --wandb_entity nmaus --surrogate_model_type NNSVGP --mll_type ELBO --num_initialization_points 1024 - run_lolbo - done 
# PDOP x5 
# RANO 
# MED1 x1 

# DCSVGP_DKL_SHARED_Z
CUDA_VISIBLE_DEVICES=3 python3 molecule_optimization.py --task_id med1 --update_e2e False --max_n_oracle_calls 100000 --bsz 10 --k 10 --track_with_wandb True --init_n_update_epochs 60 --num_update_epochs 2 --wandb_entity nmaus --surrogate_model_type DCSVGP_DKL_SHARED_Z --mll_type ELBO --num_initialization_points 1024 --dc_shared_inducing_pts True - run_lolbo - done 
# med1 x 4

CUDA_VISIBLE_DEVICES=4 python3 molecule_optimization.py --task_id med1 --update_e2e False --max_n_oracle_calls 100000 --bsz 10 --k 10 --track_with_wandb True --init_n_update_epochs 60 --num_update_epochs 2 --wandb_entity nmaus --surrogate_model_type DCSVGP_DKL --mll_type ELBO --num_initialization_points 1024 --dc_shared_inducing_pts True - run_lolbo - done 
# med1 x 4

CUDA_VISIBLE_DEVICES=5 python3 molecule_optimization.py --task_id med1 --update_e2e False --max_n_oracle_calls 100000 --bsz 10 --k 10 --track_with_wandb True --init_n_update_epochs 60 --num_update_epochs 2 --wandb_entity nmaus --surrogate_model_type DCSVGP_DKL --mll_type ELBO --num_initialization_points 1024 - run_lolbo - done 
# med1 x 4

CUDA_VISIBLE_DEVICES=6 python3 molecule_optimization.py --task_id med1 --update_e2e False --max_n_oracle_calls 100000 --bsz 10 --k 10 --track_with_wandb True --init_n_update_epochs 60 --num_update_epochs 2 --wandb_entity nmaus --surrogate_model_type ApproximateGP_DKL --mll_type ELBO --num_initialization_points 1024 - run_lolbo - done 
# med1 x 4








CUDA_VISIBLE_DEVICES=6 python3 molecule_optimization.py --task_id rano --update_e2e False --max_n_oracle_calls 100000 --bsz 10 --k 10 --track_with_wandb True --init_n_update_epochs 60 --num_update_epochs 2 --wandb_entity nmaus --surrogate_model_type DCSVGP_DKL --mll_type PPGPR --num_initialization_points 1024 - run_lolbo - done 
# med1 x 1

CUDA_VISIBLE_DEVICES=0 python3 molecule_optimization.py --task_id zale --update_e2e False --max_n_oracle_calls 100000 --bsz 10 --k 10 --track_with_wandb True --init_n_update_epochs 60 --num_update_epochs 2 --wandb_entity nmaus --surrogate_model_type ApproximateGP_DKL --mll_type PPGPR --num_initialization_points 1024 - run_lolbo - done 
# PDOP x0 

# PDOP on both Gauss and Allegro 6,7! 
# RANO on LOCUST 0-3

# LOCUST:
docker run -v /home1/n/nmaus/constrained-lolbo/:/workspace/ --gpus all -it nmaus/robot

--dc_shared_inducing_pts True ... !!! 




# TURBO: 
docker run -v /home1/n/nmaus/constrained-lolbo/:/workspace/ --gpus all -it nmaus/robot
docker run -v /shared_data/constrained-lolbo/:/workspace/constrained-lolbo/ --gpus all -it nmaus/robot

CUDA_VISIBLE_DEVICES=7 python3 run_turbo.py --task_id lunar --min_seed 0 --max_seed 9 --surrogate_model_type DCSVGP --mll_type ELBO
CUDA_VISIBLE_DEVICES=6 python3 run_turbo.py --task_id lunar --min_seed 0 --max_seed 9 --surrogate_model_type ApproximateGP --mll_type ELBO
CUDA_VISIBLE_DEVICES=5 python3 run_turbo.py --task_id lunar --min_seed 0 --max_seed 9 --surrogate_model_type DCSVGP --mll_type PPGPR
CUDA_VISIBLE_DEVICES=4 python3 run_turbo.py --task_id lunar --min_seed 0 --max_seed 9 --surrogate_model_type ApproximateGP --mll_type PPGPR

## STOCK OPT: 
CUDA_VISIBLE_DEVICES=0 python3 run_turbo.py --task_id stocks --min_seed 0 --max_seed 2 --surrogate_model_type ApproximateGP_DKL --mll_type ELBO --max_n_calls 10000000 --bsz 50 

CUDA_VISIBLE_DEVICES=0 python3 run_turbo.py --task_id stocks --min_seed 0 --max_seed 2 --surrogate_model_type NNSVGP --mll_type ELBO --max_n_calls 10000000 --bsz 50 

CUDA_VISIBLE_DEVICES=0 python3 run_turbo.py --task_id stocks --min_seed 0 --max_seed 2 --surrogate_model_type DCSVGP_DKL --mll_type ELBO --max_n_calls 10000000 --bsz 50 

CUDA_VISIBLE_DEVICES=0 python3 run_turbo.py --task_id stocks --min_seed 0 --max_seed 2 --surrogate_model_type DCSVGP_DKL --mll_type ELBO --dc_shared_inducing_pts True --max_n_calls 10000000 --bsz 50 

CUDA_VISIBLE_DEVICES=0 python3 run_turbo.py --task_id stocks --min_seed 0 --max_seed 2 --surrogate_model_type DCSVGP_DKL_SHARED_Z --mll_type ELBO --max_n_calls 10000000 --bsz 50 
 
