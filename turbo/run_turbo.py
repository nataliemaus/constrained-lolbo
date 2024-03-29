
import torch
import gpytorch
import numpy as np 
import copy 
import sys 
sys.path.append("../") 
from turbo.trust_region import (
    TrustRegionState, 
    generate_batch, 
    update_state
)
from turbo.tasks.rover.rover_objective import RoverObjective
from turbo.tasks.lunar_lander.lunarlunar_lander_objective import LunarLanderObjective
from turbo.tasks.stocks.stocks_objective import StocksObjective
from turbo.bo_utils.ppgpr import (
    GPModelDKL,
)
from turbo.bo_utils.dcsvgp import (
    DCSVGP,
    BaselineSVGP,
    DCSVGP_V2,
)
from lolbo.utils.bo_utils.dcsvgp_dkl.dcsvgp_with_deep_kernel import (
    DCSVGP_DKL,
    DCSVGP_DKL_SHARED_Z
)
from lolbo.utils.bo_utils.dcsvgp_dkl.nnsvgp import NNSVGP

from torch.utils.data import (
    TensorDataset, 
    DataLoader
)
import argparse 
import wandb 
import math 
import os 
os.environ["WANDB_SILENT"] = "true" 
import random 
import pandas as pd 
# from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
# Xinran's branch version that takes in kwargs for dcsvgp_v2: 
from bo_utils.variational_elbo import VariationalELBO
from bo_utils.predictive_log_likelihood import PredictiveLogLikelihood

class RunTurbo():
    def __init__(self, args):
        self.args = args 
    
    def initialize_surrogate_model(self ):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda() 
        if self.args.surrogate_model_type == "DCSVGP":
            self.model = DCSVGP(self.train_x.cuda() ).cuda() 
        elif self.args.surrogate_model_type == "DCSVGP_V2":
            self.model = DCSVGP_V2(self.train_x.cuda() ).cuda() 
        elif self.args.surrogate_model_type == "ApproximateGP":
            self.model = BaselineSVGP(self.train_x.cuda() ).cuda() 
        elif self.args.surrogate_model_type == "ApproximateGP_DKL": # (DEFAULT)
            self.model = GPModelDKL(
                inducing_points=self.train_x.cuda(), 
                likelihood=likelihood,
                hidden_dims=(128, 128),
            ).cuda() 
        elif self.args.surrogate_model_type == "DCSVGP_DKL":
            self.model = DCSVGP_DKL(
                inducing_points=self.train_x.cuda(), 
                likelihood=likelihood,
                hidden_dims=(128, 128), 
                shared_inducing_pts=self.args.dc_shared_inducing_pts,
            ).cuda()  
        elif self.args.surrogate_model_type == "DCSVGP_DKL_SHARED_Z":
            self.model = DCSVGP_DKL_SHARED_Z( 
                inducing_points=self.train_x.cuda(), 
                likelihood=likelihood,
                hidden_dims=(128, 128), 
            ).cuda()
        elif self.args.surrogate_model_type == "NNSVGP":
            self.model = NNSVGP( 
                inducing_points=self.train_x.cuda(), 
            ).cuda()
        else:
            assert("Invalid surrogate model type")

        if self.args.mll_type == "ELBO": # Standard SVGP
            self.mll = VariationalELBO(self.model.likelihood, self.model, num_data=self.train_x.size(-2))
        elif self.args.mll_type == "PPGPR": # PPGPR (DEFAULT)
            self.mll = PredictiveLogLikelihood(self.model.likelihood, self.model, num_data=self.train_x.size(-2))
        else:
            assert("Invalid mll type")

        self.model = self.model.eval() 
        self.model = self.model.cuda()

    def start_wandb(self):
        args_dict = vars(self.args) 
        self.tracker = wandb.init(
            entity=args_dict['wandb_entity'], 
            project=args_dict['wandb_project_name'],
            config=args_dict, 
        ) 
        print('running', wandb.run.name) 

    def set_seed(self):
        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = False
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = False
        seed = self.args.seed  
        if seed is not None:
            torch.manual_seed(seed) 
            random.seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            os.environ["PYTHONHASHSEED"] = str(seed)

    def update_surr_model(
        self,
        n_epochs
    ):
        self.model = self.model.train() 
        optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.args.lr} ], lr=self.args.lr)
        train_bsz = min(len(self.train_y),128)
        train_dataset = TensorDataset(self.train_x.cuda(), self.train_y.cuda())
        train_loader = DataLoader(train_dataset, batch_size=train_bsz, shuffle=True)
        if self.args.surrogate_model_type == "DCSVGP_V2": 
            kwargs = {'beta1': self.args.beta1, "beta2": self.args.beta2}
        for _ in range(n_epochs):
            for (inputs, scores) in train_loader:
                optimizer.zero_grad()
                output = self.model(inputs.cuda()) 
                if self.args.surrogate_model_type == "DCSVGP_V2": 
                    kwargs["x"] = inputs.cuda()
                    loss = -self.mll(output, scores.cuda(), **kwargs).sum() 
                else:
                    loss = -self.mll(output, scores.cuda()).sum() 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
        self.model = self.model.eval()


    def sample_random_searchspace_points(self, N):
        lb, ub = self.objective.lb, self.objective.ub 
        if ub is None: ub = self.search_space_data().max() 
        if lb is None: lb = self.search_space_data().max() 
        random_xs = torch.rand(N, self.objective.dim)*(ub - lb) + lb

        return random_xs

    def get_init_data(self ):
        # then get initialization prompts + scores ... 
        init_points = self.sample_random_searchspace_points(self.args.num_init_data_pts)
        out_dict = self.objective(init_points)
        self.train_x = torch.from_numpy(out_dict["valid_xs"])
        self.train_y = torch.tensor(out_dict["scores"]).unsqueeze(-1)

    def init_objective(self):
        if self.args.task_id == "rover":
            self.objective = RoverObjective()
        elif self.args.task_id == "lunar":
            self.objective = LunarLanderObjective()
        elif self.args.task_id == "stocks":
            self.objective = StocksObjective()
        else:
            assert 0 

    def run(self):
        self.set_seed()
        self.start_wandb() # initialized self.tracker
        self.init_objective() 
        self.get_init_data() 
        self.initialize_surrogate_model() 
        self.update_surr_model(self.args.init_n_epochs )
        prev_best = -torch.inf 
        num_tr_restarts = 0  
        tr = TrustRegionState(
            dim=self.objective.dim,
            failure_tolerance=self.args.failure_tolerance,
            success_tolerance=self.args.success_tolerance,
        )
        n_iters = 0
        n_calls_without_progress = 0
        while self.objective.num_calls < self.args.max_n_calls:
            self.tracker.log({
                "num_calls":self.objective.num_calls,
                "best_y":self.train_y.max().item(),
                'tr_length':tr.length,
                "tr_success_counter":tr.success_counter,
                "tr_failure_counter":tr.failure_counter,
                "num_tr_restarts":num_tr_restarts,
            } ) 
            if self.train_y.max().item() > prev_best: 
                n_calls_without_progress = 0
                prev_best = self.train_y.max().item() 
            else:
                n_calls_without_progress += self.args.bsz 
            if n_calls_without_progress > self.args.max_allowed_calls_without_progresss:
                break
            x_next = generate_batch( 
                state=tr,
                model=self.model,
                X=self.train_x,
                Y=self.train_y,
                batch_size=self.args.bsz, 
                acqf=self.args.acq_func,
                absolute_bounds=(self.objective.lb, self.objective.ub)
            ) 
            out_dict = self.objective(x_next)
            self.train_x = torch.cat((self.train_x, torch.from_numpy(out_dict['valid_xs']).detach().cpu()), dim=-2)
            y_next = torch.tensor(out_dict['scores']).unsqueeze(-1)
            self.train_y = torch.cat((self.train_y, y_next.detach().cpu()), dim=-2) 
            tr = update_state(tr, y_next) 
            if tr.restart_triggered:
                num_tr_restarts += 1
                tr = TrustRegionState(
                    dim=self.objective.dim,
                    failure_tolerance=self.args.failure_tolerance,
                    success_tolerance=self.args.success_tolerance,
                )
                self.initialize_surrogate_model() 
                self.update_surr_model(self.args.init_n_epochs)
            # flag_reset_gp_new_data 
            elif (self.train_x.shape[0] < 1024) and (n_iters % 10 == 0): # reestart gp and update on all data 
                self.initialize_surrogate_model() 
                self.update_surr_model(self.args.init_n_epochs)
            else:
                self.update_surr_model(self.args.n_epochs)
            n_iters += 1
        self.tracker.finish() 


# def tuple_type(strings):
#     strings = strings.replace("(", "").replace(")", "")
#     mapped_int = map(int, strings.split(","))
#     return tuple(mapped_int)


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    # parser.add_argument('--hidden_dims', type=tuple_type, default="(256,128,64)") 
    parser.add_argument('--work_dir', default='/home/nmaus/' ) 
    parser.add_argument('--wandb_entity', default="nmaus" ) 
    parser.add_argument('--wandb_project_name', default="turbo" )
    parser.add_argument('--failure_tolerance', type=int, default=32 )  
    parser.add_argument('--success_tolerance', type=int, default=10 )  
    parser.add_argument('--lr', type=float, default=0.001 ) 
    parser.add_argument('--n_epochs', type=int, default=2)  
    parser.add_argument('--init_n_epochs', type=int, default=60) 
    parser.add_argument('--acq_func', type=str, default='ts' )
    parser.add_argument('--bsz', type=int, default=10)  
    parser.add_argument('--max_allowed_calls_without_progresss', type=int, default=100_000 )
    # parser.add_argument('--minimize', type=bool, default=True) 
    # maybe
    parser.add_argument('--num_init_data_pts', type=int, default=1024 ) 
    parser.add_argument('--task_id', default="rover" ) 
    parser.add_argument('--max_n_calls', type=int, default=5_000)
    parser.add_argument('--seed', type=int, default=0 ) 
    parser.add_argument('--min_seed', type=int, default=3 ) 
    parser.add_argument('--max_seed', type=int, default=10 ) 
    parser.add_argument('--beta1', type=float, default=1.0 ) 
    parser.add_argument('--beta2', type=float, default=1.0 ) 
    # often 
    parser.add_argument('--surrogate_model_type', default="ApproximateGP_DKL" ) 
    parser.add_argument('--mll_type', default="PPGPR" ) 
    parser.add_argument('--dc_shared_inducing_pts', type=bool, default=False)
    og_args = parser.parse_args() 
    # assert og_args.surrogate_model_type in ["DCSVGP", "ApproximateGP", "ApproximateGP_DKL"]
    assert og_args.mll_type in ["ELBO", "PPGPR"]

    args = copy.deepcopy(og_args)
    for s in range(args.min_seed, args.max_seed + 1):
        args.seed = s
        runner = RunTurbo(args) 
        runner.run()

# cd turbo 
# CUDA_VISIBLE_DEVICES=2 python3 run_turbo.py --surrogate_model_type DCSVGP --mll_type PPGPR
# CUDA_VISIBLE_DEVICES=3 python3 run_turbo.py --surrogate_model_type DCSVGP --mll_type ELBO
# CUDA_VISIBLE_DEVICES=4 python3 run_turbo.py --surrogate_model_type ApproximateGP --mll_type PPGPR
# CUDA_VISIBLE_DEVICES=5 python3 run_turbo.py --surrogate_model_type ApproximateGP --mll_type ELBO

# running now 0-10, 10-20, 20-30 :) 

# docker pull nmaus/robot 
# docker run -v /shared_data/constrained-lolbo:/workspace/constrained-lolbo --gpus all -it nmaus/robot

# Eric 6000
# docker run -v /home1/n/nmaus/constrained-lolbo/:/workspace/ --gpus all -it nmaus/robot

# CUDA_VISIBLE_DEVICES=7 python3 run_turbo.py --task_id lunar --min_seed 0 --max_seed 9 --surrogate_model_type DCSVGP --mll_type ELBO
# CUDA_VISIBLE_DEVICES=6 python3 run_turbo.py --task_id lunar --min_seed 0 --max_seed 9 --surrogate_model_type ApproximateGP --mll_type ELBO
# CUDA_VISIBLE_DEVICES=5 python3 run_turbo.py --task_id lunar --min_seed 0 --max_seed 9 --surrogate_model_type DCSVGP --mll_type PPGPR
# CUDA_VISIBLE_DEVICES=4 python3 run_turbo.py --task_id lunar --min_seed 0 --max_seed 9 --surrogate_model_type ApproximateGP --mll_type PPGPR

