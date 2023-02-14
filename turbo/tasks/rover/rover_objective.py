import torch 
from turbo.objective import Objective
from gpytorch.kernels.kernel import Distance
from turbo.tasks.rover.rover_utils import create_large_domain, ConstantOffsetFn


class RoverObjective(Objective):
    ''' Rover optimization task
        Goal is to find a policy for the Rover which
        results in a trajectory that moves the rover from
        start point to end point while avoiding the obstacles,
        thereby maximizing reward 
    ''' 
    def __init__(
        self,
        xs_to_scores_dict={},
        num_calls=0,
        dim=60,
        **kwargs,
    ):
        assert dim % 2 == 0
        lb = -0.5 * 4 / dim 
        ub = 4 / dim 

        # create rover domain 
        self.domain = create_large_domain(n_points=dim // 2)
        # create rover oracle 
        f_max=5.0 # default
        self.oracle = ConstantOffsetFn(self.domain, f_max)
        # create distance module for divf
        self.dist_module = Distance()
        # create dict to hold mapping from points to trajectories 
        self.xs_to_trajectories_dict = {}
        # rover oracle needs torch.double datatype 
        self.tkwargs={"dtype": torch.double}

        super().__init__(
            xs_to_scores_dict=xs_to_scores_dict,
            num_calls=num_calls,
            task_id='rover',
            dim=dim,
            lb=lb,
            ub=ub,
            **kwargs,
        ) 


    def query_oracle(self, x):
        reward = torch.tensor(self.oracle(x.cpu().numpy())).to(**self.tkwargs) # .unsqueeze(-1)
        return reward 
    
    # def get_trajectory(self, x):
    #     try:
    #         trajectory = self.xs_to_trajectories_dict[x]
    #     except:
    #         trajectory = torch.from_numpy(self.domain.trajectory(x.cpu().numpy())).to(**self.tkwargs)
    #         self.xs_to_trajectories_dict[x] = trajectory

    #     return trajectory
