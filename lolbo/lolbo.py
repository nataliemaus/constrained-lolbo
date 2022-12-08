import torch
import gpytorch
import math
from gpytorch.mlls import PredictiveLogLikelihood 
from lolbo.utils.bo_utils.turbo import TurboState, update_state, generate_batch
from lolbo.utils.utils import update_models_end_to_end, update_surr_model, update_constraint_surr_models
from lolbo.utils.bo_utils.ppgpr import GPModelDKL
import numpy as np


class LOLBOState:

    def __init__(
        self,
        objective,
        train_x,
        train_y,
        train_z,
        train_c=None,
        k=1_000,
        minimize=False,
        num_update_epochs=2,
        init_n_epochs=20,
        learning_rte=0.01,
        bsz=10,
        acq_func='ts',
        verbose=True,
    ):
        self.objective          = objective         # objective with vae for particular task
        self.train_x            = train_x           # initial train x data
        self.train_y            = train_y           # initial train y data
        self.train_z            = train_z           # initial train z data
        self.train_c            = train_c           # initial constraint values data
        self.minimize           = minimize          # if True we want to minimize the objective, otherwise we assume we want to maximize the objective
        self.k                  = k                 # track and update on top k scoring points found
        self.num_update_epochs  = num_update_epochs # num epochs update models
        self.init_n_epochs      = init_n_epochs     # num epochs train surr model on initial data
        self.learning_rte       = learning_rte      # lr to use for model updates
        self.bsz                = bsz               # acquisition batch size
        self.acq_func           = acq_func          # acquisition function (Expected Improvement (ei) or Thompson Sampling (ts))
        self.verbose            = verbose

        assert acq_func in ["ei", "ts"]
        if minimize:
            self.train_y = self.train_y * -1

        self.progress_fails_since_last_e2e = 0
        self.tot_num_e2e_updates = 0
        self.best_score_seen = torch.max(train_y)
        self.best_x_seen = train_x[torch.argmax(train_y.squeeze())]
        self.initial_model_training_complete = False # initial training of surrogate model uses all data for more epochs
        self.new_best_found = False

        self.initialize_top_k()
        self.initialize_surrogate_model()
        self.initialize_tr_state()
        self.initialize_xs_to_scores_dict()


    def initialize_xs_to_scores_dict(self,):
        # put initial xs and ys in dict to be tracked by objective
        init_xs_to_scores_dict = {}
        for idx, x in enumerate(self.train_x):
            init_xs_to_scores_dict[x] = self.train_y.squeeze()[idx].item()
        self.objective.xs_to_scores_dict = init_xs_to_scores_dict


    def initialize_top_k(self):
        ''' Initialize top k x, y, and zs'''
        # if we have constriants, the top k are those that meet constraints!
        if self.train_c is not None: 
            bool_arr = torch.all(self.train_c <= 0, dim=-1) # all constraint values <= 0
            vaid_train_y = self.train_y[bool_arr]
            valid_train_z = self.train_z[bool_arr]
            valid_train_x = np.array(self.train_x)[bool_arr]
        else:
            vaid_train_y = self.train_y
            valid_train_z = self.train_z
            valid_train_x = self.train_x
        # track top k scores found
        # import pdb 
        # pdb.set_trace() 
        self.top_k_scores, top_k_idxs = torch.topk(vaid_train_y.squeeze(), min(self.k, vaid_train_y.shape[0]))
        self.top_k_scores = self.top_k_scores.tolist() 
        top_k_idxs = top_k_idxs.tolist()
        self.top_k_xs = [valid_train_x[i] for i in top_k_idxs]
        self.top_k_zs = [valid_train_z[i].unsqueeze(-2) for i in top_k_idxs]


    def initialize_tr_state(self):
        if self.train_c is not None:  # if constrained 
            bool_arr = torch.all(self.train_c <= 0, dim=-1) # all constraint values <= 0
            vaid_train_y = self.train_y[bool_arr]
            valid_c_vals = self.train_c[bool_arr]
        else:
            vaid_train_y = self.train_y
            best_constraint_values = None
        
        if len(vaid_train_y) == 0:
            best_value = -torch.inf 
            if self.minimize:
                best_value = torch.inf
            if self.train_c is not None: 
                best_constraint_values = torch.ones(1,self.train_c.shape[1])*torch.inf
        else:
            best_value=torch.max(vaid_train_y).item()
            if self.train_c is not None: 
                best_constraint_values = valid_c_vals[torch.argmax(vaid_train_y)]
                if len(best_constraint_values.shape) == 1:
                    best_constraint_values = best_constraint_values.unsqueeze(-1) 
        # initialize turbo trust region state
        self.tr_state = TurboState( # initialize turbo state
            dim=self.train_z.shape[-1],
            batch_size=self.bsz, 
            best_value=best_value,
            best_constraint_values=best_constraint_values 
        )

        return self


    def initialize_constraint_surrogates(self):
        self.c_models = []
        self.c_mlls = []
        for i in range(self.train_c.shape[1]):
            likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda() 
            n_pts = min(self.train_z.shape[0], 1024)
            c_model = GPModelDKL(self.train_z[:n_pts, :].cuda(), likelihood=likelihood ).cuda()
            c_mll = PredictiveLogLikelihood(c_model.likelihood, c_model, num_data=self.train_z.size(-2))
            c_model = c_model.eval() 
            # c_model = self.model.cuda()
            self.c_models.append(c_model)
            self.c_mlls.append(c_mll)
        return self 


    def initialize_surrogate_model(self ):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda() 
        n_pts = min(self.train_z.shape[0], 1024)
        self.model = GPModelDKL(self.train_z[:n_pts, :].cuda(), likelihood=likelihood ).cuda()
        self.mll = PredictiveLogLikelihood(self.model.likelihood, self.model, num_data=self.train_z.size(-2))
        self.model = self.model.eval() 
        self.model = self.model.cuda()

        if self.train_c is not None:
            self.initialize_constraint_surrogates()

        return self


    def update_next(
        self,
        z_next_,
        y_next_,
        x_next_,
        c_next_=None,
        acquisition=False
    ):
        '''Add new points (z_next, y_next, x_next) to train data
            and update progress (top k scores found so far)
            and update trust region state
        '''
        
        if c_next_ is not None:
            if len(c_next_.shape) == 1:
                c_next_ = c_next_.unsqueeze(-1)
            valid_points = torch.all(c_next_ <= 0, dim=-1) # all constraint values <= 0
        else:
            valid_points = torch.tensor([True]*len(y_next_))
        z_next_ = z_next_.detach().cpu() 
        y_next_ = y_next_.detach().cpu()
        if len(y_next_.shape) > 1:
            y_next_ = y_next_.squeeze() 
        if len(z_next_.shape) == 1:
            z_next_ = z_next_.unsqueeze(0)
        progress = False
        for i, score in enumerate(y_next_):
            self.train_x.append(x_next_[i] )
            if valid_points[i]: # if y is valid according to constraints 
                if len(self.top_k_scores) < self.k: 
                    # if we don't yet have k top scores, add it to the list
                    self.top_k_scores.append(score.item())
                    self.top_k_xs.append(x_next_[i])
                    self.top_k_zs.append(z_next_[i].unsqueeze(-2))
                elif score.item() > min(self.top_k_scores) and (x_next_[i] not in self.top_k_xs):
                    # if the score is better than the worst score in the top k list, upate the list
                    min_score = min(self.top_k_scores)
                    min_idx = self.top_k_scores.index(min_score)
                    self.top_k_scores[min_idx] = score.item()
                    self.top_k_xs[min_idx] = x_next_[i]
                    self.top_k_zs[min_idx] = z_next_[i].unsqueeze(-2) # .cuda()
                #if we imporve
                if score.item() > self.best_score_seen:
                    self.progress_fails_since_last_e2e = 0
                    progress = True
                    self.best_score_seen = score.item() #update best
                    self.best_x_seen = x_next_[i]
                    self.new_best_found = True
        if (not progress) and acquisition: # if no progress msde, increment progress fails
            self.progress_fails_since_last_e2e += 1
        y_next_ = y_next_.unsqueeze(-1)
        if acquisition:
            self.tr_state = update_state(
                state=self.tr_state,
                Y_next=y_next_,
                C_next=c_next_,
            )
        self.train_z = torch.cat((self.train_z, z_next_), dim=-2)
        self.train_y = torch.cat((self.train_y, y_next_), dim=-2)
        if c_next_ is not None:
            self.train_c = torch.cat((self.train_c, c_next_), dim=-2)

        return self


    def update_surrogate_model(self): 
        if not self.initial_model_training_complete:
            # first time training surr model --> train on all data
            n_epochs = self.init_n_epochs
            train_z = self.train_z
            train_y = self.train_y.squeeze(-1)
            train_c = self.train_c
        else:
            # otherwise, only train on most recent batch of data
            n_epochs = self.num_update_epochs
            train_z = self.train_z[-self.bsz:]
            train_y = self.train_y[-self.bsz:].squeeze(-1)
            if self.train_c is not None:
                train_c = self.train_c[-self.bsz:]
            else:
                train_c = None 
          
        self.model = update_surr_model(
            self.model,
            self.mll,
            self.learning_rte,
            train_z,
            train_y,
            n_epochs
        )
        if self.train_c is not None:
            self.c_models = update_constraint_surr_models(
                self.c_models,
                self.c_mlls,
                self.learning_rte,
                train_z,
                train_c,
                n_epochs
            )

        self.initial_model_training_complete = True

        return self


    def update_models_e2e(self):
        '''Finetune VAE end to end with surrogate model'''
        self.progress_fails_since_last_e2e = 0
        new_xs = self.train_x[-self.bsz:]
        new_ys = self.train_y[-self.bsz:].squeeze(-1).tolist()
        train_x = new_xs + self.top_k_xs
        train_y = torch.tensor(new_ys + self.top_k_scores).float()
        self.objective, self.model = update_models_end_to_end(
            train_x,
            train_y,
            self.objective,
            self.model,
            self.mll,
            self.learning_rte,
            self.num_update_epochs
        )
        self.tot_num_e2e_updates += 1

        return self


    def recenter(self):
        '''Pass SELFIES strings back through
            VAE to find new locations in the
            new fine-tuned latent space
        '''
        self.objective.vae.eval()
        self.model.train()
        optimizer1 = torch.optim.Adam([{'params': self.model.parameters(),'lr': self.learning_rte} ], lr=self.learning_rte)
        if self.train_c is not None: # if constrainted, update constraint models as well
            c_model_params = [{'params': c_model.parameters(),'lr': self.learning_rte} for c_model in self.c_models]
            params_list = c_model_params + [{'params': self.model.parameters(),'lr': self.learning_rte}]
            optimizer1 = torch.optim.Adam(params_list, lr=self.learning_rte)
        new_xs = self.train_x[-self.bsz:]
        train_x = new_xs + self.top_k_xs
        max_string_len = len(max(train_x, key=len))
        # max batch size smaller to avoid memory limit 
        #   with longer strings (more tokens) 
        bsz = max(1, int(2560/max_string_len))
        num_batches = math.ceil(len(train_x) / bsz) 
        for _ in range(self.num_update_epochs):
            for batch_ix in range(num_batches):
                start_idx, stop_idx = batch_ix*bsz, (batch_ix+1)*bsz
                batch_list = train_x[start_idx:stop_idx] 
                z, _ = self.objective.vae_forward(batch_list)
                out_dict = self.objective(z)
                scores_arr = out_dict['scores'] 
                constraints_tensor = out_dict['constr_vals']
                valid_zs = out_dict['valid_zs']
                xs_list = out_dict['decoded_xs']
                if len(scores_arr) > 0: # if some valid scores
                    scores_arr = torch.from_numpy(scores_arr)
                    if self.minimize:
                        scores_arr = scores_arr * -1
                    pred = self.model(valid_zs)
                    loss = -self.mll(pred, scores_arr.cuda())
                    c_loss = 0
                    if self.train_c is not None: 
                        for ix, c_model in enumerate(self.c_models):
                            pred = c_model(valid_zs)
                            c_loss += -self.c_mlls[ix](pred, constraints_tensor[:,ix].cuda())
                    loss = loss + c_loss 
                    optimizer1.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer1.step() 
                    with torch.no_grad(): 
                        z = z.detach().cpu()
                        self.update_next(
                            z,
                            scores_arr,
                            xs_list,
                            c_next_=constraints_tensor
                        )
            torch.cuda.empty_cache()
        self.model.eval() 

        return self


    def acquisition(self):
        '''Generate new candidate points, 
        evaluate them, and update data
        '''
        # 1. Generate a batch of candidates in 
        #   trust region using surrogate model
        if self.train_c is not None: # if constrained 
            constraint_model_list=self.c_models
        else:
            constraint_model_list = None 
        z_next = generate_batch(
            state=self.tr_state,
            model=self.model,
            X=self.train_z,
            Y=self.train_y,
            batch_size=self.bsz, 
            acqf=self.acq_func,
            constraint_model_list=constraint_model_list,
        )
        # 2. Evaluate the batch of candidates by calling oracle
        with torch.no_grad():
            out_dict = self.objective(z_next)
            z_next = out_dict['valid_zs']
            y_next = out_dict['scores']
            x_next = out_dict['decoded_xs']     
            c_next = out_dict['constr_vals']  
            if self.minimize:
                y_next = y_next * -1
        # 3. Add new evaluated points to dataset (update_next)
        if len(y_next) != 0:
            y_next = torch.from_numpy(y_next).float()
            self.update_next(
                z_next,
                y_next,
                x_next,
                c_next,
                acquisition=True
            )
        else:
            self.progress_fails_since_last_e2e += 1
            if self.verbose:
                print("GOT NO VALID Y_NEXT TO UPDATE DATA, RERUNNING ACQUISITOIN...")


