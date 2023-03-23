# okay 
import torch 
import numpy as np
import sys 
sys.path.append("../")
import torch 
import selfies as sf 
import wandb  
import argparse 
from lolbo.utils.mol_utils.selfies_vae.model_positional_unbounded import SELFIESDataset, InfoTransformerVAE
from lolbo.utils.mol_utils.selfies_vae.data import collate_fn
import math 
from lolbo.utils.mol_utils.load_data import load_molecule_train_data
from lolbo.utils.mol_utils.mol_utils import smile_to_guacamole_score
import random 

class Runner():
    def __init__(
        self,
        max_string_length=1000,
        path_to_vae_statedict="../lolbo/utils/mol_utils/selfies_vae/state_dict/SELFIES-VAE-state-dict.pt",
        learning_rte=0.001,
        num_update_epochs=2,
        num_initialization_points=10_000,
        task_id="rano",
        max_num_calls=200_000,
        wandb_project_name="lolbo-variant-1",
        wandb_entity="nmaus",
        top_perc=0.1,
        k=1_000,
    ):
        self.top_perc=top_perc
        self.dim=256 
        self.xs_to_scores_dict = {}
        self.num_calls = 0
        self.best_found = -torch.inf
        self.max_string_length = max_string_length
        self.path_to_vae_statedict = path_to_vae_statedict
        self.learning_rte = learning_rte
        self.num_update_epochs = num_update_epochs
        self.num_initialization_points = num_initialization_points
        self.task_id = task_id
        self.max_num_calls = max_num_calls
        self.wandb_project_name=wandb_project_name
        self.wandb_entity=wandb_entity
        self.k=k
        self.load_train_data()
        self.initialize_vae() 
        self.create_wandb_tracker() 

    def create_wandb_tracker(self):
        config_dict = { 
            "task_id":self.task_id,
            "max_string_length":self.max_string_length,
            "learning_rte":self.learning_rte,
            "num_update_epochs":self.num_update_epochs,
            "num_initialization_points":self.num_initialization_points,
            "max_num_calls":self.max_num_calls,
            "top_perc":self.top_perc,
            "k":self.k,
        }
        config_dict = {} 
        self.tracker = wandb.init(
            project=self.wandb_project_name,
            entity=self.wandb_entity,
            config=config_dict,
        ) 
        self.wandb_run_name = wandb.run.name


    def main_loop(self):
        prev_best_y = -torch.inf 
        while self.num_calls < self.max_num_calls:
            # update best found 
            self.best_y = self.train_y.squeeze().max().item()  
            self.best_smile = self.train_x[self.train_y.squeeze().argmax()]
            self.tracker.log({
                "num_calls":self.num_calls,
                "best_y":self.best_y,
                "best_smile":self.best_smile,
            })
            if self.best_y > prev_best_y:
                print(f"New best:{self.best_y}, Num calls:{self.num_calls}")
                prev_best_y = self.best_y 
            # pass in top 10%
            # compute scores and update data 
            # Update VAE on high scoring ones only 
            self.update_models_end_to_end()
            


    def update_models_end_to_end(
        self,
    ):
        '''Finetune VAE end to end with surrogate model
        This method is build to be compatible with the 
        SELFIES VAE interface
        '''
        self.vae.train()
        optimizer = torch.optim.Adam([
                {'params': self.vae.parameters()}, ], lr=self.learning_rte)
        # max batch size smaller to avoid memory limit with longer strings (more tokens)
        top_ten_perc_threshold = torch.quantile(self.train_y, 1-self.top_perc, dim=0, keepdim=True)
        train_x = (self.train_x[(self.train_y.squeeze() >= top_ten_perc_threshold).tolist()]).tolist()
        train_x = random.sample(train_x, self.k) + [self.best_smile] # only train on random k form top perc plus best smile seen
        max_string_length = len(max(train_x, key=len))
        bsz = max(1, int(2560/max_string_length)) 
        num_batches = math.ceil(len(train_x) / bsz)
        for _ in range(self.num_update_epochs):
            for batch_ix in range(num_batches):
                start_idx, stop_idx = batch_ix*bsz, (batch_ix+1)*bsz
                batch_list = train_x[start_idx:stop_idx]
                z, _ = self.vae_forward(batch_list)
                decoded_smiles = self.vae_decode(z) 
                update_on_smiles = [] 
                for ix, smile in enumerate(decoded_smiles):
                    if not smile in self.xs_to_scores_dict:
                        score = smile_to_guacamole_score(self.task_id, smile)
                        if score is not None:
                            self.xs_to_scores_dict[smile] = score
                            self.num_calls += 1
                            self.train_x = np.array(self.train_x.tolist() + [smile])
                            self.train_y = torch.tensor(self.train_y.squeeze().tolist() + [score]).unsqueeze(-1)
                            if (score.item() >= top_ten_perc_threshold)[0].item():
                                update_on_smiles.append(smile) 
                if len(update_on_smiles) > 0:
                    _, loss = self.vae_forward(update_on_smiles) 
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                    optimizer.step()
        self.vae.eval()
    

    def initialize_vae(self):
        ''' Sets self.vae to the desired pretrained vae and 
            sets self.dataobj to the corresponding data class 
            used to tokenize inputs, etc. '''
        self.dataobj = SELFIESDataset()
        self.vae = InfoTransformerVAE(dataset=self.dataobj)
        # load in state dict of trained model:
        if self.path_to_vae_statedict:
            state_dict = torch.load(self.path_to_vae_statedict) 
            self.vae.load_state_dict(state_dict, strict=True) 
        self.vae = self.vae.cuda()
        self.vae = self.vae.eval()
        # set max string length that VAE can generate
        self.vae.max_string_length = self.max_string_length

    def vae_forward(self, xs_batch):
        ''' Input: 
                a list xs 
            Output: 
                z: tensor of resultant latent space codes 
                    obtained by passing the xs through the encoder
                vae_loss: the total loss of a full forward pass
                    of the batch of xs through the vae 
                    (ie reconstruction error)
        '''
        # assumes xs_batch is a batch of smiles strings 
        X_list = []
        for smile in xs_batch:
            try:
                # avoid re-computing mapping from smiles to selfies to save time
                selfie = self.smiles_to_selfies[smile]
            except:
                selfie = sf.encoder(smile)
                self.smiles_to_selfies[smile] = selfie
            tokenized_selfie = self.dataobj.tokenize_selfies([selfie])[0]
            encoded_selfie = self.dataobj.encode(tokenized_selfie).unsqueeze(0)
            X_list.append(encoded_selfie)
        X = collate_fn(X_list)
        dict = self.vae(X.cuda())
        vae_loss, z = dict['loss'], dict['z']
        z = z.reshape(-1,self.dim)

        return z, vae_loss


    def vae_decode(self, z):
        '''Input
                z: a tensor latent space points
            Output
                a corresponding list of the decoded input space 
                items output by vae decoder 
        '''
        if type(z) is np.ndarray: 
            z = torch.from_numpy(z).float()
        z = z.cuda()
        self.vae = self.vae.eval()
        self.vae = self.vae.cuda()
        # sample molecular string form VAE decoder
        sample = self.vae.sample(z=z.reshape(-1, 2, 128))
        # grab decoded selfies strings
        decoded_selfies = [self.dataobj.decode(sample[i]) for i in range(sample.size(-2))]
        # decode selfies strings to smiles strings (SMILES is needed format for oracle)
        decoded_smiles = []
        for selfie in decoded_selfies:
            smile = sf.decoder(selfie)
            decoded_smiles.append(smile)
            # save smile to selfie mapping to map back later if needed
            self.smiles_to_selfies[smile] = selfie

        return decoded_smiles


    def load_train_data(self):
        ''' Load in or randomly initialize self.num_initialization_points
            total initial data points to kick-off optimization 
            Must define the following:
                self.init_train_x (a list of x's)
                self.init_train_y (a tensor of scores/y's)
                self.init_train_z (a tensor of corresponding latent space points)
            '''
        assert self.num_initialization_points <= 20_000 
        smiles, selfies, zs, ys = load_molecule_train_data(
            task_id=self.task_id,
            num_initialization_points=self.num_initialization_points,
            path_to_vae_statedict=self.path_to_vae_statedict
        )
        self.train_x, self.train_z, self.train_y = smiles, zs, ys
        self.train_x = np.array(self.train_x)

        # create initial smiles to selfies dict
        self.smiles_to_selfies = {}
        for ix, smile in enumerate(self.train_x):
            self.smiles_to_selfies[smile] = selfies[ix]
        
        # if self.train_z is None:
        #     self.init_train_z = compute_train_zs(
        #         self,
        #         self.train_x,
        #     )

    
if __name__=="__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--max_string_length', type=int, default=1000) 
    parser.add_argument('--learning_rte', type=float, default=0.001 )
    parser.add_argument('--num_update_epochs', type=int, default=2) 
    parser.add_argument('--num_initialization_points', type=int, default=10_000)
    parser.add_argument('--task_id', default="valt")  
    parser.add_argument('--max_num_calls', type=int, default=200_000) 
    parser.add_argument('--top_perc', type=float, default=0.01 )
    parser.add_argument('--k', type=float, default=100 )
    args = parser.parse_args() 
    runner = Runner(
        max_string_length=args.max_string_length,
        path_to_vae_statedict="../lolbo/utils/mol_utils/selfies_vae/state_dict/SELFIES-VAE-state-dict.pt",
        learning_rte=args.learning_rte,
        num_update_epochs=args.num_update_epochs,
        num_initialization_points=args.num_initialization_points,
        task_id=args.task_id,
        max_num_calls=args.max_num_calls,
        wandb_project_name="lolbo-variant-1",
        wandb_entity="nmaus",
        top_perc=args.top_perc,
        k=args.k,
    )
    runner.main_loop()
    # CUDA_VISIBLE_DEVICES=0 python3 test1.py
    # CUDA_VISIBLE_DEVICES=1 python3 test1.py

