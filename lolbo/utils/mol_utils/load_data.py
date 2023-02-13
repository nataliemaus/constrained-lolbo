import numpy as np
import pandas as pd
import argparse 
try:
    import torch
except:
    pass
import math

def load_molecule_train_data(
    task_id,
    path_to_vae_statedict,
    num_initialization_points=10_000,
): 
    df = pd.read_csv("../lolbo/utils/mol_utils/guacamol_data/guacamol_train_data_first_20k.csv")
    # df = df[0:num_initialization_points]
    train_x_smiles = df['smile'].values.tolist()
    train_x_selfies = df['selfie'].values.tolist() 
    if "drd3" in task_id: # drd3 docking, w/ dockstring or tdc! 
        if task_id == "dock_drd3": # dockstring 
            df = pd.read_csv("../lolbo/utils/mol_utils/drd3_scores.csv", header=None)
        elif task_id == "tdc_drd3":  # tdc! 
            df = pd.read_csv("../lolbo/utils/mol_utils/tdc_drd3_scores.csv", header=None)
        train_y = torch.from_numpy(df.values).float().squeeze() 
        train_y = torch.where(torch.isfinite(train_y), train_y, torch.nan) # remove inf values ... 
        # can only use ass many smiles/selfies as we have computed so far
        train_x_smiles = np.array(train_x_smiles[0:train_y.shape[0]])
        train_x_selfies = np.array(train_x_selfies[0:train_y.shape[0]])
        train_y = train_y[0:train_x_selfies.shape[0]]
        # get rid of invalid (nan) initial points 
        bool_arr = torch.logical_not(torch.isnan(train_y))
        train_y = train_y[bool_arr]
        train_x_smiles = train_x_smiles[bool_arr].tolist() 
        train_x_selfies = train_x_selfies[bool_arr].tolist() 
        num_initialization_points = min(train_y.shape[0], num_initialization_points)
        train_z = None 
    else:
        train_y = torch.from_numpy(df[task_id].values).float() 
        train_z = load_train_z(
            num_initialization_points=num_initialization_points,
            path_to_vae_statedict=path_to_vae_statedict
        )
    
    train_x_selfies = train_x_selfies[0:num_initialization_points]
    train_x_smiles = train_x_smiles[0:num_initialization_points]
    train_y = train_y[0:num_initialization_points]
    train_y = train_y.unsqueeze(-1)

    return train_x_smiles, train_x_selfies, train_z, train_y


def load_train_z(
    num_initialization_points,
    path_to_vae_statedict,
):
    state_dict_file_type = path_to_vae_statedict.split('.')[-1] # usually .pt or .ckpt
    path_to_init_train_zs = path_to_vae_statedict.replace(f".{state_dict_file_type}", '-train-zs.csv')
    # if we have a path to pre-computed train zs for vae, load them
    try:
        zs = pd.read_csv(path_to_init_train_zs, header=None).values
        # make sure we have a sufficient number of saved train zs
        assert len(zs) >= num_initialization_points
        zs = zs[0:num_initialization_points]
        zs = torch.from_numpy(zs).float()
    # otherwisee, set zs to None 
    except: 
        zs = None 

    return zs


def compute_train_zs(
    mol_objective,
    init_train_x,
    bsz=64
):
    init_zs = []
    # make sure vae is in eval mode 
    mol_objective.vae.eval() 
    n_batches = math.ceil(len(init_train_x)/bsz)
    for i in range(n_batches):
        xs_batch = init_train_x[i*bsz:(i+1)*bsz] 
        zs, _ = mol_objective.vae_forward(xs_batch)
        init_zs.append(zs.detach().cpu())
    init_zs = torch.cat(init_zs, dim=0)
    # now save the zs so we don't have to recompute them in the future:
    state_dict_file_type = mol_objective.path_to_vae_statedict.split('.')[-1] # usually .pt or .ckpt
    path_to_init_train_zs = mol_objective.path_to_vae_statedict.replace(f".{state_dict_file_type}", '-train-zs.csv')
    zs_arr = init_zs.cpu().detach().numpy()
    pd.DataFrame(zs_arr).to_csv(path_to_init_train_zs, header=None, index=None) 

    return init_zs


def save_condensed_tdc():
    import glob 
    paths = glob.glob(f"tdc_drd3_scores_*.csv")
    scores_computed = [np.nan]*20_000 
    for path in paths:
        idx = int(path.split("_")[-1].split("to")[0])
        data = pd.read_csv(path, header=None).values.squeeze()
        if len(data.shape) > 0: # if non-empty 
            for value in data:
                scores_computed[idx] = value
                idx += 1 
    score_arr = np.array(scores_computed)
    final_save_path = f"tdc_drd3_scores.csv"
    pd.DataFrame(score_arr).to_csv(final_save_path, header=None, index=None)

def save_tdc(min_ix=0, max_ix=None):
    # from dockstring import load_target 
    if max_ix is None:
        max_ix = 20_000 
    from tdc import Oracle
    oracle = Oracle('3pbl_docking')
    df = pd.read_csv("guacamol_data/guacamol_train_data_first_20k.csv")
    train_x_smiles = df['smile'].values.tolist()
    train_x_smiles = train_x_smiles[min_ix:max_ix]
    save_path = f"tdc_drd3_scores_{min_ix}to{max_ix}.csv"
    # try:
    #     # load already saved score 
    #     df1 = pd.read_csv("tdc_drd3_scores.csv", header=None)
    #     scores = df1.values.squeeze().tolist() 
    #     train_x_smiles = train_x_smiles[len(scores):]
    # except:
    #     print("no scores presaved")
    scores = [] 
    max_score = -100_000_000 
    for ix, smile in enumerate(train_x_smiles):
        try:
            score_ = oracle(smile)
            score_ = score_ * -1 # minimization! 
            if score_ > max_score:
                max_score = score_ 
        except:
            score_ = np.nan 
        scores.append(score_)
        print("max score:", max_score) 
        if ix % 10 == 0:
            score_arr = np.array(scores)
            pd.DataFrame(score_arr).to_csv(save_path, header=None, index=None)

    score_arr = np.array(scores)
    pd.DataFrame(score_arr).to_csv(save_path, header=None, index=None)
    # conda activate dockstring 
    # conda activate dock2 

def save_dockstring():
    from dockstring import load_target
    drd3_target = load_target("DRD3")
    df = pd.read_csv("guacamol_data/guacamol_train_data_first_20k.csv")
    train_x_smiles = df['smile'].values.tolist()
    scores = []
    max_score = -100_000_000 
    for ix, smile in enumerate(train_x_smiles):
        try:
            score_, _ = drd3_target.dock(smile)
            score_ = score_ * -1 # minimization! 
            if score_ > max_score:
                max_score = score_ 
        except:
            score_ = np.nan 
        scores.append(score_)
        print("max score:", max_score) 
        if ix % 10 == 0:
            score_arr = np.array(scores)
            pd.DataFrame(score_arr).to_csv('drd3_scores.csv', header=None, index=None)

    score_arr = np.array(scores)
    pd.DataFrame(score_arr).to_csv('drd3_scores.csv', header=None, index=None)
    # conda activate dockstring 
    # conda activate dock2 
    import pdb 
    pdb.set_trace() 

if __name__ == "__main__":
    # save_dockstring()
    parser = argparse.ArgumentParser() 
    parser.add_argument('--min_ix', type=int, default=0)  
    parser.add_argument('--max_ix', type=int, default=20_000)  
    args = parser.parse_args() 

    save_condensed_tdc()

    # save_tdc(min_ix=args.min_ix, max_ix=args.max_ix)
    # tmux attach -t tdc0 
    # conda activate tdc_env 
    # CUDA_VISIBLE_DEVICES=1 python3 load_data.py --min_ix 900 