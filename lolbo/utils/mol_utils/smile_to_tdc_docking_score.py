# from mol_utils import (
#     smile_to_tdc_docking_score,
#     setup_tdc_oracle,
# )
# import argparse 

# tdc_oracle = setup_tdc_oracle('3pbl_docking')  

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser() 
#     parser.add_argument('--smile', default='CCC' ) 
#     args = parser.parse_args() 
#     score = smile_to_tdc_docking_score(
#         smiles_str=args.smile, 
#         tdc_oracle=tdc_oracle, 
#         max_smile_len=400, 
#         timeout=100
#     )