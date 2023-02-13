from tdc import Oracle

oracle = Oracle('3pbl_docking')

print("loaded")

smile = "CCC"

score = oracle(smile)

print("score:", score)

# conda activate tdc_env   # !!!! works !!!! 