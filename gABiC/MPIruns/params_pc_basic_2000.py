
N = 2000
rstate = 2
output_dir = "parallel-benchmark/"
#input_file = "input_data/TNG300_new_normalized_fromv1_wo_mstar9_no_collinear.csv"
#input= "data_TNG300_disk_2p/"
#input= "micols_data_TNG300_disk_2p_wo_fdisk/"
input= "../datasets_input/magic_irri_20k.csv"
file_adjlist = output_dir + "file_adjlist-" + str(N) + "-mirri-model_rstate_" + str(rstate) + "_CPU.txt"
file_edgelist= output_dir +"file_edgelist-" + str(N) + "-mirri-model_rstate_"+ str(rstate) +"_CPU.gz"
