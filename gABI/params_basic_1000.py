
N = 1000
rstate = 2
output_dir = "serial-kpc-output/"
#input_file = "input_data/TNG300_new_normalized_fromv1_wo_mstar9_no_collinear.csv"
#input= "data_TNG300_disk_2p/"
#input= "micols_data_TNG300_disk_2p_wo_fdisk/"
input= "datasets_input/basic_example_data_30000.csv"
file_adjlist = output_dir + "file_adjlist-" + str(N) + "-basic-model_rstate_" + str(rstate) + "_GPU.txt"
file_edgelist= output_dir +"file_edgelist-" + str(N) + "-basic-model_rstate_"+ str(rstate) +"_GPU.gz"
