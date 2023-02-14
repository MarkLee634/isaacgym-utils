import numpy as np
import os
import datetime
import sys

import IP_real_tree_dataCollect as IP_dataCollect

'''
This script is to combine tree interaction data from Isaac Gym into GNN dataset format. 
Concatenate all tree variation into a single interaction npy file.
input: {[10]X_force_applied_tree 1...N}.npy 
output: {X_final, Y_final, X_edge_def, X_force_applied}.npy
'''

SIZE_TREE_NODE = 11
PUT_PATH = "/home/mark/data/IROS2023/IROS2023_sim_dataset_1015/node10/"

NUM_TREE_VARIATION = 1
INPUT_FOLDER_PATH = "/home/mark/data/IsaacGym/test_sim_uniform_distribution/dataset_by_tree" #"/home/jan-malte/IROS_tree_dataset/isaacgym-utils/dataset_15_node_only_test/dataset_by_tree/"



def load_npy(data_dir, tree_num, prefix):
    # Load npy files from dataset_dir   

    X_edges = np.load(os.path.join(data_dir, prefix + 'X_edge_def_tree%s.npy'%tree_num))
    X_force = np.load(os.path.join(data_dir, prefix + 'X_force_applied_tree%s.npy'%tree_num))
    X_pos = np.load(os.path.join(data_dir, prefix + 'X_vertex_init_pose_tree%s.npy'%tree_num))
    Y_pos = np.load(os.path.join(data_dir, prefix + 'Y_vertex_final_pos_tree%s.npy'%tree_num))

    # # Truncate node orientations and tranpose to shape (num_graphs, num_nodes, 3)
    # X_pos = X_pos[:, :7, :].transpose((0,2,1))
    # Y_pos = Y_pos[:, :7, :].transpose((0,2,1))
    # X_force = X_force.transpose((0,2,1))

    return X_edges, X_force, X_pos, Y_pos

#test for working copy version
def load_npy_test(data_dir_clean):
    desired_F = np.load(data_dir_clean + "final_F.npy")
    desired_X = np.load(data_dir_clean + "final_X.npy")
    desired_Y = np.load(data_dir_clean + "final_Y.npy")
    desired_edge = np.load(data_dir_clean + "X_edge_def.npy")

    return desired_F, desired_X, desired_Y, desired_edge


    
if __name__ == '__main__':
    print(f" ================= script start ================= ")

    # ---------------- desired structured data from working version ----------------
    # INPUT_FOLDER_PATH = "/home/mark/github/tree_is_all_you_need/data/sim_data_restructure/10Nodes_by_tree/trial0/"
    # desired_F, desired_X, desired_Y, desired_edge = load_npy_test(INPUT_FOLDER_PATH)
    # print(f"desired_F.shape: {desired_F.shape}, X {desired_X.shape}, Y {desired_Y.shape}, edge {desired_edge.shape}")
    # #desired_F.shape: (10021, 3, 16), X (10021, 7, 16), Y (10021, 7, 16), edge (15, 2)

    # ---------------- data to structure from IG ----------------

    prefix = "[" + str(SIZE_TREE_NODE) + "]"

    #iterate through total number of trees in GET_PATH and make a directory to store final_F, final_X, final_Y, X_edge_def
    for tree_index in range(NUM_TREE_VARIATION):
        print(f"tree_index {tree_index}")
        #make directory

        #folder name
        folder_name = "trial" + str(tree_index)
        folder_path = os.path.join(PUT_PATH,folder_name)

        #create folder if not exist
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print(f"[{datetime.datetime.now()}], folder created: {folder_path}")

        
        #load npy files
        X_edges, X_force, X_pos, Y_pos = load_npy(data_dir= INPUT_FOLDER_PATH, tree_num = tree_index, prefix=prefix)
        print(f"X_edges.shape: {X_edges.shape}, X_force {X_force.shape}, X_pos {X_pos.shape}, Y_pos {Y_pos.shape}")

        Real2Sim = IP_dataCollect.Real2SimEvaluator()
        sim_link_plot = Real2Sim.plot_sim_tree_measurements(X_pos, X_force, Y_pos, X_edges)

    

       
        # save npy files into directory
        # np.save(folder_path + '/final_F', X_force)
        # np.save(folder_path + '/final_X', X_pos)
        # np.save(folder_path + '/final_Y', Y_pos)
        # np.save(folder_path + '/X_edge_def', X_edges)

    print(f" ================= script end ================= ")
    





