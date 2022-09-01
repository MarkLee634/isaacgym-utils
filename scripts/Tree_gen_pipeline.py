import os
import math
import random
import argparse

import project_config as pcfg
import SCA_tree_gen as sca
import franka_import_tree_multi_env as fit
#import make_relative_rotational_dataset as mkori  # NOTE(daniel): path issues
#import combine_dataset_files as combine  # NOTE(daniel): path issues

## Parameters for the SCA. Lists specify possible parameters, of which one is chosen at random for every tree
TRUNK_HEIGHT_FACTORS = [1,2]
SCALING = 2
PIPE_MODEL_EXPONENT = 3 #suggested values: 2 or 3
MAX_ATTRACTION_POINTS = 200 #lower values lead to more irregular branches
STEP_WIDTH = 1 #determines the lenght of individual tree segments
a = 10 # suggested values Trunk_Height and higher
HEIGHT_STRECH_VALS = [0.5, 0.33] #factors to strech the crown shape of the tree to be non circular
WIDTH_STRECH_VALS = [1, 1.5, 2, 3] #factors to strech the crown shape of the tree to be non circular
ATT_PTS_NUM = [200, 400, 800, 1600] #number of attraction points
PATH = os.path.join(pcfg.PATH, '10Nodes_new_test')

yaml_paths = []
urdf_paths = []
name_dicts = []
edge_defs = []


def main(args):
    tree_pts = args.tree_pts # Specifies number of joints in the tree
    gui_on = args.gui_on # 1 activates the Isaacgym gui, 0 deactivates it
    tree_num = args.tree_num # highest generated tree index. If tree_start_idx is 0 then this equals the number of generated trees
    env_num = args.env_num # number of environments Isaacgym runs in parallel
    tree_start_idx = args.tree_start # index the tree generation starts at.
    path = args.path
    per_tree_path = args.ptpath
    if args.ori_path is None:
        ori_path = per_tree_path
    else:
        ori_path = args.ori_path
    calc_ori = args.ori == 1
    num_iter = args.num_iter
    tree = tree_start_idx

    while tree < tree_num:
        trunk_height = STEP_WIDTH * 0.75 / SCALING
        d_termination = STEP_WIDTH/random.randrange(3, 6)
        d_attraction_values = [
            math.ceil(trunk_height) +  1,
            math.ceil(trunk_height) +  2,
            math.ceil(trunk_height) +  4,
            math.ceil(trunk_height) +  8,
            math.ceil(trunk_height) + 16,
            math.ceil(trunk_height) + 32,
            math.ceil(trunk_height) + 64
        ]
        d_attraction = d_attraction_values[random.randrange(0, len(d_attraction_values) - 1)]
        height_strech = HEIGHT_STRECH_VALS[random.randrange(0,len(HEIGHT_STRECH_VALS)-1)]
        width_strech = WIDTH_STRECH_VALS[random.randrange(0,len(WIDTH_STRECH_VALS)-1)]
        att_pts_max = ATT_PTS_NUM[random.randrange(0, len(ATT_PTS_NUM)-1)]

        print(f"tree {tree}: \n\t d_termination: {d_termination:0.4f} \n\t "
            f"d_attraction: {d_attraction} \n\t height_strech: {height_strech} \n\t "
            f"width_strech: {width_strech} \n\t att_pts_max: {att_pts_max}")

        # Set up their tree generator.
        tg = sca.TreeGenerator(
            max_steps=10000,
            att_pts_max=att_pts_max,
            da=d_attraction,
            dt=d_termination,
            step_width=STEP_WIDTH,
            offset=[-0.5, -0.5, trunk_height],
            scaling=SCALING,
            max_tree_points=tree_pts,
            tip_radius=0.1,
            tree_id=tree,
            pipe_model_exponent=PIPE_MODEL_EXPONENT,
            z_strech=height_strech,
            y_strech=width_strech,
            x_strech=width_strech,
            step_width_scaling=0.65,
            env_num=env_num,
            gui_on=gui_on
        )

        # I think these are from a prior paper or some known formula.
        tg.generate_tree()
        tg.calculate_branch_thickness()

        # Generates `treeX.urdf` and `treeX.yaml`, must be in that order.
        name_dict, edge_def, urdf_path = tg.generate_urdf()
        yaml_path, stiffness_list, damping_list = tg.generate_yaml()
        edge_def2 = tg.calc_edge_tuples()

        # Runs and simulates the env(s).
        fit.import_tree(
            name_dict,
            urdf_path,
            yaml_path,
            edge_def,
            stiffness_list,
            damping_list,
            tree_num=tree,
            tree_pts=tree_pts,
            path=path,
            num_iteration=num_iter
        )
        tree+=1

    # NOTE(daniel): was only recently added, giving some path issues.
    # combine.combine(tree_start=tree_start_idx, tree_num=tree_num, env_num=env_num, get_path=path, put_path=per_tree_path, per_tree=True, tree_pts=tree_pts)

    #if calc_ori:
    #    mkori.make_orientation(tree_start=tree_start_idx, tree_num=tree_num, tree_pts=tree_pts, get_path=per_tree_path, put_path=ori_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tree_pts", type=int, dest="tree_pts",
        help="number of generated tree points")
    parser.add_argument("-gui", type=int, dest="gui_on",
        help="1: Isaac gym gui on 0: Isaac gym gui off", choices=[0,1], default=0)
    parser.add_argument("-tree_num", type=int, dest="tree_num",
        help="number of different trees to be generated", default=100)
    parser.add_argument("-env_num", type=int, dest="env_num",
        help="number of environments to be run in parallel in isaac gym simulation", default=100)
    parser.add_argument("-tree_start", type=int, dest="tree_start", default=0,
        help="start index for generated trees. Used to adjust index and make sequentially generated datasets possible")
    parser.add_argument("-path", type=str, dest="path", default=PATH+"/")
    parser.add_argument("-ptpath", type=str, dest="ptpath", default=PATH+"_per_tree/")
    parser.add_argument("-ori_path", type=str, dest="ori_path")
    parser.add_argument("-ori", type=int, default=1, dest="ori")
    parser.add_argument("-num_iter", type=int, default=10000, dest="num_iter")
    args = parser.parse_args()

    main(args)