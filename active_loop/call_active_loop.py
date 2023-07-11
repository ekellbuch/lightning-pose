# -*- coding: utf-8 -*-
"""
Call active loop functions
"""
import os
import random
import pandas as pd
from omegaconf import OmegaConf
import sys
from pathlib import Path
import numpy as np
from hydra import compose, initialize, initialize_config_dir
from datetime import datetime
import wandb

def initialize_iteration_folder(data_dir):
    """ Initialize the iteration folder
    :param data_dir: where the iteration folder will be created.
    clone
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


def select_frames(active_iter_cfg):
    """
    Step 2: select frames to label
    Implement the logic for selecting frames based on the specified method:
    :param active_iter_cfg: active loop config file
    :return:
      : selected_indices_file: csv file with selected frames from active loop.
    """
    #TODO(may not want to overwrite file)
    method = active_iter_cfg.method
    num_frames = active_iter_cfg.num_frames
    output_dir = active_iter_cfg.iteration_folder

    # We need to know the index of our selected data (list)
    if method == 'random':
      # select random frames from eval data in prev run.
      all_data = pd.read_csv(active_iter_cfg.eval_data_file_prev_run, header=[0,1,2], index_col=0)
      selected_indices = np.unique(random.sample(range(len(all_data)), num_frames))  # Get index from either places
      selected_rows = all_data.iloc[selected_indices]
      selected_indices_file = f'iteration_{method}_indices.csv'
      selected_indices_file = os.path.join(output_dir, selected_indices_file)
      selected_rows.to_csv(selected_indices_file)
    else:
      NotImplementedError(f'{method} is not implemented yet.')

    return selected_indices_file


def merge_collected_data(active_iter_cfg, selected_frames_file):
    """
    # Step 3: Merge new CollectedData.csv with the original CollectedData.csv
    # merge Collected_data.csv to include iteration_random_indices.csv
    # remove iteration_random_indices.csv from  {CollectedData}_new.csv

    :param active_iter_cfg:
    :return:
    """

    train_data_file = os.path.join(active_iter_cfg.train_data_file_prev_run)
    train_data = pd.read_csv(train_data_file, header=[0,1,2], index_col=0)
    # read selected frames
    selected_frames_df = pd.read_csv(selected_frames_file, header=[0,1,2], index_col=0)

    # concat train data and selected frames and merge
    new_train_data = pd.concat([train_data, selected_frames_df])
    new_train_data.to_csv(active_iter_cfg.train_data_file)

    # remove selected_frames from val data
    val_data = pd.read_csv(active_iter_cfg.eval_data_file_prev_run, header=[0,1,2], index_col=0)
    val_data.drop(index=selected_frames_df.index, inplace=True)
    val_data.to_csv(active_iter_cfg.eval_data_file)
    return


def update_config_yaml(config_file, merged_data_dir):
    """
    # Step 4: Update the config.yaml file
    :param config_file:
    :param merged_data_dir:
    :return:
    """
    # Load the config file
    cfg = OmegaConf.load(config_file)

    OmegaConf.update(cfg,'data.csv_file',merged_data_dir,merge=True)

    OmegaConf.save(cfg, config_file)


def active_loop_step(active_loop_cfg):
    """
    TODO(haotianxiansti) update comments
    TODO(haotianxiansti) update to use hydra?
    # Step 6: Launch the next active_loop iteration
    """
    # read yaml file
    experiment_cfg = OmegaConf.load(active_loop_cfg.active_loop.experiment_cfg)

    # read params for current active loop iteration
    iteration_number = active_loop_cfg.active_loop.current_iteration
    iterations_folder = active_loop_cfg.active_loop.iterations_folder

    iteration_key = 'iteration_{}'.format(iteration_number)
    active_iter_cfg = active_loop_cfg[iteration_key]
    iteration_folder = os.path.abspath(str(
        Path(experiment_cfg.data.data_dir,
             experiment_cfg.data.csv_file).parent.absolute() / iterations_folder / iteration_key)
    )

    # read train and eval files
    train_data_file_prev_run = str(Path(experiment_cfg.data.data_dir,
                                        active_iter_cfg.csv_file_prev_run))
    eval_data_file_prev_run = train_data_file_prev_run.replace('.csv', '_new.csv')

    # update params to config file
    active_iter_cfg.iteration_key = iteration_key
    active_iter_cfg.iteration_prefix = '{}_{}'.format(active_iter_cfg.method,
                                                      active_iter_cfg.num_frames)
    active_iter_cfg.iteration_folder = iteration_folder
    active_iter_cfg.train_data_file_prev_run = train_data_file_prev_run
    active_iter_cfg.eval_data_file_prev_run = eval_data_file_prev_run
    active_iter_cfg.train_data_file = os.path.join(
        active_iter_cfg.iteration_folder,
        '{}_{}'.format(active_iter_cfg.iteration_prefix,
                       os.path.basename(train_data_file_prev_run))
    )
    active_iter_cfg.eval_data_file = active_iter_cfg.train_data_file.replace('.csv', '_new.csv')

    # Active Loop parameters
    # Step 1: Initialize the iteration folder
    #  TODO(haotianxiansti):  add code for iter 0 (select frames when no labeles are present)
    initialize_iteration_folder(active_iter_cfg.iteration_folder)

    selected_frames_file = select_frames(active_iter_cfg)

    # Now, we have in the directory:
    # created Collected_data_new_merged and Collected_data_merged.csv
    merge_collected_data(active_iter_cfg, selected_frames_file)
    # run algorithm with new config file
    # make relative to data_dir
    relpath = os.path.relpath(active_iter_cfg.train_data_file, experiment_cfg.data.data_dir)
    #print('rerun algorithm with new config file:\n{}'.format(relpath), flush=True)

    return relpath


def call_active_all(active_cfg):
    """
    # Step 5: Call active learning algorithm
    :param config:
    :return:
    """
    # Read experiment config file
    exp_cfg = OmegaConf.load(active_cfg.active_loop.experiment_cfg)

    # inherit params from active loop:
    exp_cfg.wandb.params.project = active_cfg.project
    if active_cfg.active_loop.fast_dev_run == 1:
        exp_cfg.training.fast_dev_run = True

    num_iterations = active_cfg.active_loop.end_iteration - active_cfg.active_loop.start_iteration + 1
    for current_iteration in range(active_cfg.active_loop.start_iteration,
                                   active_cfg.active_loop.end_iteration + 1):
        print('\n\n Experiment iter {}'.format(current_iteration), flush=True)

        if current_iteration == 0:
            # step 1: select frames to label is skipped in demo mode.
            exp_cfg.model.model_name = 'iter_{}_{}'.format(current_iteration, 'baseline')

        # step 2: train model using exp_cfg
        train_output_dir = run_train(exp_cfg)

        # step 3: call active loop
        iteration_key = 'iteration_{}'.format(current_iteration + 1)
        active_cfg.active_loop.current_iteration = current_iteration
        active_cfg[iteration_key].output_prev_run = train_output_dir
        active_cfg[iteration_key].csv_file_prev_run = exp_cfg.data.csv_file
        #print('\n\nActive loop config after iter {}'.format(current_iteration), active_cfg, flush=True)
        new_train_file = active_loop_step(active_cfg)

        # update config file
        exp_cfg.data.csv_file = new_train_file
        exp_cfg.model.model_name = 'iter_{}_{}'.format(current_iteration + 1,
                                                       active_cfg[iteration_key].method)

    # write new active_cfg file
    return active_cfg


def run_train(cfg):
    sys.path.append(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
    import train_hydra
    cwd = os.getcwd()
    today_str = datetime.now().strftime("%y-%m-%d")
    ctime_str = datetime.now().strftime("%H-%M-%S")
    new_dir = f"./outputs/{today_str}/{ctime_str}"
    os.makedirs(new_dir, exist_ok=False)
    os.chdir(new_dir)
    train_output_dir = train_hydra.train(cfg)
    os.chdir(cwd)
    wandb.finish()
    return train_output_dir


if __name__ == "__main__":
    # read active config file
    active_loop_cfg = OmegaConf.load(sys.argv[1])
    # active_loop_step(active_loop_cfg)
    call_active_all(active_loop_cfg)