import argparse
from path import Path

import glob
import imageio
import joblib
import numpy as np
from natsort import natsorted
import os
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout_two_policy
import pickle

from rllab.envs.mujoco.pusher2d_vision_env import PusherEnvVision2D
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv

def eval_success(path):
      obs = path['observations']
    #   init = obs[0, -12:-10]
    #   final = obs[-10:, -12:-10]
      target = obs[:-20, -3:-1]
      obj = obs[:-20, -6:-4]
      distractor = obs[:-20, -9:-7]
      dists1 = np.sum((target-obj)**2, 1)  # distances at each timestep
      dists2 = np.sum((target-distractor)**2, 1)  # distances at each timestep
      return np.sum(dists1 < 0.025) >= 5 and np.sum(dists2 < 0.025) >= 5


#files = glob.glob('data/s3/rllab-fixed-push-experts/*/*itr_300*')
#files = glob.glob('data/s3/init5-push-experts/*/*itr_300*')
# files = '/home/kevin/rllab/data/local/trpo-push2d/trpo_push2d_2018_03_17_02_02_30_0001/itr_950.pkl'
# file1 = '/home/kevin/rllab/data/local/trpo-push2d/trpo_push2d_2018_03_19_21_58_18_0001/itr_950.pkl'
file1 = '/home/kevin/rllab/data/local/trpo-push2d/trpo_push2d_2018_03_28_17_10_40_0001/itr_200.pkl'
file2 = '/home/kevin/rllab/data/local/trpo-push2d-distractor/trpo_push2d_distractor_2018_03_31_19_54_01_0001/itr_300.pkl'
xmls = natsorted(glob.glob('/home/kevin/rllab/vendor/mujoco_models/pusher2d_multigoal_xmls/*'))
demos_per_expert = 8 #8
#output_dir = 'data/expert_demos/'

#use_filter = True
#filter_thresh = -34
#joint_thresh = 1.0
#max_num_tries=16
#output_dir = 'data/expert_demos_filter_joint0/'

use_filter = True
filter_thresh = -55
joint_thresh=0.7
max_num_tries=24 #30 #12

output_dir = 'data/test_push2d_multigoal_demos/'
output_dir = Path(output_dir)
output_dir.mkdir_p()

offset = 0
task_inds = range(0,100)
with tf.Session() as sess:
    data1 = joblib.load(file1)
    data2 = joblib.load(file2)
    policy1 = data1['policy']
    policy2 = data2['policy']
    for task_i in task_inds:
        if task_i % 25 == 0:
            print('collecting #' + str(task_i))
        # if '2017_06_23_21_04_45_0091' in expert:
        #     continue
        returns = []
        demoX = []
        demoU = []
        videos = []
        num_tries = 0
        obj_left = True
        while (len(returns) < demos_per_expert and num_tries < max_num_tries):
            xml_file = xmls[task_i*24 + num_tries]
            print(xml_file)
            pusher_env = PusherEnvVision2D(**{'xml_file':xml_file, 'distractors': True})
            env = TfEnv(normalize(pusher_env))
            num_tries += 1
            # path = rollout(env, policy, max_path_length=110, speedup=1,
            path = rollout_two_policy(env, policy1, policy2, path_length1=115, max_path_length=235, speedup=1,
                     animated=True, always_return_paths=True, save_video=False, vision=True)
            # close the window after rollout
            env.render(close=True)
            # if path['observations'][-1,0] > joint_thresh:
            #     num_tries = max_num_tries
            #if path['rewards'].sum() > filter_thresh and path['observations'][-1,0] < joint_thresh:
            if eval_success(path):# and path['observations'][-1,0] < joint_thresh:
                returns.append(path['rewards'].sum())
                demoX.append(path['nonimage_obs'])
                demoU.append(path['actions'])
                videos.append(path['image_obs'])
            print(len(returns))
        if len(returns) >= demos_per_expert:
            demoX = np.array(demoX)
            demoU = np.array(demoU)
            with open(output_dir + str(task_i) + '.pkl', 'wb') as f:
                #pickle.dump({'demoX': demoX, 'demoU': demoU, 'xml':prefix+suffix}, f, protocol=2)
                pickle.dump({'demoX': demoX, 'demoU': demoU, 'xml':xml_file}, f, protocol=2)
            video_dir = output_dir + 'object_' + str(task_i) + '/'
            video_path = Path(video_dir)
            video_path.mkdir_p()
            for demo_index in range(demos_per_expert):
                save_path = video_dir + 'cond' + str(demo_index) + '.samp0.gif'
                print('Saving video to %s' % save_path)
                imageio.mimwrite(save_path, list(videos[demo_index]), format='gif')
tf.reset_default_graph()