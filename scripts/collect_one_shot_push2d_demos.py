import argparse

import glob
import imageio
import joblib
import numpy as np
from natsort import natsorted
import os
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout
import pickle

from rllab.envs.mujoco.pusher2d_vision_env import PusherEnvVision2D
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv

def eval_success(path):
      obs = path['observations']
      init = obs[0, -9:-7]
      final = obs[-5:, -9:-7]
      back_flag = (np.sum((init-final)**2) < 0.017) >= 1
      target = obs[:, -3:-1]
      obj = obs[:, -6:-4]
      dists = np.sum((target-obj)**2, 1)  # distances at each timestep
      return np.sum(dists < 0.017) >= 10 and back_flag


#files = glob.glob('data/s3/rllab-fixed-push-experts/*/*itr_300*')
#files = glob.glob('data/s3/init5-push-experts/*/*itr_300*')
files = '/home/kevin/rllab/data/local/trpo-push2d/trpo_push2d_2018_03_16_16_54_10_0001/itr_900.pkl'
xmls = natsorted(glob.glob('/home/kevin/rllab/vendor/mujoco_models/pusher2d_xmls/*'))
demos_per_expert = 12 #8
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

output_dir = 'data/push2d_demos/'

TEST2 = False  # if True, use held out textures.
if TEST2:
    output_dir = 'data/test2_paired_consistent_push_demos/'

offset = 0
task_inds = range(0,1000)
with tf.Session() as sess:
    data = joblib.load(files)
    policy = data['policy']
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
            if TEST2:
                xml_file = xml_file.replace('test', 'test2')
            pusher_env = PusherEnvVision2D(**{'xml_file':xml_file, 'distractors': True})
            env = TfEnv(normalize(pusher_env))
            num_tries += 1
            path = rollout(env, policy, max_path_length=100, speedup=1,
                     animated=True, always_return_paths=True, save_video=False, vision=True)
            # close the window after rollout
            env.render(close=True)
            if path['observations'][-1,0] > joint_thresh:
                num_tries = max_num_tries
            #if path['rewards'].sum() > filter_thresh and path['observations'][-1,0] < joint_thresh:
            if eval_success(path) and path['observations'][-1,0] < joint_thresh:
                returns.append(path['rewards'].sum())
                demoX.append(path['nonimage_obs'])
                demoU.append(path['actions'])
                videos.append(path['image_obs'])
        if len(returns) >= demos_per_expert:
            demoX = np.array(demoX)
            demoU = np.array(demoU)
            with open(output_dir + str(expert_i) + '.pkl', 'wb') as f:
                #pickle.dump({'demoX': demoX, 'demoU': demoU, 'xml':prefix+suffix}, f, protocol=2)
                pickle.dump({'demoX': demoX, 'demoU': demoU, 'xml':xml_file}, f, protocol=2)
            video_dir = output_dir + 'object_' + str(expert_i) + '/'
            os.mkdir(video_dir)
            for demo_index in range(demos_per_expert):
                imageio.mimwrite(video_dir + 'cond' + str(demo_index) + '.samp0.gif', list(videos[demo_index]), format='gif')
tf.reset_default_graph()