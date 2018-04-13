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
from rllab.sampler.utils import rollout
import pickle

from rllab.envs.mujoco.pusher2d_vision_env import PusherEnvVision2D
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv

def eval_success(path):
      obs = path['observations']
      init = obs[0, -12:-10]
      final = obs[-3:, -12:-10]
    #   print("init is", init)
    #   print("final is", final)
      back_flag = np.sum(np.sum((init-final)**2, axis=1) < 0.025) >= 1
      target = obs[:, -3:-1]
      obj = obs[:, -6:-4]
      dists = np.sum((target-obj)**2, 1)  # distances at each timestep
      return np.sum(dists < 0.017) >= 10 and back_flag


#files = glob.glob('data/s3/rllab-fixed-push-experts/*/*itr_300*')
#files = glob.glob('data/s3/init5-push-experts/*/*itr_300*')
# files = '/home/kevin/rllab/data/local/trpo-push2d/trpo_push2d_2018_03_17_02_02_30_0001/itr_950.pkl'
# files = '/home/kevin/rllab/data/local/trpo-push2d/trpo_push2d_2018_03_19_21_58_18_0001/itr_950.pkl'
# files = '/home/kevin/rllab/data/local/trpo-push2d/trpo_push2d_2018_03_28_17_10_40_0001/itr_200.pkl'
# files = '/home/kevin/rllab/data/local/trpo-push2d/trpo_push2d_2018_03_29_14_20_08_0001/itr_650.pkl'
# files = '/home/kevin/rllab/data/local/trpo-push2d/trpo_push2d_2018_03_30_22_31_57_0001/itr_700.pkl'
# files = '/home/kevin/rllab/data/local/trpo-push2d/trpo_push2d_2018_04_04_13_10_13_0001/itr_550.pkl'
files = '/home/kevin/rllab/data/local/trpo-push2d/trpo_push2d_2018_04_06_20_04_47_0001/itr_650.pkl'

# xmls = natsorted(glob.glob('/home/kevin/rllab/vendor/mujoco_models/pusher2d_xmls/train*'))
xmls = natsorted(glob.glob('/home/kevin/rllab/vendor/mujoco_models/pusher2d_shorter_xmls/train*'))
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
max_num_tries = 24 #30 #12
max_path_length = 135 #130

offset = 0
# task_inds = range(0,1000)
# task_inds = range(535,1000)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_start_idx', type=int, default=0)
    parser.add_argument('--xml_end_idx', type=int, default=1000)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--noise', type=float, default=0.0)
    args = parser.parse_args()
    start = args.xml_start_idx
    end = args.xml_end_idx
    test = args.test
    noise = args.noise
    output_dir = 'data/push2d_demos/'

    if test:
        output_dir = 'data/test_push2d_demos/'
        # xmls = natsorted(glob.glob('/home/kevin/rllab/vendor/mujoco_models/pusher2d_xmls/test*'))
        xmls = natsorted(glob.glob('/home/kevin/rllab/vendor/mujoco_models/pusher2d_shorter_xmls/test*'))
        
    output_dir = Path(output_dir)
    output_dir.mkdir_p()
    
    # task_inds = range(start, end)
    task_inds = range(start, end, 2)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=tf_config) as sess:
        data = joblib.load(files)
        policy = data['policy']
        for task_i in task_inds:
            if task_i % 50 == 0:
                print('collecting #' + str(task_i))
            # if '2017_06_23_21_04_45_0091' in expert:
            #     continue
            returns = {j:[] for j in range(2)}
            demoX = {j:[] for j in range(2)}
            demoU = {j:[] for j in range(2)}
            videos = {j:[] for j in range(2)}
            save_xml_files = {j:[] for j in range(2)}
            num_tries = 0
            obj_left = True
            reach_half = False
            while (len(returns[0]) < demos_per_expert and num_tries < max_num_tries):
                if len(returns[0]) == demos_per_expert / 2 and not reach_half:
                    num_tries = 12
                    xml_file = xmls[task_i*24 + num_tries]
                    xml_file1 = xmls[(task_i+1)*24 + num_tries]
                    reach_half = True
                else:
                    xml_file = xmls[task_i*24 + num_tries]
                    xml_file1 = xmls[(task_i+1)*24 + num_tries]
                if len(returns[0]) < demos_per_expert / 2 and (num_tries == max_num_tries / 2):
                    break
                print('xml for object is', xml_file)
                pusher_env = PusherEnvVision2D(**{'xml_file':xml_file, 'distractors': True})
                env = TfEnv(normalize(pusher_env))
                # path = rollout(env, policy, max_path_length=120, speedup=1,
                path = rollout(env, policy, max_path_length=max_path_length, speedup=1, noise=noise,
                         animated=True, always_return_paths=True, save_video=False, vision=True)
                # close the window after rollout
                env.render(close=True)
                import time
                time.sleep(0.1)
                # if path['observations'][-1,0] > joint_thresh:
                #     num_tries = max_num_tries
                #if path['rewards'].sum() > filter_thresh and path['observations'][-1,0] < joint_thresh:
                if eval_success(path):# and path['observations'][-1,0] < joint_thresh:
                    print('xml for distractor is', xml_file1)
                    pusher_env = PusherEnvVision2D(**{'xml_file':xml_file1, 'distractors': True})
                    env = TfEnv(normalize(pusher_env))
                    # path = rollout(env, policy, max_path_length=120, speedup=1,
                    path1 = rollout(env, policy, max_path_length=max_path_length, speedup=1, noise=noise,
                             animated=True, always_return_paths=True, save_video=False, vision=True)
                    # close the window after rollout
                    env.render(close=True)
                    import time
                    time.sleep(0.1)
                    if eval_success(path1):
                        returns[0].append(path['rewards'].sum())
                        demoX[0].append(path['nonimage_obs'])
                        demoU[0].append(path['actions'])
                        videos[0].append(path['image_obs'])
                        save_xml_files[0].append(xml_file)
                        returns[1].append(path1['rewards'].sum())
                        demoX[1].append(path1['nonimage_obs'])
                        demoU[1].append(path1['actions'])
                        videos[1].append(path1['image_obs'])
                        save_xml_files[1].append(xml_file1)
                num_tries += 1
                print(len(returns[0]))
            if len(returns[0]) >= demos_per_expert:
                for j in range(2):
                    X = np.array(demoX[j])
                    U = np.array(demoU[j])
                    with open(output_dir + str(task_i + j) + '.pkl', 'wb') as f:
                        #pickle.dump({'demoX': demoX, 'demoU': demoU, 'xml':prefix+suffix}, f, protocol=2)
                        pickle.dump({'demoX': X, 'demoU': U, 'xml':save_xml_files[j]}, f, protocol=2)
                    video_dir = output_dir + 'object_' + str(task_i + j) + '/'
                    video_path = Path(video_dir)
                    video_path.mkdir_p()
                    for demo_index in range(demos_per_expert):
                        cnt = 0
                        while (np.all(videos[j][demo_index][cnt] == 0.)):
                            cnt += 1
                        if cnt > 0:
                            videos[j][demo_index][:cnt] = videos[j][demo_index][cnt]
                        save_path = video_dir + 'cond' + str(demo_index) + '.samp0.gif'
                        print('Saving video to %s' % save_path)
                        imageio.mimwrite(save_path, list(videos[j][demo_index]), format='gif')
    tf.reset_default_graph()
