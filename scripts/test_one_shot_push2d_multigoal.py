import glob
import os
from path import Path
import imageio
import joblib
import pickle
import numpy as np
import random
import tensorflow as tf
from PIL import Image
from rllab.sampler.utils import rollout_sliding_window
import sys
import argparse

from rllab.envs.mujoco.pusher2d_vision_env import PusherEnvVision2D
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv

DEMO_DIR = 'data/test_push2d_multigoal_demos/'
SCALE_FILE_PATH = '/home/kevin/maml_imitation_private/data/scale_and_bias_push2d_pair.pkl'
META_PATH = '/home/kevin/maml_imitation_private/data/checkpoints/push2d_pair.xavier_init.4_conv.2_strides.16_5x5_filters.3_fc.200_dim.bt_dim_10.mbs_15.ubs_1.meta_lr_0.001.numstep_1.updatelr_0.005.conv_bt.all_fc_bt.fp.two_heads/model_48000.meta'
LOG_DIR = '/home/kevin/maml_imitation_private/logs/'

class TFAgent(object):
    def __init__(self, feed_dict, scale_bias_file, sess):
        self.sess = sess
        self.feed_dict = feed_dict

        if scale_bias_file:
            with open(SCALE_FILE_PATH, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                self.scale = data['scale']
                self.bias = data['bias']
        else:
            self.scale = None

    def reset(self):
        pass

    def set_demo(self, demo_gif, demoX, demoU):
        #import pdb; pdb.set_trace()

        # concatenate demos in time
        demo_gif = np.array(demo_gif)
        try:
            N, T, H, W, C = demo_gif.shape
        except:
            import pdb; pdb.set_trace()
        self.update_batch_size = N
        self.T = T
        demo_gif = np.reshape(demo_gif, [N*T, H, W, C])
        demo_gif = np.array(demo_gif)[:,:,:,:3].transpose(0,3,2,1).astype('float32') / 255.0
        self.demoVideo = demo_gif.reshape(1, N*T, -1)
        self.demoX = demoX
        self.demoU = demoU

    def get_action(self, obs):
        obs = obs.reshape((1,1,18))
        action = self.sess.run(self.feed_dict['output'], {self.feed_dict['statea']: self.demoX.dot(self.scale) + self.bias,
                               self.feed_dict['actiona']: self.demoU,
                               self.feed_dict['stateb']: obs.dot(self.scale) + self.bias})
        return action, dict()

    def get_vision_action(self, image, obs, t=-1):
        # if CROP:
        #     image = np.array(Image.fromarray(image).crop((40,25,120,90)))
        
        T = 1
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0).transpose(0,3,2,1).astype('float32') / 255.0
            image = image.reshape((1, 1, -1))
    
            obs = obs.reshape((1,1,12))
        else:
            T, _, _, _ = image.shape
            image = np.expand_dims(image, 0).transpose(0,1,4,3,2).astype('float32') / 255.0
            image = image.reshape((1, T, -1))
    
            obs = obs.reshape((1,T,12))

        action = self.sess.run(self.feed_dict['output'],
                           {self.feed_dict['statea']: self.demoX.dot(self.scale) + self.bias,
                           self.feed_dict['obsa']: self.demoVideo,
                           self.feed_dict['actiona']: self.demoU,
                           self.feed_dict['stateb']: obs.dot(self.scale) + self.bias,
                           self.feed_dict['obsb']: image})
        if T > 1:
            action = np.squeeze(action)[t]
        return action, dict()

def find_xml_filepath(demo_info):
    xml_filepath = demo_info['xml']
    # # suffixs = [xml_path[xml_path.index('test_'):] for xml_path in xml_filepath]
    # suffixs = [xml_path[xml_path.index('train_'):] for xml_path in xml_filepath]
    # prefix = XML_PATH
    # xml_filepath = [str(prefix + suffix) for suffix in suffixs]

    return xml_filepath
    
def load_env(xml_filepath):
    pusher_env = PusherEnvVision2D(**{'xml_file':xml_filepath, 'distractors': True})
    env = TfEnv(normalize(pusher_env))
    return env

def load_demo(task_id, demo_dir, demo_inds):
    demo_info = pickle.load(open(demo_dir+task_id+'.pkl', 'rb'))
    demoX = demo_info['demoX'][demo_inds,:,:]
    demoU = demo_info['demoU'][demo_inds,:,:]
    d1, d2, _ = demoX.shape
    demoX = np.reshape(demoX, [1, d1*d2, -1])
    demoU = np.reshape(demoU, [1, d1*d2, -1])

    # read in demo video
    demo_gifs = [imageio.mimread(demo_dir+'object_'+task_id+'/cond%d.samp0.gif' % demo_ind) for demo_ind in demo_inds]
    print([demo_dir+'object_'+task_id+'/cond%d.samp0.gif' % demo_ind for demo_ind in demo_inds])
    return demoX, demoU, demo_gifs, demo_info

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

def main(meta_path, demo_dir, log_dir, test=True, window_size=135,
        run_steps=135, save_video=True, lstm=False, num_input_demos=1):
    model_dir = meta_path[:meta_path.index('model')]
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, meta_path[:-5])
        
        feed_dict = {
            'obsa': tf.get_default_graph().get_tensor_by_name('obsa:0'),
            'statea': tf.get_default_graph().get_tensor_by_name('statea:0'),
            'actiona': tf.get_default_graph().get_tensor_by_name('actiona:0'),
            'obsb': tf.get_default_graph().get_tensor_by_name('obsb:0'),
            'stateb': tf.get_default_graph().get_tensor_by_name('stateb:0'),
            'output': tf.get_default_graph().get_tensor_by_name('output_action:0')   ,  
        }

        scale_file = SCALE_FILE_PATH
        files = glob.glob(os.path.join(demo_dir, '*.pkl'))
        all_ids = [int(f.split('/')[-1][:-4]) for f in files]
        all_ids.sort()
        num_success = 0
        num_trials = 0
        trials_per_task = 5
        
        if not test:
            task_ids = all_ids[-60:]
        else:
            task_ids = all_ids
    
        for task_id in task_ids:
            demo_inds = [0] # for consistency of comparison
            if num_input_demos > 1:
                demo_inds += range(12, 12+int(num_input_demos / 2))
                demo_inds += range(2, 2+int((num_input_demos-1) / 2))
            assert len(demo_inds) == num_input_demos
            demoX, demoU, demo_gifs, demo_info = load_demo(str(task_id), demo_dir, demo_inds)
            demo_gifs_arr = np.array(demo_gifs)
            N, T, H, W, C = demo_gifs_arr.shape
            # demo_gifs_arr = np.concatenate((demo_gifs_arr[:, :80, :, :, :], demo_gifs_arr[:, 135:, :, :, :]), axis=1)
            # demo_gifs = list(demo_gifs_arr)
            # demoX = demoX.reshape(-1, T, demoX.shape[-1])
            # demoU = demoU.reshape(-1, T, demoU.shape[-1])
            # demoX = np.concatenate((demoX[:, :80, :], demoX[:, 135:, :]), axis=1)
            # demoU = np.concatenate((demoU[:, :80, :], demoU[:, 135:, :]), axis=1)
            # demoX = demoX.reshape(1, -1, demoX.shape[-1])
            # demoU = demoU.reshape(1, -1, demoU.shape[-1])
            # T -= 55
            # load xml file
            xml_filepath = find_xml_filepath(demo_info)
            # env = load_env(demo_info)
    
            policy = TFAgent(feed_dict, scale_file, sess)
            demo_data = (demo_gifs, demoX, demoU)
            returns = []
            gif_dir = model_dir + '/evaluated_gifs/'
            gif_path = Path(gif_dir)
            gif_path.mkdir_p()
            for j in range(1, trials_per_task+1):
                print(xml_filepath[j])
                env = load_env(xml_filepath[j])
                video_suffix = gif_dir + str(id) + 'demo_' + str(num_input_demos) + '_' + str(len(returns)) + '.gif'
                path = rollout_sliding_window(env, policy, max_path_length=T, env_reset=True,
                                               animated=True, speedup=1, always_return_paths=True, 
                                               save_video=save_video, video_filename=video_suffix, 
                                               vision=True, lstm=lstm, is_push_2d=True, demo=demo_data,
                                               window_size=window_size, run_steps=run_steps)
                env.render(close=True)
                # val_demoX, val_demoU, val_demo_gifs, val_demo_info = load_demo(str(task_id), demo_dir, [j])
                # init_act = policy.get_vision_action(np.array(val_demo_gifs)[0, 0, :, :, :3], val_demoX[0, 0, :])[0]
                # curr_loss = np.mean((val_demoU*50.0 - path['actions']*50.0)**2)
                # import pdb; pdb.set_trace()
                # print('Loss is ', curr_loss)
                num_trials += 1
                if eval_success(path):
                    num_success += 1
                print('Return: '+str(path['rewards'].sum()))
                returns.append(path['rewards'].sum())
                print('Average Return so far: ' + str(np.mean(returns)))
                print('Success Rate so far: ' + str(float(num_success)/num_trials))
                sys.stdout.flush()
                if len(returns) > trials_per_task:
                    break
    success_rate_msg = "Final success rate is %.5f" % (float(num_success)/num_trials)
    with open(log_dir + '/log_push2d_multigoal.txt', 'a') as f:
        f.write(meta_path[:-5] + ':\n')
        f.write(success_rate_msg + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=135)
    parser.add_argument('--run_steps', type=int, default=135)
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--save_video', type=bool, default=False)
    args = parser.parse_args()
    window_size = args.window_size
    run_steps = args.run_steps
    test = args.test
    save_video = args.save_video
    main(meta_path=META_PATH, 
        demo_dir=DEMO_DIR,
        log_dir=LOG_DIR, 
        test=test, 
        window_size=window_size,
        run_steps=run_steps,
        save_video=save_video)
