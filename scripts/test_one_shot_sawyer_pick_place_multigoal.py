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

from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place_mil_vision_seq import SawyerPickPlaceMILVisionSeqEnv
from multiworld.core.flat_goal_env import FlatGoalEnv

# DEMO_DIR = 'data/sawyer_pick_and_place_noisy_larger_textures_goalpos_push_test/'#_push/'
# DEMO_DIR = 'data/sawyer_pick_and_place_noisy_larger_textures_goalpos_push_vanish_test/'#_distr2/'#_4obj/'#_push/'
DEMO_DIR = 'data/sawyer_pick_and_place_noisy_larger_textures_goalpos_push_test/'#_distr2/'#_4obj/'#_push/'
# DEMO_DIR = 'data/sawyer_pick_and_place_noisy_larger_single/'
# DEMO_DIR = 'data/sawyer_pick_and_place_noisy_larger_single_1206/'
# TEST_DEMO_DIR = 'data/sawyer_pick_and_place_test/'
# SCALE_FILE_PATH = '/home/kevin/maml_imitation_private/data/scale_and_bias_sawyer_pick_and_place_noisy_larger_single.pkl'
# SCALE_FILE_PATH = '/home/kevin/maml_imitation_private/data/scale_and_bias_sawyer_pick_and_place_noisy_larger_single_1205.pkl'
# SCALE_FILE_PATH = '/home/kevin/maml_imitation_private/data/scale_and_bias_sawyer_pick_and_place_noisy_larger_textures_goalpos_push.pkl'
# SCALE_FILE_PATH = '/home/kevin/maml_imitation_private/data/scale_and_bias_sawyer_pick_and_place_noisy_larger_textures_goalpos_push_initobj.pkl'
SCALE_FILE_PATH = '/home/kevin/maml_imitation_private/data/scale_and_bias_sawyer_pick_and_place_textures_goalpos_push_initobj_4.pkl'
# META_PATH = '/home/kevin/maml_imitation_private/data/checkpoints/sawyer_pick_and_place_noisy_larger_single_1205.xavier_init.4_conv.2_strides.32_5x5_filters.bt_dim_20.mbs_1.ubs_1.meta_lr_0.001.numstep_1.updatelr_0.0.3_fc.100_dim.conv_bt.fp.learn_ee_pos/model_34000.meta'
# LOG_DIR = '/home/kevin/maml_imitation_private/data/checkpoints/sawyer_pick_and_place_noisy_larger_single_1205.xavier_init.4_conv.2_strides.32_5x5_filters.bt_dim_20.mbs_1.ubs_1.meta_lr_0.001.numstep_1.updatelr_0.0.3_fc.100_dim.conv_bt.fp.learn_ee_pos/'
# META_PATH = '/home/kevin/maml_imitation_private/data/checkpoints/sawyer_pick_and_place_noisy_larger_single.xavier_init.4_conv.2_strides.32_5x5_filters.bt_dim_20.mbs_1.ubs_1.meta_lr_0.001.numstep_1.updatelr_0.0.3_fc.100_dim.conv_bt.fp.learn_ee_pos/model_42000.meta'
# LOG_DIR = '/home/kevin/maml_imitation_private/data/checkpoints/sawyer_pick_and_place_noisy_larger_single.xavier_init.4_conv.2_strides.32_5x5_filters.bt_dim_20.mbs_1.ubs_1.meta_lr_0.001.numstep_1.updatelr_0.0.3_fc.100_dim.conv_bt.fp.learn_ee_pos/'
# META_PATH = '/home/kevin/maml_imitation_private/data/checkpoints/sawyer_pick_and_place_noisy_larger_textures_goalpos_push.xavier_init.4_conv.2_strides.24_5x5_filters.bt_dim_20.mbs_10.ubs_1.meta_lr_0.001.numstep_1.updatelr_0.01.3_fc.200_dim.conv_bt.all_fc_bt.fp.learn_ee_pos.two_heads.final_ee_two_heads/model_91000.meta'
# LOG_DIR = '/home/kevin/maml_imitation_private/data/checkpoints/sawyer_pick_and_place_noisy_larger_textures_goalpos_push.xavier_init.4_conv.2_strides.24_5x5_filters.bt_dim_20.mbs_10.ubs_1.meta_lr_0.001.numstep_1.updatelr_0.01.3_fc.200_dim.conv_bt.all_fc_bt.fp.learn_ee_pos.two_heads.final_ee_two_heads/'
# META_PATH = '/home/kevin/maml_imitation_private/data/checkpoints/sawyer_pick_and_place_noisy_larger_textures_goalpos_push.xavier_init.4_conv.2_strides.24_5x5_filters.bt_dim_20.mbs_10.ubs_1.meta_lr_0.001.numstep_1.updatelr_0.05.3_fc.200_dim.conv_bt.all_fc_bt.fp.learn_ee_pos.two_heads.final_ee_two_heads/model_79000.meta'
# LOG_DIR = '/home/kevin/maml_imitation_private/data/checkpoints/sawyer_pick_and_place_noisy_larger_textures_goalpos_push.xavier_init.4_conv.2_strides.24_5x5_filters.bt_dim_20.mbs_10.ubs_1.meta_lr_0.001.numstep_1.updatelr_0.05.3_fc.200_dim.conv_bt.all_fc_bt.fp.learn_ee_pos.two_heads.final_ee_two_heads/'
# META_PATH = '/home/kevin/maml_imitation_private/data/checkpoints/sawyer_pick_and_place_textures_goalpos_push_initobj.xavier_init.4_conv.2_strides.24_5x5_filters.bt_dim_20.mbs_10.ubs_1.meta_lr_0.001.numstep_1.updatelr_0.05.3_fc.400_dim.fp.learn_ee_pos.eps_10.two_heads.final_ee_two_heads/model_50000.meta'
# LOG_DIR = '/home/kevin/maml_imitation_private/data/checkpoints/sawyer_pick_and_place_textures_goalpos_push_initobj.xavier_init.4_conv.2_strides.24_5x5_filters.bt_dim_20.mbs_10.ubs_1.meta_lr_0.001.numstep_1.updatelr_0.05.3_fc.400_dim.fp.learn_ee_pos.eps_10.two_heads.final_ee_two_heads/'
META_PATH = '/home/kevin/maml_imitation_private/data/checkpoints/sawyer_pick_and_place_textures_goalpos_push_initobj_4.xavier_init.4_conv.2_strides.24_5x5_filters.bt_dim_20.mbs_10.ubs_1.meta_lr_0.001.numstep_1.updatelr_0.05.3_fc.200_dim.fp.learn_ee_pos.eps_10.two_heads.final_ee_two_heads/model_74000.meta'
LOG_DIR = '/home/kevin/maml_imitation_private/data/checkpoints/sawyer_pick_and_place_textures_goalpos_push_initobj_4.xavier_init.4_conv.2_strides.24_5x5_filters.bt_dim_20.mbs_10.ubs_1.meta_lr_0.001.numstep_1.updatelr_0.05.3_fc.200_dim.fp.learn_ee_pos.eps_10.two_heads.final_ee_two_heads/'
# META_PATH = '/home/kevin/maml_imitation_private/data/checkpoints/sawyer_pick_and_place_textures_goalpos_push_initobj.xavier_init.4_conv.2_strides.24_5x5_filters.bt_dim_20.mbs_10.ubs_1.meta_lr_0.001.numstep_1.updatelr_0.05.3_fc.200_dim.fp.learn_ee_pos.eps_10.two_heads.1d_conv_act_3_32_ee_3_32_10x1_filters/model_98000.meta'
# LOG_DIR = '/home/kevin/maml_imitation_private/data/checkpoints/sawyer_pick_and_place_textures_goalpos_push_initobj.xavier_init.4_conv.2_strides.24_5x5_filters.bt_dim_20.mbs_10.ubs_1.meta_lr_0.001.numstep_1.updatelr_0.05.3_fc.200_dim.fp.learn_ee_pos.eps_10.two_heads.1d_conv_act_3_32_ee_3_32_10x1_filters/'
# META_PATH = '/home/kevin/maml_imitation_private/data/checkpoints/sawyer_pick_and_place_noisy_larger_textures_goalpos_push_initobj.xavier_init.lstm_size_512.4_conv.2_strides.24_5x5_filters.mbs_10.3_fc.200_dim.conv_bt.all_fc_bt.fp.learn_ee_pos/model_74000.meta'
# LOG_DIR = '/home/kevin/maml_imitation_private/data/checkpoints/sawyer_pick_and_place_noisy_larger_textures_goalpos_push_initobj.xavier_init.lstm_size_512.4_conv.2_strides.24_5x5_filters.mbs_10.3_fc.200_dim.conv_bt.all_fc_bt.fp.learn_ee_pos/'

MAX_PATH_LENGTH = 440#330#440

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

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
		obs = obs.reshape((1,1,3))
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
	
			obs = obs.reshape((1,1,3))
		else:
			T, _, _, _ = image.shape
			image = np.expand_dims(image, 0).transpose(0,1,4,3,2).astype('float32') / 255.0
			image = image.reshape((1, T, -1))
	
			obs = obs.reshape((1,T,3))

		action = self.sess.run(self.feed_dict['output'],
						   {self.feed_dict['statea']: self.demoX.dot(self.scale) + self.bias,
						   self.feed_dict['obsa']: self.demoVideo,
						   self.feed_dict['actiona']: self.demoU,
						   self.feed_dict['stateb']: obs.dot(self.scale) + self.bias,
						   self.feed_dict['obsb']: image})
		if T > 1:
			action = np.squeeze(action)[t]
		return np.squeeze(action), dict()
	
	def get_final_eept(self, image, obs, t=-1):
		# if CROP:
		#     image = np.array(Image.fromarray(image).crop((40,25,120,90)))
		
		T = 1
		if len(image.shape) == 3:
			image = np.expand_dims(image, 0).transpose(0,3,2,1).astype('float32') / 255.0
			image = image.reshape((1, 1, -1))
	
			obs = obs.reshape((1,1,3))
		else:
			T, _, _, _ = image.shape
			image = np.expand_dims(image, 0).transpose(0,1,4,3,2).astype('float32') / 255.0
			image = image.reshape((1, T, -1))
	
			obs = obs.reshape((1,T,3))

		final_eept = self.sess.run(self.feed_dict['final_eept'],
						   {self.feed_dict['statea']: self.demoX.dot(self.scale) + self.bias,
						   self.feed_dict['obsa']: self.demoVideo,
						   self.feed_dict['actiona']: self.demoU,
						   self.feed_dict['stateb']: obs.dot(self.scale) + self.bias,
						   self.feed_dict['obsb']: image})
		if T > 1:
			action = np.squeeze(action)[t]
		return np.squeeze(final_eept)

def find_xml_filepath(demo_info):
	xml_filepath = demo_info['xml']
	# # suffixs = [xml_path[xml_path.index('test_'):] for xml_path in xml_filepath]
	# suffixs = [xml_path[xml_path.index('train_'):] for xml_path in xml_filepath]
	# prefix = XML_PATH
	# xml_filepath = [str(prefix + suffix) for suffix in suffixs]
	# import pdb; pdb.set_trace()
	return xml_filepath
	
def load_env(xml_file=None, **kwargs):
	if xml_file is None:
		xml_file = '/home/kevin/multiworld/multiworld/envs/assets/sawyer_xyz/sawyer_pick_and_place_test_distr.xml'
	# xml_file = '/home/kevin/multiworld/multiworld/envs/assets/sawyer_xyz/sawyer_pick_and_place/4distr_train_1.xml'
	# xml_file = '/home/kevin/multiworld/multiworld/envs/assets/sawyer_xyz/sawyer_pick_and_place/4distr_test_506.xml'
	if '4distr' in xml_file:
		n_distractors = 4
		hand_pos_is_init = False#True
	elif '3distr' in xml_file:
		n_distractors = 3
		hand_pos_is_init = False
	else:
		n_distractors = 2
		hand_pos_is_init = False
	baseEnv = SawyerPickPlaceMILVisionSeqEnv(xml_file=xml_file, include_distractors=True, n_distractors=n_distractors, random_hand_init_pos=False, hand_pos_is_init=hand_pos_is_init, **kwargs)
	env = FlatGoalEnv(baseEnv, obs_keys=['state_observation', 'desired_goal'])
	return env

def load_demo(task_id, demo_dir, demo_inds):
	demo_info = pickle.load(open(demo_dir+task_id+'.pkl', 'rb'))
	demoX = demo_info['demoX'][demo_inds,:,:]
	demoU = demo_info['demoU'][demo_inds,:,:]
	d1, d2, _ = demoX.shape
	demoX = np.reshape(demoX, [1, d1*d2, -1])
	demoU = np.reshape(demoU, [1, d1*d2, -1])
	
	# init_objPos = demo_info['init_objPos'][demo_inds,:-1]
	# init_objPos = np.tile(init_objPos.reshape(1, 1, -1), (1, d1*d2, 1))
	# demoU = np.concatenate((demoU, init_objPos), axis=-1)
	# read in demo video
	# demo_gifs = [imageio.mimread(demo_dir+'object_'+task_id+'/cond%d.samp0.gif' % (demo_ind+180)) for demo_ind in demo_inds]
	# print([demo_dir+'object_'+task_id+'/cond%d.samp0.gif' % (demo_ind+180) for demo_ind in demo_inds])
	demo_gifs = [imageio.mimread(demo_dir+'object_'+task_id+'/cond%d.samp0.gif' % demo_ind) for demo_ind in demo_inds]
	print([demo_dir+'object_'+task_id+'/cond%d.samp0.gif' % demo_ind for demo_ind in demo_inds])
	return demoX, demoU, demo_gifs, demo_info

def eval_success(path, env):
	# if np.abs(env.get_goal_pos()[0] - env.get_push_goal_pos()[0]) < 0.15 and \
	# 	np.abs(env.get_obj_pos()[0] - env.get_goal_pos()[0]) < 0.1 and \
	# 	np.abs(env.get_obj_pos()[1] - env.get_goal_pos()[1]) < 0.1 and \
	# 	np.abs(env.get_distr_pos()[0][0] - env.get_goal_pos()[0]) < 0.1 and \
	# 	np.abs(env.get_distr_pos()[0][1] - env.get_goal_pos()[1]) < 0.1 and \
	# 	np.abs(env.get_distr_pos()[1][0] - env.get_goal_pos()[0]) < 0.1 and \
	# 	np.abs(env.get_distr_pos()[1][1] - env.get_goal_pos()[1]) < 0.1:
	# if np.abs(env.get_goal_pos()[0] - env.get_push_goal_pos()[0]) < 0.15 and \
	# 	np.abs(env.get_obj_pos()[0] - env.get_goal_pos()[0]) < 0.1 and \
	# 	np.abs(env.get_obj_pos()[1] - env.get_goal_pos()[1]) < 0.1:
	# if np.abs(env.get_goal_pos()[0] - env.get_push_goal_pos()[0]) < 0.15 and \
	# 	np.abs(env.get_distr_pos()[0][0] - env.get_goal_pos()[0]) < 0.1 and \
	# 	np.abs(env.get_distr_pos()[0][1] - env.get_goal_pos()[1]) < 0.1:
	if np.abs(env.get_goal_pos()[0] - env.get_push_goal_pos()[0]) < 0.15 and \
		np.abs(env.get_distr_pos()[1][0] - env.get_goal_pos()[0]) < 0.1 and \
		np.abs(env.get_distr_pos()[1][1] - env.get_goal_pos()[1]) < 0.1:
		return True
	else:
		return False
	

def main(meta_path, demo_dir, log_dir, test=True, window_size=135,
		run_steps=135, save_video=True, lstm=False, num_input_demos=1,
		get_fp=False, downsample=False, upsample=False, pred_phase=False):
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
			'output': tf.get_default_graph().get_tensor_by_name('output_action:0'),  
			# 'final_eept': tf.get_default_graph().get_tensor_by_name('final_eept:0'),  
		}

		scale_file = SCALE_FILE_PATH
		files = glob.glob(os.path.join(demo_dir, '*.pkl'))
		all_ids = [int(f.split('/')[-1][:-4]) for f in files]
		all_ids.sort()
		num_success = 0
		num_trials = 0
		trials_per_task = 2#3 #10 #3
		
		task_ids = all_ids[:5]#[1:5]#[1:2]#[1:]#[1:5]
	
		for task_id in task_ids:
			demo_inds = [0] #[4] # for consistency of comparison
			assert len(demo_inds) == num_input_demos
			demoX, demoU, demo_gifs, demo_info = load_demo(str(task_id), demo_dir, demo_inds)
	
			# load xml file
			xml_filepath = find_xml_filepath(demo_info)
			print(xml_filepath)
			# env = load_env(demo_info)
	
			policy = TFAgent(feed_dict, scale_file, sess)
			demo_data = (demo_gifs, demoX, demoU)
			returns = []
			gif_dir = log_dir + '/evaluated_gifs/'
			gif_path = Path(gif_dir)
			gif_path.mkdir_p()
			demo_range = range(1, trials_per_task+1)
# 			import pdb; pdb.set_trace()
			for j in demo_range:
			# for j in range(4, 4+trials_per_task+1):
				init_objPos, init_distrPos = list(demo_info['init_objPos'][j]), list(demo_info['init_distrPos'][j])
				distrPos0 = demo_info['demoU'][j, 110, -2:]
				distrPos1 = demo_info['demoU'][j, 220, -2:]
				print('Actual final eept obj is', init_objPos)
				print('Actual final eept distr 0 is', distrPos0)
				print('Actual final eept distr 1 is', distrPos1)
				env = load_env(xml_filepath, obj_init_pos=init_objPos, distr_init_pos=init_distrPos, random_reset=False)
				# video_suffix = gif_dir + str(id) + 'demo_' + str(num_input_demos) + '_' + str(len(returns)) + '.gif'
				# video_suffix = gif_dir + str(id) + 'demo_' + str(num_input_demos) + '_' + str(len(returns)) + '.mp4'
				video_suffix = gif_dir + str(id) + 'demo_' + str(num_input_demos) + '_' + str(len(returns)) + '.mp4'
				path = rollout_sliding_window(env, policy, max_path_length=MAX_PATH_LENGTH, env_reset=True,
							   animated=True, speedup=1, always_return_paths=True, 
							   save_video=save_video, video_filename=video_suffix, demo=demo_data,
							   window_size=window_size, run_steps=run_steps,
							   vision=True, lstm=lstm, is_sawyer=True)
				env.wrapped_env.close()
				num_trials += 1
				if eval_success(path, env):
					num_success += 1
				# print('Final eept obj loss is', (np.abs(init_objPos[:-1] - path['final_eept_list'][0]) + 0.01*((init_objPos[:-1] - path['final_eept_list'][0])**2)).mean())
				# print('Final eept distr0 loss is', (np.abs(distrPos0 - path['final_eept_list'][1]) + 0.01*((distrPos0 - path['final_eept_list'][1])**2)).mean())
				# print('Final eept distr1 loss is', (np.abs(distrPos1 - path['final_eept_list'][2]) + 0.01*((distrPos1 - path['final_eept_list'][2])**2)).mean())
				print('Return: '+str(path['rewards'].sum()))
				returns.append(path['rewards'].sum())
				print('Average Return so far: ' + str(np.mean(returns)))
				print('Success Rate so far: ' + str(float(num_success)/num_trials))
				sys.stdout.flush()
				if len(returns) > trials_per_task:
					break
	success_rate_msg = "Final success rate is %.5f" % (float(num_success)/num_trials)
	with open(log_dir + '/log_sawyer_pick_and_place.txt', 'a') as f:
		f.write(meta_path[:-5] + ':\n')
		f.write(success_rate_msg + '\n')

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--lstm', type=str2bool, default=False)
	parser.add_argument('--window_size', type=int, default=135)
	parser.add_argument('--run_steps', type=int, default=135)
	parser.add_argument('--test', action='store_false')
	parser.add_argument('--save_video', action='store_true')
	parser.add_argument('--pred_phase', action='store_true')
	parser.add_argument('--pred_demo_phase', action='store_true')
	args = parser.parse_args()
	window_size = args.window_size
	run_steps = args.run_steps
	test = args.test
	save_video = args.save_video
	pred_phase = args.pred_phase
	lstm = args.lstm
	main(meta_path=META_PATH, 
		demo_dir=DEMO_DIR,
		log_dir=LOG_DIR, 
		test=test, 
		lstm=lstm,
		window_size=window_size,
		run_steps=run_steps,
		save_video=save_video,
		pred_phase=pred_phase)
