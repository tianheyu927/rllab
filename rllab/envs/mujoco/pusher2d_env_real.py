import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import logger
from rllab.misc.overrides import overrides
from PIL import Image
import pickle


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param


class PusherEnv2DReal(MujocoEnv, Serializable):

    FILE = '3link_gripper_push_2d_real.xml'

    def __init__(self, *args, **kwargs):
        self.frame_skip = 5
        if 'xml_file' in kwargs:
            self.__class__.FILE = kwargs['xml_file']
        if 'distractors' in kwargs:
            self.include_distractors = kwargs['distractors']
        else:
            self.include_distractors = False
        super(PusherEnv2DReal, self).__init__(*args, **kwargs)
        self.frame_skip = 5
        self.dist = []
        if not hasattr(self, 'qposes'):
            with open('/home/kevin/rllab/data/qpos.pkl', 'rb') as f:
                self.qposes = pickle.load(f)
        Serializable.__init__(self, *args, **kwargs)

    def get_current_obs(self):
        if self.iteration <= 75:
            pgoal = self.get_body_com("goal")
            pdistr = self.get_body_com("distractor").copy()
            ptip = self.get_body_com("distal_4").copy()
            pobj = self.get_body_com("object").copy()
            # pdistr[0] = max(pdistr[0], pobj[0]) + 0.2
            # if ptip[1] >= pdistr[1]:
            #     pdistr_up[1] += 0.2
            # else:
            # pdistr[0] += 0.2
            # pdistr[1] = pgoal[1]
            # pdistr[0] = ptip[0]
            # pdistr[1] += 0.2
            pdistr[:-1] += 0.2
            return np.concatenate([
            self.model.data.qpos.flat[:-6],
            self.model.data.qvel.flat[:-6],
            self.get_body_com("distal_4"),
            self.get_body_com("object"),
            pdistr,
            self.get_body_com("goal"),
        ])
        return np.concatenate([
            self.model.data.qpos.flat[:-6],
            self.model.data.qvel.flat[:-6],
            self.get_body_com("distal_4"),
            self.get_body_com("object"),
            self.get_body_com("distractor"),
            self.get_body_com("goal"),
        ])
        # return np.concatenate([
        #     self.model.data.qpos.flat[:-6],
        #     self.model.data.qvel.flat[:-6],
        #     self.get_body_com("distal_4"),
        #     self.get_body_com("distractor"),
        #     self.get_body_com("object"),
        #     self.get_body_com("goal"),
        # ])
        
    def get_current_image_obs(self):
        image = self.viewer.get_image()
        pil_image = Image.frombytes('RGB', (image[1], image[2]), image[0])
        # pil_image = pil_image.resize((125,125), Image.ANTIALIAS)
        pil_image = pil_image.resize((100,100), Image.ANTIALIAS)
        image = np.flipud(np.array(pil_image))
        return image, np.concatenate([
            self.model.data.qpos.flat[:-6],
            self.model.data.qvel.flat[:-6],
            self.get_body_com("distal_4"),
            self.get_body_com('goal'),
            ])
    
    def getcolor(self):
        color = np.random.uniform(low=0, high=1, size=3)
        while np.linalg.norm(color - np.array([1.,0.,0.])) < 0.5:
            color = np.random.uniform(low=0, high=1, size=3)
        if self.include_distractors:
            distractor_color = np.random.uniform(low=0, high=1, size=3)
            while np.linalg.norm(distractor_color - np.array([1.,0.,0.])) < 0.5 and \
                    np.linalg.norm(distractor_color - color) < 1.0:
                distractor_color = np.random.uniform(low=0, high=1, size=3)
            return np.concatenate((color, [1.0], distractor_color, [1.0]))
        else:
            return np.concatenate((color, [1.0]))

    #def get_body_xmat(self, body_name):
    #    idx = self.model.body_names.index(body_name)
    #    return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def step(self, action):
        if not hasattr(self, "iteration"):
            self.iteration = 0
            import pdb; pdb.set_trace()
            self.init_pos = self.get_body_com("distal_4")
        self.frame_skip = 5
        pobj = self.get_body_com("object")
        pdistr = self.get_body_com("distractor")
        pgoal = self.get_body_com("goal")
        ptip = self.get_body_com("distal_4")
        reward_ctrl = - np.square(action).sum()
        pdistr_up = pdistr.copy() #pobj.copy()
        # pdistr_up[0] = ptip[0]
        # pdistr_up[1] += 0.2
        pdistr_up[:-1] += 0.2
        if self.iteration <= 75:
            reward_near = - np.linalg.norm(pdistr_up-ptip) #push distractor
        else:
            reward_near = - np.linalg.norm(pdistr-ptip) #push distractor
        reward_dist = - np.linalg.norm(pgoal-pobj)
        reward_dist_distr = - np.linalg.norm(pgoal-pdistr)
        # reward_near = - np.linalg.norm(pobj-ptip) # push object!!!
        # reward_return = - np.linalg.norm(self.init_pos-ptip)
        # for second policy
        # reward = reward_dist + reward_dist_distr + 0.1 * reward_ctrl + 0.5 * reward_near
        # for starting from goal position
        reward = reward_dist_distr + 0.1 * reward_ctrl + 0.5 * reward_near
        # reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        self.forward_dynamics(action) # TODO - frame skip
        next_obs = self.get_current_obs()

        done = False
        self.iteration += 1
        return Step(next_obs, reward, done)

    @overrides
    def reset(self, init_state=None):
        self.iteration = 0
        self.init_pos = self.get_body_com("distal_4")
        # qpos = np.random.uniform(low=-0.1, high=0.1, size=self.model.nq) + np.squeeze(self.init_qpos)
        qpos = np.squeeze(self.init_qpos.copy())
        while True:
            # TODO: seems like x, y are flipped in mujoco-py?
            object_ = [np.random.uniform(low=-1.1, high=-0.2),
                        np.random.uniform(low=0., high=0.4)]
            # for starting from the goal position
            # object_ = [np.random.uniform(low=-1.1, high=-0.8),
            #             np.random.uniform(low=0.0, high=0.4)]
            # goal = [-0.65, -0.4]
            goal = [-0.65, -0.25]
            # goal = [np.random.uniform(low=-1.2, high=-0.8),
            #              np.random.uniform(low=0.8, high=1.2)]
            if self.include_distractors:
                # distractor_ = [np.random.uniform(low=-0.8, high=-0.3),
                #                 np.random.uniform(low=-0.1, high=0.3)]
                # distractor_ = [np.random.uniform(low=-1.1, high=-0.2),
                #                 np.random.uniform(low=-0.1, high=0.4)]
                # distractor_ = [np.random.uniform(low=-1.1, high=-0.2),
                #                 np.random.uniform(low=0., high=0.4)]
                distractor_ = [np.random.uniform(low=-1.1, high=-0.8),
                                np.random.uniform(low=0.0, high=0.4)]
            if np.linalg.norm(np.array(object_)-np.array(goal)) > 0.3:
                if self.include_distractors: 
                    # if np.linalg.norm(np.array(object_)[0]-np.array(distractor_)[0]) > 0.2 and \
                    if np.linalg.norm(np.array(object_)[0]-np.array(distractor_)[0]) > 0.65 and \
                        np.linalg.norm(np.array(distractor_)-np.array(goal)) > 0.3 and \
                        np.linalg.norm(np.array(object_)[1]-np.array(distractor_)[1]) > 0.2 and \
                        np.array(object_)[1] < np.array(distractor_)[1] and \
                        np.array(object_)[0] > np.array(distractor_)[0]: # # for the second policy pushing from outer to inner and for the second policy
                        break
                else:
                    break
        # object_ = [np.random.uniform(low=-0.4, high=-0.2),
        #             np.random.uniform(low=-0.1, high=0.1)]
        # if np.random.random() > 0.5:
        #     tmp = object_.copy()
        #     object_ = distractor_.copy()
        #     distractor_ = tmp
        self.object = np.array(object_)
        # for the second policy!!!
        # self.object = np.array([-0.35, -0.2])
        # self.object = np.array(goal)
        # self.object = [np.random.uniform(low=-0.3, high=-0.2),
        #             np.random.uniform(low=-0.2, high=-0.1)]
        self.goal = np.array(goal)
        if self.include_distractors:
            self.distractor = np.array(distractor_)
        if hasattr(self, "_kwargs") and 'goal' in self._kwargs:
            self.object = np.array(self._kwargs['object'])
            self.goal = np.array(self._kwargs['goal'])
        
        #initialize at the goal position
        if not hasattr(self, 'qposes'):
            # with open('/home/kevin/rllab/data/qpos_outer.pkl', 'rb') as f:
            with open('/home/kevin/rllab/data/qpos_inner.pkl', 'rb') as f:
                self.qposes = pickle.load(f)
        qpos[:3] = self.qposes[np.random.choice(range(self.qposes.shape[0]))]
        
        if self.include_distractors:
            qpos[-6:-4] = self.distractor
        qpos[-4:-2] = self.object
        # qpos[-2:] = self.goal
        qvel = np.squeeze(self.init_qvel.copy())
        qvel[-4:] = 0
        if self.include_distractors:
            qvel[-6:-4] = 0
        setattr(self.model.data, 'qpos', qpos)
        setattr(self.model.data, 'qvel', qvel)
        self.model.data.qvel = qvel
        self.model._compute_subtree()
        self.model.forward()

        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()

    @overrides
    def log_diagnostics(self, paths):
        pass