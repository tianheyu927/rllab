import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import logger
from rllab.misc.overrides import overrides
from PIL import Image


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param


class PusherEnv2D(MujocoEnv, Serializable):

    FILE = '3link_gripper_push_2d.xml'

    def __init__(self, *args, **kwargs):
        self.frame_skip = 5
        if 'xml_file' in kwargs:
            self.__class__.FILE = kwargs['xml_file']
        if 'distractors' in kwargs:
            self.include_distractors = kwargs['distractors']
        else:
            self.include_distractors = False
        super(PusherEnv2D, self).__init__(*args, **kwargs)
        self.frame_skip = 5
        self.dist = []
        Serializable.__init__(self, *args, **kwargs)

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:-6],
            self.model.data.qvel.flat[:-6],
            self.get_body_com("distal_4"),
            self.get_body_com("distractor"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])
    
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
        if not hasattr(self, "iter"):
            self.iteration = 0
            self.init_pos = self.get_body_com("distal_4")
        self.frame_skip = 5
        pobj = self.get_body_com("object")
        pgoal = self.get_body_com("goal")
        ptip = self.get_body_com("distal_4")
        reward_ctrl = - np.square(action).sum()
        if self.iteration >= 100:# and np.mean(self.dist[-3:]) <= 0.05:
            # print('going back!')
            reward_dist = - np.linalg.norm(self.init_pos-ptip)
            reward = reward_dist + 0.1 * reward_ctrl
        else:
            reward_dist = - np.linalg.norm(pgoal-pobj)
            reward_near = - np.linalg.norm(pobj - ptip)
            self.dist.append(-reward_dist)
            reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        self.forward_dynamics(action) # TODO - frame skip
        next_obs = self.get_current_obs()

        done = False
        self.iteration += 1
        return Step(next_obs, reward, done)

    @overrides
    def reset(self, init_state=None):
        self.itr = 0
        # qpos = np.random.uniform(low=-0.1, high=0.1, size=self.model.nq) + np.squeeze(self.init_qpos)
        qpos = np.squeeze(self.init_qpos.copy())
        while True:
            object_ = [np.random.uniform(low=-0.4, high=0.4),
                        np.random.uniform(low=-0.8, high=-0.4)]
            goal = [0., -1.2]
            # goal = [np.random.uniform(low=-1.2, high=-0.8),
            #              np.random.uniform(low=0.8, high=1.2)]
            if self.include_distractors:
                distractor_ = [np.random.uniform(low=-0.4, high=0.4),
                                np.random.uniform(low=-0.8, high=-0.4)]
            if np.linalg.norm(np.array(object_)-np.array(goal)) > 0.3:
                if self.include_distractors: 
                    if np.linalg.norm(np.array(object_)-np.array(distractor_)) > 0.5 and \
                        np.linalg.norm(np.array(distractor_)-np.array(goal)) > 0.3:
                        break
                else:
                    break
        self.object = np.array(object_)
        self.goal = np.array(goal)
        if self.include_distractors:
            self.distractor = np.array(distractor_)
        if hasattr(self, "_kwargs") and 'goal' in self._kwargs:
            self.object = np.array(self._kwargs['object'])
            self.goal = np.array(self._kwargs['goal'])

        # rgbatmp = np.copy(self.model.geom_rgba)
        # geompostemp = np.copy(self.model.geom_pos)
        # for body in range(len(geompostemp)):
        #     if 'object' in str(self.model.geom_names[body]):
        #         pos_x = np.random.uniform(low=-0.9, high=0.9)
        #         pos_y = np.random.uniform(low=0, high=1.0)
        #         rgba = self.getcolor()
        #         isinv = np.random.random()
        #         if isinv>0.5:
        #             rgba[-1] = 0.
        #         rgbatmp[body, :] = rgba
        #         geompostemp[body, 0] = pos_x
        #         geompostemp[body, 1] = pos_y

        # if hasattr(self, "_kwargs") and 'geoms' in self._kwargs:
        #     geoms = self._kwargs['geoms']
        #     ct = 0
        #     for body in range(len(geompostemp)):
        #         if 'object' in str(self.model.geom_names[body]):
        #             rgbatmp[body, :] = geoms[ct][0]
        #             geompostemp[body, 0] = geoms[ct][1]
        #             geompostemp[body, 1] = geoms[ct][2]
        #             ct += 1

        # self.model.geom_rgba = rgbatmp
        # self.model.geom_pos = geompostemp
        
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