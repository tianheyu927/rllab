import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import logger
from rllab.misc.overrides import overrides
from PIL import Image


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param


class PusherEnvVision2D(MujocoEnv, Serializable):

    FILE = '3link_gripper_push_2d.xml'

    def __init__(self, *args, **kwargs):
        self.frame_skip = 5
        if 'xml_file' in kwargs:
            self.__class__.FILE = kwargs['xml_file']
        if 'distractors' in kwargs:
            self.include_distractors = kwargs['distractors']
        else:
            self.include_distractors = False
        super(PusherEnvVision2D, self).__init__(*args, **kwargs)
        self.frame_skip = 5
        self.dist = []
        Serializable.__init__(self, *args, **kwargs)
    
    def adjust_viewer(self):
        viewer = self.get_viewer()
        viewer.autoscale()
        viewer.cam.trackbodyid=0
        # viewer.cam.distance = 4.0
        viewer.cam.distance = 4.0
        rotation_angle = 0
        cam_dist = 2 #4
        # cam_pos = np.array([0, 0, 0, cam_dist, -90, rotation_angle])
        cam_pos = np.array([0.2, -0.4, 0, cam_dist, -90, rotation_angle])
        for i in range(3):
            viewer.cam.lookat[i] = cam_pos[i]
        viewer.cam.distance = cam_pos[3]
        viewer.cam.elevation = cam_pos[4]
        viewer.cam.azimuth = cam_pos[5]
        viewer.cam.trackbodyid=-1

    def get_current_obs(self):
        # if self.iteration >= 100 and self.iteration < 130:#130:
        if self.iteration >= 100 and self.iteration < 130:#130:
            return np.concatenate([
                self.model.data.qpos.flat[:-6],
                self.model.data.qvel.flat[:-6],
                self.get_body_com("distal_4"),
                self.get_body_com("distractor"),
                self.init_pos,
                # self.get_body_com("object"),
                self.get_body_com("goal"),
            ])
        # elif self.iteration >= 130:#130:
        elif self.iteration >= 130:#130:
            return np.concatenate([
                self.model.data.qpos.flat[:-6],
                self.model.data.qvel.flat[:-6],
                self.get_body_com("distal_4"),
                self.get_body_com("object"),
                self.get_body_com("distractor"),
                self.get_body_com("goal"),
            ])
        return np.concatenate([
            self.model.data.qpos.flat[:-6],
            self.model.data.qvel.flat[:-6],
            self.get_body_com("distal_4"),
            self.get_body_com("distractor"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
        ])
        # return np.concatenate([
        #     self.model.data.qpos.flat[:-6],
        #     self.model.data.qvel.flat[:-6],
        #     self.get_body_com("distal_4"),
        #     self.get_body_com("object"),
        #     self.get_body_com("distractor"),
        #     self.get_body_com("goal"),
        # ])
    
    def get_current_obs_true(self):
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
                    np.linalg.norm(distractor_color - color) < 0.5:
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
            self.init_pos = self.get_body_com("distal_4")
        self.frame_skip = 5
        pobj = self.get_body_com("object")
        pdistr = self.get_body_com("distractor")
        pgoal = self.get_body_com("goal")
        ptip = self.get_body_com("distal_4")
        reward_ctrl = - np.square(action).sum()
        # if self.iteration >= 100 and self.iteration < 130:# and np.mean(self.dist[-10:]) <= 0.017:
        if self.iteration >= 100 and self.iteration < 130:# and np.mean(self.dist[-10:]) <= 0.017:
            # print('going back!')
            reward_dist = - np.linalg.norm(self.init_pos[:-1]-ptip[:-1])
            reward_distr = np.linalg.norm(pdistr[:-1]-ptip[:-1])
            # reward = reward_dist + 0.01 * reward_ctrl# + 0.1 * reward_distr
            reward = 1.2 * reward_dist + 0.1 * reward_ctrl + 0.1 * reward_distr# + 0.1 * reward_distr
            # reward = reward_dist + 0.1 * reward_ctrl + 0.1 * reward_distr# + 0.1 * reward_distr
        # elif self.iteration >= 130:
        elif self.iteration >= 130:
            reward_dist = - np.linalg.norm(pgoal[:-1]-pobj[:-1])
            reward_dist_distr = - np.linalg.norm(pgoal[:-1]-pdistr[:-1])
            reward_near = - np.linalg.norm(pdistr[:-1]-ptip[:-1])
            # reward_return = - np.linalg.norm(self.init_pos-ptip)
            reward = reward_dist + reward_dist_distr + 0.1 * reward_ctrl + 0.5 * reward_near
        else:
            reward_dist = - np.linalg.norm(pgoal[:-1]-pobj[:-1])
            reward_near = - np.linalg.norm(pobj[:-1] - ptip[:-1])
            # reward_return = - np.linalg.norm(self.init_pos-ptip)
            self.dist.append(-reward_dist)
            reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near
        self.forward_dynamics(action) # TODO - frame skip
        next_obs = self.get_current_obs()

        done = False
        self.iteration += 1
        return Step(next_obs, reward, done)

    @overrides
    def reset(self, init_state=None):
        self.iteration = 0
        self.init_pos = self.get_body_com("distal_4")
        self.reset_mujoco(init_state)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()

    @overrides
    def log_diagnostics(self, paths):
        pass