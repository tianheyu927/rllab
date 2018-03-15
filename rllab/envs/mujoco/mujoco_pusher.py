import gym
import time
from contextlib import contextmanager
import random
import tempfile
import os
import numpy as np

from shutil import copyfile, copy2

RLLAB_PATH = '/home/kevin/rllab/vendor/mujoco_models/pusher2d_xmls/'

class MJCModel(object):
    def __init__(self, name):
        self.name = name
        self.root = MJCTreeNode("mujoco").add_attr('model', name)

    @contextmanager
    def asfile(self):
        """
        Usage:

        model = MJCModel('reacher')
        with model.asfile() as f:
            print f.read()  # prints a dump of the model

        """
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.xml', delete=True) as f:
            self.root.write(f)
            f.seek(0)
            yield f

    def open(self):
        self.file = tempfile.NamedTemporaryFile(mode='w+b', suffix='.xml', delete=True)
        self.root.write(self.file)
        self.file.seek(0)
        return self.file

    def save(self, path):
        with open(path, 'w') as f:
            self.root.write(f)

    def close(self):
        self.file.close()


class MJCModelRegen(MJCModel):
    def __init__(self, name, regen_fn):
        super(MJCModelRegen, self).__init__(name)
        self.regen_fn = regen_fn

    def regenerate(self):
        self.root = self.regen_fn().root



class MJCTreeNode(object):
    def __init__(self, name):
        self.name = name
        self.attrs = {}
        self.children = []

    def add_attr(self, key, value):
        if isinstance(value, str):  # should be basestring in python2
            pass
        elif isinstance(value, list) or isinstance(value, np.ndarray):
            value = ' '.join([str(val) for val in value])

        self.attrs[key] = value
        return self

    def __getattr__(self, name):
        def wrapper(**kwargs):
            newnode =  MJCTreeNode(name)
            for (k, v) in kwargs.items(): # iteritems in python2
                newnode.add_attr(k, v)
            self.children.append(newnode)
            return newnode
        return wrapper

    def dfs(self):
        yield self
        if self.children:
            for child in self.children:
                for node in child.dfs():
                    yield node

    def write(self, ostream, tabs=0):
        contents = ' '.join(['%s="%s"'%(k,v) for (k,v) in self.attrs.items()])
        if self.children:

            ostream.write('\t'*tabs)
            ostream.write('<%s %s>\n' % (self.name, contents))
            for child in self.children:
                child.write(ostream, tabs=tabs+1)
            ostream.write('\t'*tabs)
            ostream.write('</%s>\n' % self.name)
        else:
            ostream.write('\t'*tabs)
            ostream.write('<%s %s/>\n' % (self.name, contents))

    def __str__(self):
        s = "<"+self.name
        s += ' '.join(['%s="%s"'%(k,v) for (k,v) in self.attrs.items()])
        return s+">"

def pusher(object_pos=(0., 0., -0.1), goal_pos=(0., 0., -0.145), distractor_pos=(0.5,0.3,-0.1), object_color=(1., 0., 0., 1.), distractor_color=(0., 1., 0., 1.)):
    object_pos, goal_pos, distractor_pos, object_color, distractor_color = list(object_pos), list(goal_pos), list(distractor_pos), list(object_color), list(distractor_color)
    # For now, only supports one distractor

    mjcmodel = MJCModel('arm3d')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(timestep="0.01",gravity="0 0 0",iterations="20",integrator="Euler")
    default = mjcmodel.root.default()
    default.joint(armature='0.04', damping=1, limited='true')
    default.geom(friction="0.8 0.1 0.1",density="300",margin="0.002",condim="1",contype="1",conaffinity="1")

    # Make table
    worldbody = mjcmodel.root.worldbody()
    worldbody.light(diffuse=".5 .5 .5", pos="0 0 3", dir="0 0 -1")
    worldbody.geom(name="table", type="plane", pos="0 0.5 -0.15", size="2 2 0.1", contype="1", conaffinity="1")
    # Make arm
    palm = worldbody.body(name="palm", pos="0 0 0")
    palm.geom(rgba="0. 1. 0. 1", type="capsule", fromto="0 0 -0.1 0 0 0.1", size="0.12")
    proximal_1 = palm.body(name="proximal_1", pos="0 0 -0.075", axisangle="0 0 1 0.785")
    proximal_1.joint(name="proximal_j_1", type="hinge", pos="0 0 0", axis="0 0 1", range="-2.5 2.5", damping="1.0")
    proximal_1.geom(rgba="0. 1. 0. 1", type="capsule",  fromto="0 0 0 0.4 0 0", size="0.06", contype="1", conaffinity="1")
    distal_1 = proximal_1.body(name="distal_1", pos="0.4 0 0", axisangle="0 0 1 -0.785")
    distal_1.joint(name="distal_j_1", type="hinge", pos="0 0 0", axis="0 0 1", range="-2.3213 2.3", damping="1.0")
    distal_1.geom(rgba="0. 1. 0. 1", type="capsule",  fromto="0 0 0 0.4 0 0", size="0.06", contype="1", conaffinity="1")
    distal_2 = distal_1.body(name="distal_2", pos="0.4 0 0", axisangle="0 0 1 -1.57")
    distal_2.joint(name="distal_j_2", type="hinge", pos="0 0 0", axis="0 0 1", range="-2.3213 2.3", damping="1.0")
    distal_2.geom(rgba="0. 1. 0. 1", type="capsule", fromto="0 0 0 0.4 0 0", size="0.06", contype="1", conaffinity="1")
    distal_4 = distal_2.body(name="distal_4", pos="0.4 0 0")
    distal_4.site(name="tip arml", pos="0.1 -0.2 0", size="0.01")
    distal_4.site(name="tip armr", pos="0.1 0.2 0", size="0.01")
    distal_4.geom(rgba="0. 1. 0. 1", type="capsule", fromto="0 -0.2 0 0 0.2 0", size="0.04", contype="1", conaffinity="1")
    distal_4.geom(rgba="0. 1. 0. 1", type="capsule", fromto="0 -0.2 0 0.2 -0.2 0", size="0.04", contype="1", conaffinity="1")
    distal_4.geom(rgba="0. 1. 0. 1", type="capsule", fromto="0 0.2 0 0.2 0.2 0", size="0.04", contype="1", conaffinity="1")

    ## MAKE DISTRACTOR
    distractor = worldbody.body(name="distractor", pos=distractor_pos)
    distractor.geom(rgba=distractor_color, type="cylinder", size="0.1 0.1 0.1", density="0.00001", contype="1", conaffinity="1")
    distractor.joint(name="distractor_slidey", type="slide", pos="0.025 0.025 0.025", axis="0 1 0", range="-10.3213 10.3", damping="0.5")
    distractor.joint(name="distractor_slidex", type="slide", pos="0.025 0.025 0.025", axis="1 0 0", range="-10.3213 10.3", damping="0.5")

    # MAKE TARGET OBJECT
    obj = worldbody.body(name="object", pos=object_pos)
    obj.geom(rgba=object_color, type="cylinder", size="0.1 0.1 0.1", density="0.00001", contype="1", conaffinity="1")
    obj.joint(name="obj_slidey", type="slide", pos="0.025 0.025 0.025", axis="0 1 0", range="-10.3213 10.3", damping="0.5")
    obj.joint(name="obj_slidex", type="slide", pos="0.025 0.025 0.025", axis="1 0 0", range="-10.3213 10.3", damping="0.5")

    goal = worldbody.body(name="goal", pos=goal_pos)
    goal.geom(rgba="1 0 0 1", type="cylinder", size="0.17 0.005 0.2", density='0.00001', contype="0", conaffinity="0")
    goal.joint(name="goal_slidey", type="slide", pos="0 0 0", axis="0 1 0", range="-10.3213 10.3", damping="0.5")
    goal.joint(name="goal_slidex", type="slide", pos="0 0 0", axis="1 0 0", range="-10.3213 10.3", damping="0.5")

    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="proximal_j_1", ctrlrange="-3.0 3.0", ctrllimited="true")
    actuator.motor(joint="distal_j_1", ctrlrange="-3.0 3.0", ctrllimited="true")
    actuator.motor(joint="distal_j_2", ctrlrange="-3.0 3.0", ctrllimited="true")

    return mjcmodel

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # Could edit this to be the path to the object file instead
    parser.add_argument('--xml_filepath', type=str, default='None')
    parser.add_argument('--debug_log', type=str, default='None')
    args = parser.parse_args()

    if args.debug_log != 'None':
        with open(args.debug_log, 'r') as f:
            i = 0
            mass = None
            scale = None
            obj = None
            damp = None
            xml_file = None
            for line in f:
                if 'scale:' in line:
                    # scale
                    string = line[line.index('scale:'):]
                    scale = float(string[7:])
                if 'damp:' in line:
                    # damping
                    string = line[line.index('damp:'):]
                    damp = float(string[6:])
                if 'obj:' in line:
                    # obj
                    string = line[line.index('obj:'):]
                    string = string[string.index('rllab'):-1]
                    obj = '/home/cfinn/code/' + string
                if 'mass:' in line:
                    # mass
                    string = line[line.index('mass:'):]
                    mass = float(string[6:])
                if 'xml:' in line:
                    string = line[line.index('xml:'):]
                    xml_file = string[5:-1]
                    suffix = xml_file[xml_file.index('pusher'):]
                    xml_file = '/home/cfinn/code/rllab/vendor/local_mujoco_models/' + suffix
                if (mass and scale and obj and damp) or xml_file:
                    break
        if not xml_file:
            print(obj)
            print(scale)
            print(mass)
            print(damp)
            model = pusher(mesh_file=obj,mesh_file_path=obj, obj_scale=scale,obj_mass=mass,obj_damping=damp)
            model.save(GYM_PATH+'/gym/envs/mujoco/assets/pusher.xml')
        else:
            copyfile(xml_file, GYM_PATH + '/gym/envs/mujoco/assets/pusher.xml')
    else:
        # TODO - could call code to autogenerate xml file here
        goal_pos = (-1.0, 1.0, -0.145)
        for i in range(500):
            color = np.random.uniform(low=0, high=1, size=3)
            while np.linalg.norm(color - np.array([1.,0.,0.])) < 0.5:
                color = np.random.uniform(low=0, high=1, size=3)
            distractor_color = np.random.uniform(low=0, high=1, size=3)
            while np.linalg.norm(distractor_color - np.array([1.,0.,0.])) < 0.5 and \
                    np.linalg.norm(distractor_color - color) < 0.5:
                distractor_color = np.random.uniform(low=0, high=1, size=3)
            color = np.concatenate((color, [1.0]))
            distractor_color = np.concatenate((distractor_color, [1.0]))
            for j in range(24):
                while True:
                    object_ = [np.random.uniform(low=-1.0, high=-0.4),
                                 np.random.uniform(low=0.3, high=1.2)]
                    distractor_ = [np.random.uniform(low=-1.0, high=-0.4),
                             np.random.uniform(low=0.3, high=1.2)]
                    if np.linalg.norm(np.array(object_)-np.array(goal_pos)[:-1]) > 0.45 and \
                        np.linalg.norm(np.array(object_)-np.array(distractor_)) > 0.17 and \
                        np.linalg.norm(np.array(distractor_)-np.array(goal_pos)[:-1]) > 0.45:
                        break
                object_ = np.concatenate((object_, [-0.1]))
                distractor_ = np.concatenate((distractor_, [-0.1]))
                model1 = pusher(object_pos=object_, goal_pos=goal_pos, distractor_pos=distractor_, object_color=color, distractor_color=distractor_color)
                model2 = pusher(object_pos=distractor_, goal_pos=goal_pos, distractor_pos=object_, object_color=distractor_color, distractor_color=color)
                dir1 = RLLAB_PATH + 'train_%d.xml' % (24*2*i + j)
                dir2 = RLLAB_PATH + 'train_%d.xml' % (24*(2*i+1) + j)
                print('Saving xml to %s' % dir1)
                model1.save(dir1)
                print('Saving xml to %s' % dir2)
                model2.save(dir2)
