import gym
import time
from contextlib import contextmanager
import random
import tempfile
import os
import numpy as np
import stl
from stl import mesh

from shutil import copyfile, copy2

RLLAB_PATH = '/home/kevin/rllab/vendor/mujoco_models/pusher2d_xmls/'
RLLAB_SHORTER_PATH = '/home/kevin/rllab/vendor/mujoco_models/pusher2d_shorter_xmls/'
RLLAB_REAL_PATH = '/home/kevin/rllab/vendor/mujoco_models/pusher2d_real_xmls/'
GYM_PATH = '/home/kevin/gym/gym/envs/mujoco/assets/pusher2d_xmls/'
# RLLAB_MULTIGOAL_PATH = '/home/kevin/rllab/vendor/mujoco_models/pusher2d_multigoal_xmls/'
RLLAB_MULTIGOAL_PATH = '/home/kevin/rllab/vendor/mujoco_models/pusher2d_multigoal_xmls_real/'
MESH_PATH = '/home/kevin/gym/gym/envs/mujoco/assets/sim_push_xmls/mujoco_models/'
OBJ_TEXTURE_PATH = '/home/kevin/gym/gym/envs/mujoco/assets/sim_push_xmls/textures/obj_textures/'
CONFIG_XML = 'shared_config.xml'
BASE_XML = 'sawyer_xyz_base.xml'

# find the max dimensions, so we can know the bounding box, getting the height,
# width, length (because these are the step size)...
def find_mins_maxs(obj):
    minx = maxx = miny = maxy = minz = maxz = None
    for p in obj.points:
        # p contains (x, y, z)
        if minx is None:
            minx = p[stl.Dimension.X]
            maxx = p[stl.Dimension.X]
            miny = p[stl.Dimension.Y]
            maxy = p[stl.Dimension.Y]
            minz = p[stl.Dimension.Z]
            maxz = p[stl.Dimension.Z]
        else:
            maxx = max(p[stl.Dimension.X], maxx)
            minx = min(p[stl.Dimension.X], minx)
            maxy = max(p[stl.Dimension.Y], maxy)
            miny = min(p[stl.Dimension.Y], miny)
            maxz = max(p[stl.Dimension.Z], maxz)
            minz = min(p[stl.Dimension.Z], minz)
    return minx, maxx, miny, maxy, minz, maxz


class MJCModel(object):
    def __init__(self, name, include_config=False):
        self.name = name
        if not include_config:
            self.root = MJCTreeNode("mujoco").add_attr('model', name)
        else:
            self.root = MJCTreeNode("mujoco")
            
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
    def __init__(self, name, end_with_name=False):
        self.name = name
        self.attrs = {}
        self.children = []
        self.end_with_name = end_with_name

    def add_attr(self, key, value):
        if isinstance(value, str):  # should be basestring in python2
            pass
        elif isinstance(value, list) or isinstance(value, np.ndarray):
            value = ' '.join([str(val) for val in value])

        self.attrs[key] = value
        return self

    def __getattr__(self, name):
        def wrapper(**kwargs):
            if 'end_with_name' in kwargs.keys():
                end_with_name = kwargs.pop('end_with_name')
            else:
                end_with_name = False
            newnode =  MJCTreeNode(name, end_with_name=end_with_name)
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
            if self.end_with_name:
                ostream.write('<%s %s></%s>\n' % (self.name, contents, self.name))
            else:
                ostream.write('<%s %s/>\n' % (self.name, contents))

    def __str__(self):
        s = "<"+self.name
        s += ' '.join(['%s="%s"'%(k,v) for (k,v) in self.attrs.items()])
        return s+">"

def sawyer(obj_scale=None,
            obj_mass=None,
            obj_damping=None,
            object_pos=(-0.1, 0.6, 0.02),
            distr_scale=None, 
            distr_mass=None,
            distr_damping=None, 
            goal_pos=(0.0, 0.9, 0.02), 
            distractor_poses=[(0.5,0.5,0.02)], 
            mesh_file=None,
            mesh_file_path=None, 
            distractor_mesh_files=None, 
            friction=(2.0, 0.10, 0.002), 
            distractor_textures=None,
            obj_texture=None,
            config_xml=None,
            base_xml=None):
    object_pos, goal_pos, distractor_poses, friction = list(object_pos), list(goal_pos), [list(pos) for pos in distractor_poses], list(friction)
    n_distractors = len(distractor_poses)
    
    if obj_scale is None:
        obj_scale = random.uniform(0.2, 0.3)
    if obj_mass is None:
        obj_mass = random.uniform(0.1, 0.2)
    if obj_damping is None:
        obj_damping = 0.
    obj_damping = str(obj_damping)

    if distractor_mesh_files:
        if distr_scale is None:
            distr_scale = random.uniform(0.2, 0.3)
        if distr_mass is None:
            distr_mass = random.uniform(0.1, 0.2)
        if distr_damping is None:
            distr_damping = 0.
        distr_damping = str(distr_damping)

    # For now, only supports one distractor

    mjcmodel = MJCModel('sawyer', include_config=True)
    mjcmodel.root.include(file=config_xml, end_with_name=True)

    # Make base
    worldbody = mjcmodel.root.worldbody()
    worldbody.include(file=base_xml, end_with_name=True)
    
    # Process object physical properties
    if mesh_file is not None:
        mesh_object = mesh.Mesh.from_file(mesh_file)
        vol, cog, inertia = mesh_object.get_mass_properties()
        minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(mesh_object)
        max_length = max((maxx-minx),max((maxy-miny),(maxz-minz)))
        scale = obj_scale*0.0012 * (200.0 / max_length)
        object_density = obj_mass / (vol*scale*scale*scale)
        object_pos[0] -= scale*(minx+maxx)/2.0
        object_pos[1] -= scale*(miny+maxy)/2.0
        # object_pos[2] = -0.324 - scale*minz
        object_pos[2] = 0.02 - scale*minz
        object_scale = scale
    distr_densities, distr_scales = [], []
    if distractor_mesh_files is not None:
        for i in range(n_distractors):
            distr_mesh_object = mesh.Mesh.from_file(distractor_mesh_files[i])
            vol, cog, inertia = distr_mesh_object.get_mass_properties()
            minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(distr_mesh_object)
            max_length = max((maxx-minx),max((maxy-miny),(maxz-minz)))
            distr_scale = distr_scale*0.0012 * (200.0 / max_length)
            distr_density = distr_mass / (vol*distr_scale*distr_scale*distr_scale)
            distr_scales.append(distr_scale)
            distr_densities.append(distr_density)
            distractor_poses[i][0] -= distr_scale*(minx+maxx)/2.0
            distractor_poses[i][1] -= distr_scale*(miny+maxy)/2.0
            # distractor_poses[i][2] = -0.324 - distr_scale*minz
            distractor_poses[i][2] = 0.02 - distr_scale*minz

    # MAKE TARGET OBJECT
    obj = worldbody.body(name="obj", pos=object_pos)
    if mesh_file is None:
        obj.geom(name="objbox", type="sphere", pos="0 0 0",
                  size="0.02 0.02 0.02 0.02", rgba=".1 .1 .9 1", solimp="0.99 0.99 0.01", solref="0.01 1",
                  contype="1", conaffinity="1", friction="10.0 0.10 0.002", condim="4", mass=1.0)
    else:
        if obj_texture:
            obj.geom(material='object', conaffinity="1", contype="1", condim="4", solimp="0.99 0.99 0.01", solref="0.01 1", friction=friction, density=str(object_density), mesh="object_mesh", rgba="1 1 1 1", type="mesh")
        else:
            obj.geom(conaffinity="1", contype="1", condim="4", solimp="0.99 0.99 0.01", solref="0.01 1", friction=friction, density=str(object_density), mesh="object_mesh", rgba="1 1 1 1", type="mesh")
        obj.joint(name="objjoint", type="free", limited="false", damping=obj_damping, armature="0")
        obj.inertial(pos="0 0 0", mass=".1", diaginertia="100000 100000 100000")
        
    ## MAKE DISTRACTOR
    if distractor_mesh_files:
        for i in range(n_distractors):
            distractor = worldbody.body(name="distractor_%d" % i, pos=distractor_poses[i])
            if distractor_textures:
                distractor.geom(material='distractor_%d' % i, conaffinity="1", contype="1", condim="4", solimp="0.99 0.99 0.01", solref="0.01 1", friction=friction, density=str(distr_densities[i]), mesh="distractor_%d_mesh" % i, rgba="1 1 1 1", type="mesh")
            else:
                distractor.geom(conaffinity="1", contype="1", condim="4", solimp="0.99 0.99 0.01", solref="0.01 1", friction=friction, density=str(distr_densities[i]), mesh="distractor_%d_mesh" % i, rgba="1 1 1 1", type="mesh")
            distractor.joint(name="distractor_%d_joint" % i, type="free", limited="false", damping=distr_damping, armature="0")
            distractor.inertial(pos="0 0 0", mass=".1", diaginertia="100000 100000 100000")
    
    goal = worldbody.body(name="goal", pos=goal_pos)
    dragonball1 = goal.body(name="dragonball1", pos="0.075 0 0.06")
    dragonball1.geom(rgba="1 0 1 1", type="sphere", size="0.005 0.005 0.005")
    dragonball2 = goal.body(name="dragonball2", pos="-0.075 0 0.06")
    dragonball2.geom(rgba="1 0 1 1", type="sphere", size="0.005 0.005 0.005")
    goal.geom(name="goal_bottom", rgba="1 1 0 1", type="box", pos="0 0 0.005", size="0.075 0.075 0.001", contype="1", conaffinity="0", mass="1000")
    goal.geom(name="goal_wall1", rgba="1 1 1 1", type="box", pos="0.0 0.075 0.034", size="0.075 0.001 0.03", contype="0", conaffinity="0", mass="1000")
    goal.geom(name="goal_wall2", rgba="1 1 1 1", type="box", pos="0.0 -0.075 0.034", size="0.075 0.001 0.03", contype="0", conaffinity="0", mass="1000")
    goal.geom(name="goal_wall3", rgba="1 1 1 1", type="box", pos="0.075 0 0.034", size="0.001 0.075 0.03", contype="0", conaffinity="0", mass="1000")
    goal.geom(name="goal_wall4", rgba="1 1 1 1", type="box", pos="-0.076 0 0.034", size="0.001 0.075 0.03", contype="0", conaffinity="0", mass="1000")
    goal.geom(rgba="1 1 1 1", type="capsule", fromto="0.073 0.073 0.0075 0.073 0.073 0.06", size="0.005", contype="0", conaffinity="0")
    goal.geom(rgba="1 1 1 1", type="capsule", fromto="0.073 -0.073 0.0075 0.073 -0.073 0.06", size="0.005", contype="0", conaffinity="0")
    goal.geom(rgba="1 1 1 1", type="capsule", fromto="-0.073 0.073 0.0075 -0.073 0.073 0.06", size="0.005", contype="0", conaffinity="0")
    goal.geom(rgba="1 1 1 1", type="capsule", fromto="-0.073 -0.073 0.0075 -0.073 -0.073 0.06", size="0.005", contype="0", conaffinity="0")
    goal.joint(name="goal_slidey", type="slide", pos="0 0 0", axis="0 1 0", range="-10.3213 10.3", damping="1.0")
    goal.joint(name="goal_slidex", type="slide", pos="0 0 0", axis="1 0 0", range="-10.3213 10.3", damping="1.0")

    asset = mjcmodel.root.asset()
    asset.mesh(file=mesh_file_path, name="object_mesh", scale=[object_scale]*3) # figure out the proper scale
    if distractor_mesh_files:
        for i in range(n_distractors):
            asset.mesh(file=distractor_mesh_files[i], name="distractor_%d_mesh" % i, scale=[distr_scales[i]]*3)
            if distractor_textures:
                asset.texture(name='distractor_%d' % i, file=distractor_textures[i])
                asset.material(shininess='0.3', specular='1', name='distractor_%d' % i, rgba='0.9 0.9 0.9 1', texture='distractor_%d' % i)
    if obj_texture:
        asset.texture(name='object', file=obj_texture)
        asset.material(shininess='0.3', specular='1', name='object', rgba='0.9 0.9 0.9 1', texture='object')

    actuator = mjcmodel.root.actuator()
    actuator.position(ctrllimited="true", ctrlrange="-1 1", joint="r_close", kp="400",  user="1")
    actuator.position(ctrllimited="true", ctrlrange="-1 1", joint="l_close", kp="400",  user="1")

    return mjcmodel

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # Could edit this to be the path to the object file instead
    parser.add_argument('--xml_filepath', type=str, default='None')
    parser.add_argument('--debug_log', type=str, default='None')
    args = parser.parse_args()
    # model = sawyer(mesh_file=MESH_PATH+'fox.stl', mesh_file_path=MESH_PATH+'fox.stl', distractor_mesh_files=[MESH_PATH+'Keysafe.stl'],
    #                 obj_texture=OBJ_TEXTURE_PATH+'banded_0002.png', distractor_textures=[OBJ_TEXTURE_PATH+'banded_0004.png'],
    #                 config_xml=CONFIG_XML, base_xml=BASE_XML)
    # model.save('/home/kevin/multiworld/multiworld/envs/assets/sawyer_xyz/sawyer_pick_and_place_fox_keysafe.xml')
    model = sawyer(mesh_file=MESH_PATH+'vase1.stl', mesh_file_path=MESH_PATH+'vase1.stl', distractor_mesh_files=None,
                    obj_texture=OBJ_TEXTURE_PATH+'banded_0002.png',
                    config_xml=CONFIG_XML, base_xml=BASE_XML)
    model.save('/home/kevin/multiworld/multiworld/envs/assets/sawyer_xyz/sawyer_pick_and_place_vase1.xml')
    import pdb; pdb.set_trace()
    # model = pusher(object_pos=(0, 0, -0.1), goal_pos=(-0.25, -0.65, -0.145), distractor_pos=(0, 0, -0.1),
    #                 object_color=(0.6464944792711915, 0.8851453486090576, 0.9337627557555863, 1.0), 
    #                 distractor_color=(0.9950538605327119, 0.2617251721988596, 0.06634001915093157, 1.0),
    #                 table_texture=TABLE_TEXTURE_PATH)
    # model.save('/home/kevin/rllab/vendor/mujoco_models/3link_gripper_push_2d_real.xml')
    # import pdb; pdb.set_trace()
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
        np.random.seed(3) #0
        # goal_pos = (-0.4, -0.65, -0.145)
        goal_pos = (-0.25, -0.65, -0.145)
        train_colors = []
        colors = []
        # for i in range(500):
        # for i in range(50):
        # for i in range(100):
        for i in range(550):
            color = np.random.uniform(low=0, high=1, size=3)
            while np.linalg.norm(color - np.array([1.,0.,0.])) < 0.5 or (i >= 470 and color in train_colors):
                color = np.random.uniform(low=0, high=1, size=3)
            distractor_color = np.random.uniform(low=0, high=1, size=3)
            while (np.linalg.norm(distractor_color - np.array([1.,0.,0.])) < 0.5 and \
                    np.linalg.norm(distractor_color - color) < 1.2) or \
                    (i >= 470 and i < 500 and distractor_color in train_colors) or \
                    (i >= 500 and distractor_color in colors):
                distractor_color = np.random.uniform(low=0, high=1, size=3)
            color = np.concatenate((color, [1.0]))
            distractor_color = np.concatenate((distractor_color, [1.0]))
            if i < 470:
                train_colors.extend([color, distractor_color])
                # continue
            elif i >= 470 and i < 500:
                colors.extend([color, distractor_color])
                # continue

            for j in range(24):
            # for j in range(12):
                if j >= 12:
                    while True:
                        # object_ = [np.random.uniform(low=-0.1, high=0.3),
                        #             np.random.uniform(low=-0.9, high=-0.4)]
                        # distractor_ = [np.random.uniform(low=-0.1, high=0.3),
                        #             np.random.uniform(low=-0.9, high=-0.4)]
                        # object_ = [np.random.uniform(low=-0.1, high=0.4),
                        #             np.random.uniform(low=-1.0, high=-0.2)]
                        # distractor_ = [np.random.uniform(low=-0.1, high=0.4),
                        #             np.random.uniform(low=-1.0, high=-0.2)]
                        # object_ = [np.random.uniform(low=0.0, high=0.4),
                        #             np.random.uniform(low=-1.05, high=-0.3)]
                        # distractor_ = [np.random.uniform(low=0.0, high=0.4),
                        #             np.random.uniform(low=-1.05, high=-0.3)]
                        object_ = [np.random.uniform(low=0.0, high=0.4),
                                    np.random.uniform(low=-1.1, high=-0.8)]
                        distractor_ = [np.random.uniform(low=0.0, high=0.4),
                                        np.random.uniform(low=-1.1, high=-0.8)]
                        if np.linalg.norm(np.array(object_)-np.array(goal_pos)[:-1]) > 0.3 and \
                            np.linalg.norm(np.array(object_)[0]-np.array(distractor_)[0]) > 0.2 and \
                            np.linalg.norm(np.array(distractor_)-np.array(goal_pos)[:-1]) > 0.3 and \
                            np.linalg.norm(np.array(object_)[1]-np.array(distractor_)[1]) > 0.2 and \
                            ((j < 6 and object_[1] < distractor_[1]) or \
                            (j >= 6 and object_[1] >= distractor_[1])):
                        # if np.linalg.norm(np.array(object_)-np.array(goal_pos)[:-1]) > 0.3 and \
                        #     np.linalg.norm(np.array(object_)[-1]-np.array(distractor_)[-1]) > 0.45 and \
                        #     np.linalg.norm(np.array(distractor_)-np.array(goal_pos)[:-1]) > 0.3:# and \
                            # object_[1] > distractor_[1]:
                            break
                else:
                    while True:
                        object_ = [np.random.uniform(low=0.0, high=0.3),
                                    np.random.uniform(low=-1.0, high=-0.3)]
                        distractor_ = [np.random.uniform(low=0.0, high=0.3),
                                    np.random.uniform(low=-1.0, high=-0.3)]
                        if np.linalg.norm(np.array(object_)-np.array(goal_pos)[:-1]) > 0.3 and \
                            np.linalg.norm(np.array(object_)[-1]-np.array(distractor_)[-1]) > 0.55 and \
                            np.linalg.norm(np.array(object_)[0]-np.array(distractor_)[0]) > 0.2 and \
                            np.linalg.norm(np.array(distractor_)-np.array(goal_pos)[:-1]) > 0.3 and \
                            ((j < 18 and object_[1] > distractor_[1] and object_[0] < distractor_[0]) or \
                            (j >= 18 and object_[1] <= distractor_[1] and object_[0] > distractor_[0])):
                            # (object_[1] > distractor_[1] and object_[0] < distractor_[0]):
                        # if np.linalg.norm(np.array(object_)-np.array(goal_pos)[:-1]) > 0.3 and \
                        #     np.linalg.norm(np.array(object_)[-1]-np.array(distractor_)[-1]) > 0.45 and \
                        #     np.linalg.norm(np.array(distractor_)-np.array(goal_pos)[:-1]) > 0.3:# and \
                            # object_[1] > distractor_[1]:
                            break
                # object_, distractor_ = [0., 0.], [0., 0.]
                # while True:
                #     object_ = [np.random.uniform(low=0.0, high=0.4),
                #                 np.random.uniform(low=-1.1, high=-0.2)]
                #     distractor_ = [np.random.uniform(low=0.0, high=0.4),
                #                 np.random.uniform(low=-1.1, high=-0.2)]
                #     if np.linalg.norm(np.array(object_)-np.array(goal_pos)[:-1]) > 0.3 and \
                #         np.linalg.norm(np.array(object_)[-1]-np.array(distractor_)[-1]) > 0.5 and \
                #         np.linalg.norm(np.array(object_)[0]-np.array(distractor_)[0]) > 0.2 and \
                #         np.linalg.norm(np.array(distractor_)-np.array(goal_pos)[:-1]) > 0.3 and \
                #         (object_[1] > distractor_[1] and object_[0] < distractor_[0]):
                #         break
                object_ = np.concatenate((object_, [-0.1]))
                distractor_ = np.concatenate((distractor_, [-0.1]))
                model1 = pusher(object_pos=object_, goal_pos=goal_pos, distractor_pos=distractor_, object_color=color, distractor_color=distractor_color,
                                table_texture=TABLE_TEXTURE_PATH)
                
                model2 = pusher(object_pos=distractor_, goal_pos=goal_pos, distractor_pos=object_, object_color=distractor_color, distractor_color=color,
                                table_texture=TABLE_TEXTURE_PATH)
                # dir1 = RLLAB_PATH + 'train_%d.xml' % (24*2*i + j)
                # dir2 = RLLAB_PATH + 'train_%d.xml' % (24*(2*i+1) + j)
                # dir1 = RLLAB_PATH + 'test_%d.xml' % (24*2*i + j)
                # dir2 = RLLAB_PATH + 'test_%d.xml' % (24*(2*i+1) + j)
                # dir1 = RLLAB_MULTIGOAL_PATH + 'test_%d.xml' % (24*i + j)
                # dir1 = RLLAB_SHORTER_PATH + 'train_%d.xml' % (24*2*i + j)
                # dir2 = RLLAB_SHORTER_PATH + 'train_%d.xml' % (24*(2*i+1) + j)
                # dir1 = RLLAB_SHORTER_PATH + 'test_%d.xml' % (24*2*(i-500) + j)
                # dir2 = RLLAB_SHORTER_PATH + 'test_%d.xml' % (24*(2*(i-500)+1) + j)
                # dir1 = RLLAB_MULTIGOAL_PATH + 'test_%d.xml' % (24*(i-500) + j)
                if i < 500:
                    dir1 = RLLAB_REAL_PATH + 'train_%d.xml' % (24*2*i + j)
                    dir2 = RLLAB_REAL_PATH + 'train_%d.xml' % (24*(2*i+1) + j)
                else:
                    dir1 = RLLAB_REAL_PATH + 'test_%d.xml' % (24*2*i + j)
                    dir2 = RLLAB_REAL_PATH + 'test_%d.xml' % (24*(2*i+1) + j)
                print('Saving xml to %s' % dir1)
                model1.save(dir1)
                print('Saving xml to %s' % dir2)
                model2.save(dir2)