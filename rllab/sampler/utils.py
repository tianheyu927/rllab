import numpy as np
import time

def stack_tensor_dict_list(tensor_dict_list):
    """
    Stack a list of dictionaries of {tensors or dictionary of tensors}.
    :param tensor_dict_list: a list of dictionaries of {tensors or dictionary of tensors}.
    :return: a dictionary of {stacked tensors or dictionary of stacked tensors}
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = stack_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def stack_tensor_list(tensor_list):
    return np.array(tensor_list)

def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1,
            always_return_paths=False, env_reset=True, save_video=True, video_filename='sim_out.mp4', vision=False):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    images = []
    o = env.wrapped_env.wrapped_env.reset()
    agent.reset()
    if animated:
        if 'viewer' in dir(env):
            viewer = env.viewer
            if viewer == None:
                env.render()
                viewer = env.viewer
                print('hi')
                #import pdb; pdb.set_trace()
        else:
            viewer = env.wrapped_env.wrapped_env.get_viewer()
        viewer.autoscale()
        viewer.cam.trackbodyid=0
        viewer.cam.distance = 4.0
        rotation_angle = -15
        cam_dist = 4
        cam_pos = np.array([0, 0, 0, cam_dist, -45, rotation_angle])
        for i in range(3):
            viewer.cam.lookat[i] = cam_pos[i]
        viewer.cam.distance = cam_pos[3]
        viewer.cam.elevation = cam_pos[4]
        viewer.cam.azimuth = cam_pos[5]
        viewer.cam.trackbodyid=-1
        env.render()

    if vision:
        image_obses = []
        nonimage_obses = []
        if 'get_current_image_obs' in dir(env):
            image_obs, nonimage_obs = env.get_current_image_obs()
        else:
            image_obs, nonimage_obs = env.wrapped_env.wrapped_env.get_current_image_obs()
    path_length = 0
    while path_length < max_path_length:
        if vision and 'get_vision_action' in dir(agent):
            a, agent_info = agent.get_vision_action(image_obs, nonimage_obs, t=path_length)
        else:
            a, agent_info = agent.get_action(o)

        next_o, r, d, env_info = env.step(a)
        if vision:
            if 'get_current_image_obs' in dir(env):
                next_image_obs, next_nonimage_obs = env.get_current_image_obs()
            else:
                next_image_obs, next_nonimage_obs = env.wrapped_env.wrapped_env.get_current_image_obs()

        #observations.append(env.observation_space.flatten(o))
        observations.append(np.squeeze(o))
        if vision:
            image_obses.append(image_obs)
            nonimage_obses.append(nonimage_obs)
        rewards.append(r)
        #actions.append(env.action_space.flatten(a))
        actions.append(np.squeeze(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if vision:
            image_obs = next_image_obs
            nonimage_obs = next_nonimage_obs
        if animated:
            env.render()
            timestep = 0.05
            #time.sleep(timestep / speedup)
            if save_video:
                from PIL import Image
                image = viewer.get_image()
                #image = env.wrapped_env.wrapped_env.get_viewer().get_image()
                pil_image = Image.frombytes('RGB', (image[1], image[2]), image[0])
                images.append(np.flipud(np.array(pil_image)))
    if animated:
        if save_video and len(images) >= max_path_length:
            import moviepy.editor as mpy
            clip = mpy.ImageSequenceClip(images, fps=20*speedup)
            if video_filename[-3:] == 'gif':
                clip.write_gif(video_filename, fps=20*speedup)
            else:
                clip.write_videofile(video_filename, fps=20*speedup)

    if animated and not always_return_paths:
        return
    if vision:
        return dict(
            observations=stack_tensor_list(observations),
            actions=stack_tensor_list(actions),
            rewards=stack_tensor_list(rewards),
            image_obs=stack_tensor_list(image_obses),
            nonimage_obs=stack_tensor_list(nonimage_obses),
            agent_infos=stack_tensor_dict_list(agent_infos),
            env_infos=stack_tensor_dict_list(env_infos),
        )
    else:
        return dict(
            observations=stack_tensor_list(observations),
            actions=stack_tensor_list(actions),
            rewards=stack_tensor_list(rewards),
            agent_infos=stack_tensor_dict_list(agent_infos),
            env_infos=stack_tensor_dict_list(env_infos),
        )