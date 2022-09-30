import numpy as np
from isaacgym import gymutil, gymapi, gymtorch
from math import sqrt
import math
from base import create_basic_isaac

gym, sim = create_basic_isaac()

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load ball asset
asset_root = "../../assets"
# asset_file = "mjcf/nv_humanoid.xml"
asset_file = "urdf/cartpole.urdf"
# asset_file = "urdf/ball.urdf"
# asset_file = "mjcf/ai_baby_origin.xml"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

env_spacing = 5
env_lower = gymapi.Vec3(-1, 0.0, -1)
env_upper = gymapi.Vec3(1, env_spacing, 1)

envs = []
actor_handles = []
num_per_row = 3

num_dofs = gym.get_asset_dof_count(asset)
print("\n\n\n\n\n")
print("num_dof: ",  num_dofs)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

num_envs = 10

for i in range(num_envs):
    # create env
    env_ptr = gym.create_env(sim, env_lower, env_upper, 4)
    envs.append(env_ptr)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 1.32, 0.5)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    # pose_opposite = gymapi.Transform()
    # pose_opposite.p = gymapi.Vec3(0.0, 1.32, -0.5)
    # pose_opposite.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    actor_handle = gym.create_actor(env_ptr, asset, pose, "actor", i, 1)
    # actor_handle = gym.create_actor(env_ptr, asset, pose_opposite, "actor", i, 1)
    actor_handles.append(actor_handle)

    dof_pros = gym.get_actor_dof_properties(env_ptr, actor_handle)
    print(dof_pros)

    # set default DOF positions
    gym.set_actor_dof_states(env_ptr, actor_handle, dof_states, gymapi.STATE_ALL)

#
#
# gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
# env = gym.create_env(sim, env_lower, env_upper, 1)
# pose = gymapi.Transform()
# pose.r = gymapi.Quat.from_euler_zyx(-math.pi*0.5, 0, 0)
# pose.p = gymapi.Vec3(0, 1.5, 0)
# handler = gym.create_actor(env, asset, pose, None, 0, 1)
#
#
# cam_pos = gymapi.Vec3(2, 2, 2)
# cam_target = gymapi.Vec3(-10, -2.5, -13)
# gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# create a local copy of initial state, which we can send back for reset
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

# gym.subscribe_viewer_keyboard_event(
#     viewer, gymapi.KEY_ESCAPE, "QUIT")
# gym.subscribe_viewer_keyboard_event(
#     viewer, gymapi.KEY_V, "toggle_viewer_sync")

while not gym.query_viewer_has_closed(viewer):

    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)



gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
