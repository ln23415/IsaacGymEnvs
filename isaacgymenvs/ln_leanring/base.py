import numpy as np
from isaacgym import gymutil, gymapi, gymtorch


def create_basic_isaac(physics_engine=gymapi.SIM_PHYSX):
    sim_params = gymapi.SimParams()
    if physics_engine == gymapi.SIM_FLEX:
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif physics_engine == gymapi.SIM_PHYSX:
        sim_params.substeps = 1
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = 8
        sim_params.physx.use_gpu = False

    gym = gymapi.acquire_gym()
    sim_params.use_gpu_pipeline = False
    sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0)
    sim = gym.create_sim(0, 0, physics_engine, sim_params)

    if sim is None:
        print("*** Failed to create sim")
        quit()

    # add ground plane
    plane_params = gymapi.PlaneParams()
    gym.add_ground(sim, plane_params)

    return gym, sim

