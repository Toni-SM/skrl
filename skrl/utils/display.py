
def print_arguments(args):
    print("")
    print("Arguments")
    for a in args.__dict__:
        print("  |-- {}: {}".format(a, args.__getattribute__(a)))

def print_asset_option(option):
    print("")
    print("Asset option")
    print("  |-- angular_damping:", option.angular_damping)
    print("  |-- armature:", option.armature)
    print("  |-- collapse_fixed_joints:", option.collapse_fixed_joints)
    print("  |-- convex_decomposition_from_submeshes:", option.convex_decomposition_from_submeshes)
    print("  |-- default_dof_drive_mode:", option.default_dof_drive_mode)
    print("  |-- density:", option.density)
    print("  |-- disable_gravity:", option.disable_gravity)
    print("  |-- fix_base_link:", option.fix_base_link)
    print("  |-- flip_visual_attachments:", option.flip_visual_attachments)
    print("  |-- linear_damping:", option.linear_damping)
    print("  |-- max_angular_velocity:", option.max_angular_velocity)
    print("  |-- max_linear_velocity:", option.max_linear_velocity)
    print("  |-- mesh_normal_mode:", option.mesh_normal_mode)
    print("  |-- min_particle_mass:", option.min_particle_mass)
    print("  |-- override_com:", option.override_com)
    print("  |-- override_inertia:", option.override_inertia)
    print("  |-- replace_cylinder_with_capsule:", option.replace_cylinder_with_capsule)
    print("  |-- slices_per_cylinder:", option.slices_per_cylinder)
    print("  |-- tendon_limit_stiffness:", option.tendon_limit_stiffness)
    print("  |-- thickness:", option.thickness)
    print("  |-- use_mesh_materials:", option.use_mesh_materials)
    print("  |-- use_physx_armature:", option.use_physx_armature)
    print("  |-- vhacd_enabled:", option.vhacd_enabled)
    # print("  |-- vhacd_param:", option.vhacd_param)   # AttributeError: 'isaacgym._bindings.linux-x86_64.gym_36.AssetOption' object has no attribute 'vhacd_param'

def print_sim_components(gym, sim):
    print("")
    print("Sim components")
    print("  |--  env count:", gym.get_env_count(sim))
    print("  |--  actor count:", gym.get_sim_actor_count(sim))
    print("  |--  rigid body count:", gym.get_sim_rigid_body_count(sim))
    print("  |--  joint count:", gym.get_sim_joint_count(sim))
    print("  |--  dof count:", gym.get_sim_dof_count(sim))
    print("  |--  force sensor count:", gym.get_sim_force_sensor_count(sim))

def print_env_components(gym, env):
    print("")
    print("Env components")
    print("  |--  actor count:", gym.get_actor_count(env))
    print("  |--  rigid body count:", gym.get_env_rigid_body_count(env))
    print("  |--  joint count:", gym.get_env_joint_count(env))
    print("  |--  dof count:", gym.get_env_dof_count(env))

def print_actor_components(gym, env, actor):
    print("")
    print("Actor components")
    print("  |--  rigid body count:", gym.get_actor_rigid_body_count(env, actor))
    print("  |--  joint count:", gym.get_actor_joint_count(env, actor))
    print("  |--  dof count:", gym.get_actor_dof_count(env, actor))
    print("  |--  actuator count:", gym.get_actor_actuator_count(env, actor))
    print("  |--  rigid shape count:", gym.get_actor_rigid_shape_count(env, actor))
    print("  |--  soft body count:", gym.get_actor_soft_body_count(env, actor))
    print("  |--  tendon count:", gym.get_actor_tendon_count(env, actor))

def print_dof_properties(gymapi, props):
    print("")
    print("DOF properties")
    print("  |--  hasLimits:", props["hasLimits"])
    print("  |--  lower:", props["lower"])
    print("  |--  upper:", props["upper"])
    print("  |--  driveMode:", props["driveMode"])
    print("  |      |-- {}: gymapi.DOF_MODE_NONE".format(int(gymapi.DOF_MODE_NONE)))
    print("  |      |-- {}: gymapi.DOF_MODE_POS".format(int(gymapi.DOF_MODE_POS)))
    print("  |      |-- {}: gymapi.DOF_MODE_VEL".format(int(gymapi.DOF_MODE_VEL)))
    print("  |      |-- {}: gymapi.DOF_MODE_EFFORT".format(int(gymapi.DOF_MODE_EFFORT)))
    print("  |--  stiffness:", props["stiffness"])
    print("  |--  damping:", props["damping"])
    print("  |--  velocity (max):", props["velocity"])
    print("  |--  effort (max):", props["effort"])
    print("  |--  friction:", props["friction"])
    print("  |--  armature:", props["armature"])

def print_links_and_dofs(gym, asset):
    link_dict = gym.get_asset_rigid_body_dict(asset)
    dof_dict = gym.get_asset_dof_dict(asset)

    print("")
    print("Links")
    for k in link_dict:
        print("  |-- {}: {}".format(k, link_dict[k]))
    print("DOFs")
    for k in dof_dict:
        print("  |-- {}: {}".format(k, dof_dict[k]))
