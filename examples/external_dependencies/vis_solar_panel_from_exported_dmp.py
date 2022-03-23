import numpy as np
import pytransform3d.visualizer as pv
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
from movement_primitives.kinematics import Kinematics
from movement_primitives.testing.simulation import get_absolute_path


def animation_callback(step, graph, left_arm, right_arm, left_joint_trajectory, right_joint_trajectory):
    left_arm.forward(left_joint_trajectory[step])
    right_arm.forward(right_joint_trajectory[step])
    graph.set_data()
    return graph


solar_panel_idx = 0
panel_rotation_angle = np.deg2rad(90)
n_steps = 201

q0_left = np.array([-1.57, 0.9, 0, -1.3, 0, 0, -0.55])
q0_right = np.array([1.57, -0.9, 0, 1.3, 0, 0, -0.55])

with open(get_absolute_path("pybullet-only-arms-urdf/urdf/RH5.urdf", "models/robots/rh5_models"), "r") as f:
    kin = Kinematics(f.read(), mesh_path=get_absolute_path("pybullet-only-arms-urdf/urdf/", "models/robots/rh5_models"))
left_arm = kin.create_chain(
    ["ALShoulder1", "ALShoulder2", "ALShoulder3",
     "ALElbow", "ALWristRoll", "ALWristYaw", "ALWristPitch"],
    "RH5_Root_Link", "LTCP_Link")
right_arm = kin.create_chain(
    ["ARShoulder1", "ARShoulder2", "ARShoulder3",
     "ARElbow", "ARWristRoll", "ARWristYaw", "ARWristPitch"],
    "RH5_Root_Link", "RTCP_Link")

left_arm.forward(q0_left)
right_arm.forward(q0_right)

left2base_start = kin.tm.get_transform("ALWristPitch_Link", "RH5_Root_Link")
right2base_start = kin.tm.get_transform("ARWristPitch_Link", "RH5_Root_Link")
start_y = np.hstack((pt.pq_from_transform(left2base_start),
                     pt.pq_from_transform(right2base_start)))

# Confuse the DMP, just for testing purposes...
start_y[3:7] *= -1.0
start_y[10:14] *= -1.0

# Read DMP from your favorite format.
from movement_primitives.io import read_yaml

dmp = read_yaml("rh5_dual_arm_dmp.yaml")
#dmp = read_json("rh5_dual_arm_dmp.json")
#dmp = read_pickle("rh5_dual_arm_dmp.pickle")

# Configure start poses of end effectors. Should be initialized from the
# current end-effector poses.
# start_y is a numpy array with 14 entries: position of the left end effector,
# orientation quaternion of the left end effector, position of the right
# end effector, and orientation quaternion of the right end effector.

# WARNING! There are always two quaternions that represent the exact same
# orientation: q and -q. The problem is that a DMP that moves from q to q
# does not move at all, while a DMP that moves from q to -q moves a lot. We
# have to make sure that start_y always contains the quaternion representation
# that is closest to the previous start_y!
start_y[3:7] = pr.pick_closest_quaternion(start_y[3:7], target_quaternion=dmp.start_y[3:7])
start_y[10:14] = pr.pick_closest_quaternion(start_y[10:14], target_quaternion=dmp.start_y[10:14])
dmp.configure(start_y=start_y)

# Generate trajectory with open loop.
# T - time for each step in the trajectory
# Y_pq - array of shape (number of steps, 14) that contains left position,
#        left orientation, right position, and right orientation for each step
T, Y_pq = dmp.open_loop()

# Generate trajectory with closed loop.
y = np.copy(start_y)  # positions
yd = np.zeros(12)  # velocities
dmp.reset()
dmp.configure(start_y=y)
for i in range(n_steps):
    # get state of end effectors:
    # y = ...
    # yd = ...  # if it is not available, you can ignore this and take the DMP's output

    y, yd = dmp.step(y, yd)  # generate desired poses and velocities

    # here you can send commands to WBC

    Y_pq[i] = y

Y_transforms = np.empty((len(Y_pq), 2, 4, 4))
Y_transforms[:, 0] = ptr.transforms_from_pqs(Y_pq[:, :7])
Y_transforms[:, 1] = ptr.transforms_from_pqs(Y_pq[:, 7:])

lwp_trajectory = Y_transforms[:, 0]
rwp_trajectory = Y_transforms[:, 1]

ltcp2lwp = kin.tm.get_transform("LTCP_Link", "ALWristPitch_Link")
rtcp2rwp = kin.tm.get_transform("RTCP_Link", "ARWristPitch_Link")

left_trajectory = np.array([pt.concat(ltcp2lwp, lwp2base) for lwp2base in lwp_trajectory])
right_trajectory = np.array([pt.concat(rtcp2rwp, rwp2base) for rwp2base in rwp_trajectory])

print("Inverse kinematics...")
random_state = np.random.RandomState(0)
left_joint_trajectory = left_arm.inverse_trajectory(left_trajectory, q0_left, random_state=random_state)
right_joint_trajectory = right_arm.inverse_trajectory(right_trajectory, q0_right, random_state=random_state)

lwp2ltcp = kin.tm.get_transform("ALWristPitch_Link", "LTCP_Link")
rwp2rtcp = kin.tm.get_transform("ARWristPitch_Link", "RTCP_Link")

lwp_trajectory = np.array([pt.concat(lwp2ltcp, ltcp2base) for ltcp2base in left_trajectory])
rwp_trajectory = np.array([pt.concat(rwp2rtcp, rtcp2base) for rtcp2base in right_trajectory])

fig = pv.figure()
fig.plot_transform(s=0.3)

graph = fig.plot_graph(
    kin.tm, "RH5_Root_Link", show_visuals=True, show_collision_objects=False, show_frames=True, s=0.1,
    whitelist=["ALWristPitch_Link", "ARWristPitch_Link", "LTCP_Link", "RTCP_Link"])

left2base_start = kin.tm.get_transform("LTCP_Link", "RH5_Root_Link")
right2base_start = kin.tm.get_transform("RTCP_Link", "RH5_Root_Link")
fig.plot_transform(left2base_start, s=0.15)
fig.plot_transform(right2base_start, s=0.15)

fig.plot_transform(left_trajectory[-1], s=0.15)
fig.plot_transform(right_trajectory[-1], s=0.15)

pv.Trajectory(lwp_trajectory, s=0.05).add_artist(fig)
pv.Trajectory(rwp_trajectory, s=0.05).add_artist(fig)

fig.view_init()
fig.animate(
    animation_callback, len(left_joint_trajectory), loop=True,
    fargs=(graph, left_arm, right_arm, left_joint_trajectory, right_joint_trajectory))
fig.show()
