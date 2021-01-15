import numpy as np
import pybullet
import hyrodyn


print(pybullet.connect(pybullet.GUI))
pybullet.resetSimulation()
pybullet.setTimeStep(0.001)
urdf =  "hyrodyn_dev/control/hyrodyn/data/hybrid/rh5/full/urdf/RH5.urdf"
submechamisms = "hyrodyn_dev/control/hyrodyn/data/hybrid/rh5/full/submechanisms/submechanisms.yml"
pybullet.loadURDF(urdf, [0, 0, 0], useFixedBase=1)
robot = hyrodyn.RobotModel(urdf, submechamisms)
#print("Names of active joints: ", robot.jointnames_active)
#print("Names of independent joints: ", robot.jointnames_independent)
#print("Names of spanning tree joints: ", robot.jointnames_spanningtree)
robot.calculate_system_state()
print("Spanning tree joint positions: ", robot.Q)
print("Actuator joint positions: ", robot.u)
print(robot.pose)
body_name = "ALWrist_FT"
robot.calculate_forward_kinematics(body_name)
print("Forward kinematics of the body " + body_name + " ([X Y Z Qx Qy Qz Qw]):", robot.pose)
ee2robot = robot.pose
robot.pose_input = [ee2robot]
robot.calculate_inverse_kinematics([body_name])
print(robot.u)

# TODO make IK work
target_frame = np.eye(4)
target_frame[:3, -1] = np.array([0.3, 0.0, 0.3])
while True:
    pybullet.stepSimulation()
