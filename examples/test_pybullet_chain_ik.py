import numpy as np
import pybullet


print(pybullet.connect(pybullet.GUI))
pybullet.resetSimulation()
pybullet.setTimeStep(0.001)
pybullet.loadURDF("abstract-urdf-gripper/urdf/rh5_left_arm.urdf", [-1, 0, 5], useFixedBase=1)
pybullet.loadURDF("abstract-urdf-gripper/urdf/rh5_right_arm.urdf", [1, 0, 5], useFixedBase=1)
while True:
    pybullet.stepSimulation()