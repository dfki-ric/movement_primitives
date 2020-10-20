import numpy as np
import pybullet
import hyrodyn


print(pybullet.connect(pybullet.GUI))
pybullet.resetSimulation()
pybullet.setTimeStep(0.001)
urdf =  "hyrodyn_dev/control/hyrodyn/data/hybrid/rh5/full/urdf/RH5_PyBullet.urdf"
submechamisms = "hyrodyn_dev/control/hyrodyn/data/hybrid/rh5/full/submechanisms/submechanisms.yml"
pybullet.loadURDF(urdf, [0, 0, 0], useFixedBase=1)
robot = hyrodyn.RobotModel(urdf, submechamisms)

# TODO make IK work
target_frame = np.eye(4)
target_frame[:3, -1] = np.array([0.3, 0.0, 0.3])
while True:
    pybullet.stepSimulation()