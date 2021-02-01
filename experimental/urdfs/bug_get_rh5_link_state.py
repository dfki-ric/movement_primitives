import pybullet


pybullet.connect(pybullet.GUI)
pybullet.resetSimulation()
pybullet.setTimeStep(0.001)
pybullet.setGravity(0, 0, -9.81)
rh5 = pybullet.loadURDF("pybullet-only-arms-urdf/urdf/RH5.urdf", useFixedBase=True)
pybullet.stepSimulation()
pybullet.getLinkState(rh5, linkIndex=1)
while True:
    pybullet.stepSimulation()
