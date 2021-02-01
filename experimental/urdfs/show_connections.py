from pytransform3d.urdf import UrdfTransformManager


tm = UrdfTransformManager()
with open("pybullet-only-arms-urdf/urdf/RH5.urdf", "r") as f:
    tm.load_urdf(f.read())
tm.write_png("graph.png", "twopi")
