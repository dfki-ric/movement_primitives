import matplotlib.pyplot as plt
from pytransform3d.urdf import UrdfTransformManager


tm = UrdfTransformManager()
with open("pybullet-urdf/urdf/RH5v2.urdf", "r") as f:
    tm.load_urdf(f.read(), mesh_path="pybullet-urdf/")
ax = tm.plot_frames_in("RH5v2_Root_Link", s=0.1, whitelist=["RH5v2_Root_Link", "LTCP_Link", "RTCP_Link", "ALWristPitch_Link"])
tm.plot_connections_in("RH5v2_Root_Link", ax=ax)
plt.show()


tm = UrdfTransformManager()
with open("pybullet-only-arms-urdf/urdf/RH5.urdf", "r") as f:
    tm.load_urdf(f.read(), mesh_path="pybullet-only-arms-urdf/")
ax = tm.plot_frames_in("RH5_Root_Link", s=0.1, whitelist=["RH5_Root_Link", "LTCP_Link", "RTCP_Link", "ALWristPitch_Link"])
tm.plot_connections_in("RH5_Root_Link", ax=ax)
plt.show()
