import matplotlib.pyplot as plt
from pytransform3d.urdf import UrdfTransformManager


tm = UrdfTransformManager()
with open("abstract-urdf-gripper/urdf/GripperLeft.urdf", "r") as f:
    tm.load_urdf(f, mesh_path="abstract-urdf-gripper/urdf/")
ax = tm.plot_frames_in("ALWristPitch_Link", ax_s=0.2, s=0.1, whitelist=["GripperLeft", "ALWristPitch_Link"])
tm.plot_visuals(frame="ALWristPitch_Link", ax=ax)
plt.show()
