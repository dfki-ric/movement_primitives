import matplotlib.pyplot as plt
from pytransform3d.urdf import UrdfTransformManager


tm = UrdfTransformManager()
with open("abstract-urdf-gripper/urdf/rh5_left_arm.urdf", "r") as f:
    tm.load_urdf(f, mesh_path="abstract-urdf-gripper/urdf/")
whitelist = [
    "RH5",
    "ALShoulder1_Link", "ALShoulder2_Link", "ALShoulder3_Link",
    "ALElbow_Link",
    "ALWristPitch_Link", "ALWristRoll_Link", "ALWristYaw_Link"]
ax = tm.plot_frames_in("RH5", ax_s=1, s=0.1, whitelist=whitelist)
tm.plot_visuals(frame="RH5", ax=ax)
plt.show()
