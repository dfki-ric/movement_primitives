import numpy as np
import pytransform3d.visualizer as pv
from movement_primitives.kinematics import Kinematics


with open("kuka_lbr/urdf/kuka_lbr.urdf", "r") as f:
    kin = Kinematics(f.read(), mesh_path="kuka_lbr/urdf/")
right_chain = kin.create_chain(
    ["kuka_lbr_r_joint_%d" % i for i in range(1, 8)],
    "kuka_lbr", "kuka_lbr_r_tcp", verbose=0)
left_chain = kin.create_chain(
    ["kuka_lbr_l_joint_%d" % i for i in range(1, 8)],
    "kuka_lbr", "kuka_lbr_l_tcp", verbose=0)
left_ee2base = left_chain.forward(np.array([0.5, 0.1, 0.0, -0.3, 0.0, 0.3, 0]) * np.pi)
right_ee2base = right_chain.forward(np.array([-0.5, 0.1, 0.0, -0.3, 0.0, 0.3, 0]) * np.pi)
right_chain.forward(np.zeros(7))
left_chain.forward(np.zeros(7))
random_state = np.random.RandomState(5)


def animation_callback(step, right_chain, left_chain, graph, right_ee2base, left_ee2base, random_state):
    right_chain.inverse_with_random_restarts(right_ee2base, random_state=random_state)
    left_chain.inverse_with_random_restarts(left_ee2base, random_state=random_state)
    graph.set_data()
    return graph


fig = pv.figure()
fig.plot_basis(s=0.1)
graph = fig.plot_graph(kin.tm, "kuka_lbr", show_visuals=True)
#graph = fig.plot_graph(kin.tm, "kuka_lbr", show_frames=True, s=0.1)
fig.view_init()
fig.animate(animation_callback, 1, loop=True, fargs=(right_chain, left_chain, graph, right_ee2base, left_ee2base, random_state))
fig.show()
