import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mocap.pandas_utils import match_columns, rename_stream_groups
from mocap.conversion import array_from_dataframe
from pytransform3d import transformations as pt
from pytransform3d import trajectories as ptr
from pytransform3d.urdf import UrdfTransformManager
from movement_primitives.promp import ProMP
from gmr import GMM


########################################################################################################################
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras.layers import Dense, Activation, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class MDN(tf.keras.Model):
    """Mixture density network with two hidden layers.

    Parameters
    ----------
    units : int, optional (default: 100)
        Number of units per hidden layer

    n_components : int, optional (default: 2)
        Number of output Gaussians

    n_outputs : int, optional (default: 1)
        Number of output dimensions
    """
    def __init__(self, units=100, n_components=2, n_outputs=1):
        super(MDN, self).__init__(name="MDN")
        self.neurons = units
        self.n_components = n_components
        self.n_outputs = n_outputs

        self.h1 = Dense(units, activation="relu", name="h1")
        self.h2 = Dense(units, activation="relu", name="h2")

        self.alphas = Dense(n_components, activation="softmax", name="alphas")
        self.mus = Dense(n_components * n_outputs, name="mus")
        self.sigmas = Dense(n_components * n_outputs, activation="nnelu", name="sigmas")

        self.output_layer = Concatenate(name="pvec")

    def call(self, inputs, training=None, mask=None):
        x = self.h1(inputs)
        x = self.h2(x)

        alpha = self.alphas(x)
        mu = self.mus(x)
        sigma = self.sigmas(x)

        return self.output_layer([alpha, mu, sigma])

    def predict_parameters(self, x):
        return slice_parameter_vectors(self.predict(x), self.n_components, self.n_outputs)

    def predict_distribution(self, x):
        alpha, mu, sigma = self.predict_parameters(x)
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=alpha),
            components_distribution=tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma))

    def to_gmm(self, x):
        alpha, mu, sigma = self.predict_parameters(np.atleast_2d(x))
        alpha = np.asarray(alpha[0])
        mu = np.asarray(mu[0])
        sigma = np.asarray(sigma[0])

        covs = np.empty((self.n_components, self.n_outputs, self.n_outputs))
        for i in range(self.n_components):
            covs[i] = np.diag(sigma[i]) ** 2
        gmm = GMM(
            n_components=self.n_components, priors=alpha, means=mu, covariances=covs)

        return gmm


def nnelu(input):
    """Computes the Non-Negative Exponential Linear Unit"""
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))


tf.keras.utils.get_custom_objects().update({'nnelu': Activation(nnelu)})


class GNLLLoss:
    def __init__(self, n_components, n_outputs):
        self.n_components = n_components
        self.n_outputs = n_outputs

    def loss(self, y, parameter_vector):
        """Computes the mean negative log-likelihood loss of y given the mixture parameters."""
        alpha, mu, sigma = slice_parameter_vectors(parameter_vector, self.n_components, self.n_outputs)

        gm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=alpha),
            components_distribution=tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma))

        log_likelihood = gm.log_prob(y)  # Evaluate log-probability of y

        return -tf.reduce_mean(log_likelihood, axis=0)


def slice_parameter_vectors(parameter_vector, n_components, n_outputs):
    """Returns an unpacked list of parameter vectors."""
    return [parameter_vector[:, :n_components],
            tf.reshape(parameter_vector[:, n_components:n_components + n_components * n_outputs], (-1, n_components, n_outputs)),
            tf.reshape(parameter_vector[:, n_components + n_components * n_outputs:], (-1, n_components, n_outputs))]


########################################################################################################################


# available contexts: "panel_width", "clockwise", "counterclockwise", "left_arm", "right_arm"
def generate_training_data(
        pattern, n_weights_per_dim, context_names, verbose=0):
    Ts, Ps, contexts = load_demos(pattern, context_names, verbose=verbose)

    n_demos = len(Ts)
    n_dims = Ps[0].shape[1]

    promp = ProMP(n_dims=n_dims, n_weights_per_dim=n_weights_per_dim)
    weights = np.empty((n_demos, n_dims * n_weights_per_dim))
    for demo_idx in range(n_demos):
        weights[demo_idx] = promp.weights(Ts[demo_idx], Ps[demo_idx])

    return weights, Ts, Ps, contexts


def load_demos(pattern, context_names, verbose=0):
    Ts = []
    Ps = []
    contexts = []
    for idx, path in enumerate(list(glob.glob(pattern))):
        if verbose:
            print("Loading %s" % path)
        T, P, context = load_demo(path, context_names, verbose=verbose - 1)
        Ts.append(T)
        Ps.append(P)
        contexts.append(context)
    return Ts, Ps, contexts


def load_demo(path, context_names, verbose=0):
    trajectory = pd.read_csv(path, sep=" ")

    context = trajectory[list(context_names)].iloc[0].to_numpy()
    if verbose:
        print("Context: %s" % (context,))

    patterns = ["time\.microseconds",
                "kuka_lbr_cart_pos_ctrl_left\.current_feedback\.pose\.position\.data.*",
                "kuka_lbr_cart_pos_ctrl_left\.current_feedback\.pose\.orientation\.re.*",
                "kuka_lbr_cart_pos_ctrl_left\.current_feedback\.pose\.orientation\.im.*",
                "kuka_lbr_cart_pos_ctrl_right\.current_feedback\.pose\.position\.data.*",
                "kuka_lbr_cart_pos_ctrl_right\.current_feedback\.pose\.orientation\.re.*",
                "kuka_lbr_cart_pos_ctrl_right\.current_feedback\.pose\.orientation\.im.*"]
    columns = match_columns(trajectory, patterns)
    trajectory = trajectory[columns]

    group_rename = {
        "(time\.microseconds)": "Time",
        "(kuka_lbr_cart_pos_ctrl_left\.current_feedback\.pose\.position\.data).*": "left_pose",
        "(kuka_lbr_cart_pos_ctrl_left\.current_feedback\.pose\.orientation).*": "left_pose",
        "(kuka_lbr_cart_pos_ctrl_right\.current_feedback\.pose\.position\.data).*": "right_pose",
        "(kuka_lbr_cart_pos_ctrl_right\.current_feedback\.pose\.orientation).*": "right_pose"
    }
    trajectory = rename_stream_groups(trajectory, group_rename)

    trajectory["Time"] = trajectory["Time"] / 1e6
    trajectory["Time"] -= trajectory["Time"].iloc[0]
    T = trajectory["Time"].to_numpy()

    P = array_from_dataframe(
        trajectory,
        ["left_pose[0]", "left_pose[1]", "left_pose[2]",
         "left_pose.re", "left_pose.im[0]", "left_pose.im[1]", "left_pose.im[2]",
         "right_pose[0]", "right_pose[1]", "right_pose[2]",
         "right_pose.re", "right_pose.im[0]", "right_pose.im[1]", "right_pose.im[2]"])

    return T, P, context


n_dims = 14
n_weights_per_dim = 10
# omitted contexts: "left_arm", "right_arm"
context_names = ["panel_width", "clockwise", "counterclockwise"]

#pattern = "data/kuka/20200129_peg_in_hole/csv_processed/*/*.csv"
#pattern = "data/kuka/20191213_carry_heavy_load/csv_processed/*/*.csv"
pattern = "data/kuka/20191023_rotate_panel_varying_size/csv_processed/*/*.csv"

weights, Ts, Ps, contexts = generate_training_data(
    pattern, n_weights_per_dim, context_names=context_names, verbose=2)

random_state = np.random.RandomState(0)

X = np.array(contexts)
Y = np.array(weights)
print(X.shape)
print(Y.shape)

min_max_scaler = MinMaxScaler()
Y = min_max_scaler.fit_transform(Y)

n_components = 5
mdn = MDN(units=100, n_components=n_components, n_outputs=Y.shape[1])
mdn.compile(loss=GNLLLoss(n_components=n_components, n_outputs=Y.shape[1]).loss, optimizer=tf.keras.optimizers.Adam(1e-3))
mon = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
mdn.fit(x=X, y=Y, epochs=200, validation_data=(X, Y),  # validation data
        callbacks=[mon], batch_size=32, verbose=1)

"""
gmm = GMM(n_components=5, random_state=random_state)
X = np.hstack((contexts, weights))
gmm.from_samples(X)
"""

n_steps = 100
T_query = np.linspace(0, 1, n_steps)

fig = plt.figure()
ax = pt.plot_transform(s=0.1)
tm = UrdfTransformManager()
with open("kuka_lbr/urdf/kuka_lbr.urdf", "r") as f:
    tm.load_urdf(f.read(), mesh_path="kuka_lbr/urdf/")
tm.plot_visuals("kuka_lbr", ax=ax, convex_hull_of_mesh=True)


for panel_width, color, idx in zip([0.3, 0.4, 0.5], ("r", "g", "b"), range(3)):
    print("panel_width = %.2f, color = %s" % (panel_width, color))

    context = np.array([panel_width, 0.0, 1.0])
    conditional_weight_distribution = mdn.to_gmm(context).to_mvn()
    #conditional_weight_distribution = gmm.condition(np.arange(len(context)), context).to_mvn()
    promp = ProMP(n_dims=n_dims, n_weights_per_dim=n_weights_per_dim)
    # TODO scale weights
    mean = min_max_scaler.inverse_transform([conditional_weight_distribution.mean])[0]
    scale = np.diag(1.0 / min_max_scaler.scale_)
    covariance = scale.dot(conditional_weight_distribution.covariance).dot(scale)  # .T for non-diagonal
    promp.from_weight_distribution(mean, covariance)
    samples = promp.sample_trajectories(T_query, 3, random_state)

    pcl_points = []
    for P in samples:
        pcl_points.extend(P[:, :3])
        pcl_points.extend(P[:, 7:10])

        ee_distances = np.linalg.norm(P[:, :3] - P[:, 7:10], axis=1)
        average_ee_distance = np.mean(ee_distances)
        print("Average distance = %.2f" % average_ee_distance)
    pcl_points = np.array(pcl_points)

    ax.scatter(pcl_points[:, 0], pcl_points[:, 1], pcl_points[:, 2], c=color, alpha=0.2)

"""
# plot training data
for P in Ps:
    ptr.plot_trajectory(P=P[:, :7], ax=ax, s=0.02)
    ptr.plot_trajectory(P=P[:, 7:], ax=ax, s=0.02)
"""

ax.view_init(azim=0, elev=25)
plt.show()
