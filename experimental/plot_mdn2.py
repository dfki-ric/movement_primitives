# https://towardsdatascience.com/a-hitchhikers-guide-to-mixture-density-networks-76b435826cca
# https://colab.research.google.com/drive/1at5lIq0jYvA58AmJ0aVgn2aUVpzIbwS3#scrollTo=zcuy684luZtM
# https://www.tensorflow.org/probability
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.keras.layers import Dense, Activation, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
from gmr import GMM


tf.random.set_seed(42)
np.random.seed(42)

warnings.filterwarnings("ignore")

x_size = 8
y_size = 4


class MDN(tf.keras.Model):
    def __init__(self, units=100, n_components=2):
        super(MDN, self).__init__(name="MDN")
        self.neurons = units
        self.n_components = n_components

        self.h1 = Dense(units, activation="relu", name="h1")
        self.h2 = Dense(units, activation="relu", name="h2")

        self.alphas = Dense(n_components, activation="softmax", name="alphas")
        self.mus = Dense(n_components, name="mus")
        self.sigmas = Dense(n_components, activation="nnelu", name="sigmas")
        self.pvec = Concatenate(name="pvec")

    def call(self, inputs, training=None, mask=None):
        x = self.h1(inputs)
        x = self.h2(x)

        alpha_v = self.alphas(x)
        mu_v = self.mus(x)
        sigma_v = self.sigmas(x)

        return self.pvec([alpha_v, mu_v, sigma_v])

    def predict_parameters(self, x):
        return slice_parameter_vectors(self.predict(x), self.n_components)

    def predict_distribution(self, x):
        alpha, mu, sigma = self.predict_parameters(x)
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=alpha),
            components_distribution=tfd.Normal(loc=mu, scale=sigma))

    def to_gmm(self, x):
        n_features = len(x)
        alpha, mu, sigma = self.predict_parameters(x)
        covs = np.empty((self.n_components, n_features, n_features))
        for i in range(self.n_components):
            covs[i] = np.eye(n_features) * sigma
        gmm = GMM(n_components=self.n_components, priors=alpha, means=mu, covariances=covs)
        return gmm


def nnelu(input):
    """Computes the Non-Negative Exponential Linear Unit"""
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))


def slice_parameter_vectors(parameter_vector, n_components):
    """Returns an unpacked list of parameter vectors."""
    return [parameter_vector[:, i * n_components:(i + 1) * n_components] for i in range(no_parameters)]


class GNLLLoss:
    def __init__(self, n_components):
        self.n_components = n_components

    def loss(self, y, parameter_vector):
        """Computes the mean negative log-likelihood loss of y given the mixture parameters."""
        alpha, mu, sigma = slice_parameter_vectors(parameter_vector, self.n_components)  # Unpack parameter vectors

        gm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=alpha),
            components_distribution=tfd.Normal(loc=mu, scale=sigma))

        log_likelihood = gm.log_prob(tf.transpose(y))  # Evaluate log-probability of y

        return -tf.reduce_mean(log_likelihood, axis=-1)


def gnll_eval(y, alpha, mu, sigma):
    """Computes the mean negative log-likelihood loss of y given the mixture parameters."""
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=alpha),
        components_distribution=tfd.Normal(loc=mu, scale=sigma))
    log_likelihood = gm.log_prob(tf.transpose(y))
    return -tf.reduce_mean(log_likelihood, axis=-1)


tf.keras.utils.get_custom_objects().update({'nnelu': Activation(nnelu)})

samples = 10000

x_data = np.float32(np.random.uniform(-10, 10, (1, samples)))
r_data = np.array([np.random.normal(scale=np.abs(i)) for i in x_data])
y_data = np.float32(np.square(x_data)+r_data*2.0)

x_data2 = np.float32(np.random.uniform(-10, 10, (1, samples)))
r_data2 = np.array([np.random.normal(scale=np.abs(i)) for i in x_data2])
y_data2 = np.float32(-np.square(x_data2)+r_data2*2.0)

x_data = np.concatenate((x_data,x_data2),axis=1).T
y_data = np.concatenate((y_data,y_data2),axis=1).T

min_max_scaler = MinMaxScaler()
y_data = min_max_scaler.fit_transform(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42, shuffle=True)

no_parameters = 3
components = 2
units = 200

mdn = MDN(units=units, n_components=components)
mdn.compile(loss=GNLLLoss(n_components=components).loss, optimizer=tf.keras.optimizers.Adam(1e-3))
mon = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
mdn.fit(x=x_train, y=y_train, epochs=200, validation_data=(x_test, y_test),
        callbacks=[mon], batch_size=128, verbose=1)

# Plot predictions
s = np.linspace(-10, 10, 1000)[:, np.newaxis].astype(np.float32)
alpha_pred, mu_pred, sigma_pred = mdn.predict_parameters(s)

fig = plt.figure(figsize=(x_size, y_size))
ax = plt.gca()
ax.scatter(x_train, y_train, s=1, alpha=0.2, color=sns.color_palette()[0])
for mx in range(components):
    plt.fill_between(
        s.ravel(), mu_pred[:, mx] - sigma_pred[:, mx], mu_pred[:, mx] + sigma_pred[:, mx],
        color=sns.color_palette()[1 + mx], alpha=0.2)
    plt.plot(s, mu_pred[:, mx], color=sns.color_palette()[1 + mx], linewidth=5, linestyle='-', markersize=3)
    plt.plot(s, mu_pred[:, mx] - sigma_pred[:, mx], color=sns.color_palette()[1 + mx], linewidth=3, linestyle='--',
             markersize=3)
    plt.plot(s, mu_pred[:, mx] + sigma_pred[:, mx], color=sns.color_palette()[1 + mx], linewidth=3, linestyle='--',
             markersize=3)
data_leg = mpatches.Patch(color=sns.color_palette()[0])
data_mdn1 = mpatches.Patch(color=sns.color_palette()[1])
data_mdn2 = mpatches.Patch(color=sns.color_palette()[2])
ax.legend(handles=[data_leg, data_mdn1, data_mdn2],
          labels=["Data", "MDN (c=1)", "MDN (c=2)"],
          loc=9, borderaxespad=0.1, framealpha=1.0, fancybox=True,
          bbox_to_anchor=(0.5, -0.05), ncol=6, shadow=True, frameon=False)

# Plot pdf for x=8
gm = mdn.predict_distribution([8.0])
#mdn.to_gmm([8.0])
x = np.linspace(0, 1, 1000)
pyx = gm.prob(x)

fig = plt.figure(figsize=(x_size, y_size // 2))
plt.plot(x, pyx, alpha=1, color=sns.color_palette()[0], linewidth=2)
plt.xlabel(r"y")
plt.ylabel(r"$p(y|x=8)$")
plt.show()
