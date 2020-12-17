# https://towardsdatascience.com/a-hitchhikers-guide-to-mixture-density-networks-76b435826cca
# https://colab.research.google.com/drive/1at5lIq0jYvA58AmJ0aVgn2aUVpzIbwS3#scrollTo=zcuy684luZtM
# https://www.tensorflow.org/probability
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

#tf.random.set_random_seed(42)
np.random.seed(42)

from tensorflow_probability import distributions as tfd

from tensorflow.keras.layers import Input, Dense, Activation, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import matplotlib as mpl

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({'font.size': 12})

import warnings

warnings.filterwarnings("always")


def remove_ax_window(ax):
    """
        Remove all axes and tick params in pyplot.
        Input: ax object.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis=u'both', which=u'both', length=0)


dpi = 140
x_size = 8
y_size = 4
alt_font_size = 14

save_figure = False
use_tb = False


class MDN(tf.keras.Model):
    def __init__(self, neurons=100, components=2):
        super(MDN, self).__init__(name="MDN")
        self.neurons = neurons
        self.components = components

        self.h1 = Dense(neurons, activation="relu", name="h1")
        self.h2 = Dense(neurons, activation="relu", name="h2")

        self.alphas = Dense(components, activation="softmax", name="alphas")
        self.mus = Dense(components, name="mus")
        self.sigmas = Dense(components, activation="nnelu", name="sigmas")
        self.pvec = Concatenate(name="pvec")

    def call(self, inputs, training=None, mask=None):
        x = self.h1(inputs)
        x = self.h2(x)

        alpha_v = self.alphas(x)
        mu_v = self.mus(x)
        sigma_v = self.sigmas(x)

        return self.pvec([alpha_v, mu_v, sigma_v])


def nnelu(input):
    """ Computes the Non-Negative Exponential Linear Unit
    """
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))


def slice_parameter_vectors(parameter_vector):
    """ Returns an unpacked list of paramter vectors.
    """
    return [parameter_vector[:, i * components:(i + 1) * components] for i in range(no_parameters)]


def gnll_loss(y, parameter_vector):
    """ Computes the mean negative log-likelihood loss of y given the mixture parameters.
    """
    alpha, mu, sigma = slice_parameter_vectors(parameter_vector)  # Unpack parameter vectors

    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=alpha),
        components_distribution=tfd.Normal(
            loc=mu,
            scale=sigma))

    log_likelihood = gm.log_prob(tf.transpose(y))  # Evaluate log-probability of y

    return -tf.reduce_mean(log_likelihood, axis=-1)


tf.keras.utils.get_custom_objects().update({'nnelu': Activation(nnelu)})

no_parameters = 3
components = 1
neurons = 200

opt = tf.keras.optimizers.Adam(1e-3)

mon = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

mdn = MDN(neurons=neurons, components=components)
mdn.compile(loss=gnll_loss, optimizer=opt)


samples = int(1e5)

x_data = np.random.sample(samples)[:, np.newaxis].astype(np.float32)
y_data = np.add(5*x_data, np.multiply((x_data)**2, np.random.standard_normal(x_data.shape)))

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.5, random_state=42)

fig = plt.figure(figsize=(x_size,y_size), dpi=dpi)
ax = plt.gca()

ax.set_title(r"$y = 5x + (x^2 * \epsilon)$"+"\n"+r"$\epsilon \backsim \mathcal{N}(0,1)$", fontsize=alt_font_size)
ax.plot(x_train,y_train, "x",alpha=1., color=sns.color_palette()[0])

remove_ax_window(ax)
plt.show()

s = np.linspace(0.,1.,int(1e3))[:, np.newaxis].astype(np.float32)

mdn.fit(x=x_train, y=y_train,epochs=200, validation_data=(x_test, y_test), callbacks=[mon], batch_size=128, verbose=0)
y_pred = mdn.predict(s)
alpha_pred, mu_pred, sigma_pred = slice_parameter_vectors(y_pred)

fig = plt.figure(figsize=(x_size, y_size), dpi=dpi)
ax = plt.gca()

ax.set_title(r"$y = 5x + (x^2 * \epsilon)$" + "\n" + r"$\epsilon \backsim \mathcal{N}(0,1)$", fontsize=alt_font_size)
ax.plot(x_train, y_train, "x", alpha=1, color=sns.color_palette()[0])

plt.plot(s, mu_pred + sigma_pred, color=sns.color_palette()[1], linewidth=5, linestyle='--', markersize=3)
plt.plot(s, mu_pred - sigma_pred, color=sns.color_palette()[1], linewidth=5, linestyle='--', markersize=3)
plt.plot(s, mu_pred, color=sns.color_palette()[1], linewidth=5, linestyle='-', markersize=3)

remove_ax_window(ax)

data_leg = mpatches.Patch(color=sns.color_palette()[0])
data_mdn = mpatches.Patch(color=sns.color_palette()[1])

ax.legend(handles=[data_leg, data_mdn],
          labels=["Data", "MDN (c=1)"],
          loc=9, borderaxespad=0.1, framealpha=1.0, fancybox=True,
          bbox_to_anchor=(0.5, -0.05), ncol=2, shadow=True, frameon=False,
          fontsize=alt_font_size)

plt.tight_layout()

if save_figure:
    plt.savefig("graphics/mdn_linear_prediction.png", format='png', dpi=dpi, bbox_inches='tight')

plt.show()


def gnll_eval(y, alpha, mu, sigma):
    """ Computes the mean negative log-likelihood loss of y given the mixture parameters.
    """
    gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=alpha),
        components_distribution=tfd.Normal(
            loc=mu,
            scale=sigma))
    log_likelihood = gm.log_prob(tf.transpose(y))
    return -tf.reduce_mean(log_likelihood, axis=-1)


def eval_mdn_model(x_test, y_test, mdn_model):
    y_pred = mdn_model.predict(x_test)
    alpha_pred, mu_pred, sigma_pred = slice_parameter_vectors(y_pred)

    print("MDN-MSE: %s" %
          (tf.losses.mean_squared_error(np.multiply(alpha_pred, mu_pred).sum(axis=-1)[:, np.newaxis], y_test).numpy(),))
    print("MDN-NLL: %s\n" % (gnll_eval(y_test.astype(np.float32), alpha_pred, mu_pred, sigma_pred).numpy()[0],))


eval_mdn_model(x_test, y_test, mdn)


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

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.5, random_state=42, shuffle=True)

fig = plt.figure(figsize=(x_size,y_size), dpi=dpi)
ax = plt.gca()

ax.set_title(r"$y = \pm x^2 + \epsilon$"+"\n"+r"$\epsilon\backsim\mathcal{N}(0,|x|)$", fontsize=alt_font_size)
ax.plot(x_train,y_train, "x",alpha=1., color=sns.color_palette()[0])

remove_ax_window(ax)
plt.show()

no_parameters = 3
components = 2
neurons = 200

mdn_2 = MDN(neurons=neurons, components=components)
mdn_2.compile(loss=gnll_loss, optimizer=opt)

s = np.linspace(-10,10,int(1e3))[:, np.newaxis].astype(np.float32)

mdn_2.fit(x=x_train, y=y_train,epochs=200, validation_data=(x_test, y_test), callbacks=[mon], batch_size=128, verbose=0)
y_pred = mdn_2.predict(s)
alpha_pred, mu_pred, sigma_pred = slice_parameter_vectors(y_pred)

fig = plt.figure(figsize=(x_size, y_size), dpi=dpi)
ax = plt.gca()

ax.set_title(r"$y = \pm x^2 + \epsilon$" + "\n" + r"$\epsilon\backsim\mathcal{N}(0,|x|)$", fontsize=alt_font_size)
ax.plot(x_train, y_train, "x", alpha=1, color=sns.color_palette()[0])

for mx in range(components):
    plt.plot(s, mu_pred[:, mx], color=sns.color_palette()[1 + mx], linewidth=5, linestyle='-', markersize=3)
    plt.plot(s, mu_pred[:, mx] - sigma_pred[:, mx], color=sns.color_palette()[1 + mx], linewidth=3, linestyle='--',
             markersize=3)
    plt.plot(s, mu_pred[:, mx] + sigma_pred[:, mx], color=sns.color_palette()[1 + mx], linewidth=3, linestyle='--',
             markersize=3)

remove_ax_window(ax)

data_leg = mpatches.Patch(color=sns.color_palette()[0])
data_mdn1 = mpatches.Patch(color=sns.color_palette()[1])
data_mdn2 = mpatches.Patch(color=sns.color_palette()[2])

ax.legend(handles=[data_leg, data_mdn1, data_mdn2],
          labels=["Data", "MDN (c=1)", "MDN (c=2)"],
          loc=9, borderaxespad=0.1, framealpha=1.0, fancybox=True,
          bbox_to_anchor=(0.5, -0.05), ncol=6, shadow=True, frameon=False,
          fontsize=alt_font_size)

plt.tight_layout()

if save_figure:
    plt.savefig("graphics/mdn_nonlinear_prediction.png", format='png', dpi=dpi, bbox_inches='tight')

plt.show()

alpha, mu, sigma = slice_parameter_vectors(mdn_2.predict([8.]))

gm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=alpha),
        components_distribution=tfd.Normal(
            loc=mu,
            scale=sigma))

x = np.linspace(0,1,int(1e3))
pyx = gm.prob(x)

fig = plt.figure(figsize=(x_size,int(y_size/2)), dpi=dpi)
ax = plt.gca()

ax.plot(x,pyx,alpha=1, color=sns.color_palette()[0], linewidth=2)

ax.set_xlabel(r"y")
ax.set_ylabel(r"$p(y|x=8)$")

remove_ax_window(ax)

plt.tight_layout()

if save_figure:
    plt.savefig("graphics/mdn_x8_density.png", format='png',dpi=dpi, bbox_inches='tight')

plt.show()