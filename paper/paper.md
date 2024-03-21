---
title: 'movement_primitives: Imitation Learning of Cartesian Motion with Movement Primitives'
tags:
  - Python
  - robotics
  - imitation learning
  - dynamical movement primitive
  - probabilistic movement primitive
authors:
  - name: Alexander Fabisch
    orcid: 0000-0003-2824-7956
    affiliation: 1
affiliations:
 - name: Robotics Innovation Center, German Research Center for Artificial Intelligence (DFKI GmbH), Bremen, Germany
   index: 1
date: 21 March 2024
bibliography: paper.bib
---

# Summary

Movement primitives are a common representation of movements in robotics
[@Maeda2017] for imitation learning, reinforcement learning, and black-box
optimization of behaviors.
There are many types and variations. The Python library *movement_primitives*
focuses on imitation learning, generalization, and adaptation of movement
primitives in Cartesian space. It implements dynamical movement primitives,
probabilistic movement primitives, as well as Cartesian and dual Cartesian
movement primitives with coupling terms to constrain relative movements in
bimanual manipulation. They are implemented in Cython to speed up online
execution and batch processing in an offline setting. In addition, the
library provides tools for data analysis and movement evaluation.

# Statement of Need

Movement primitives are a common group of policy representations in robotics.
Although movement primitives are limited in their capacity to represent
behavior that takes into account complex sensor data during execution in
comparison to general function approximators such as neural networks, several
instances (e.g., dynamical movement primitives) have proven to be a reliable
and effective tool in robot learning.
A reliable tool deserves a similarly reliable implementation. However, there
are only a few actively maintained, documented, and easy to use
implementations. One of these is *movement_primitives*, which we present in
this article.

# Movement Primitives

Dynamical Movement Primitives (DMPs) are the most prominent example of
movement primitives [@Ijspeert2002; @Ijspeert2013]. From a high-level
perspective [@Fabisch2014], a DMP is a policy
$$\boldsymbol{x}_{t+1} = \pi_{\boldsymbol{w}, \boldsymbol{v}}(\boldsymbol{x}_t, t),$$
where $\boldsymbol{x}_t$ is the state of an agent (position, velocity, and
acceleration) at time $t$, $\boldsymbol{w}$ are the weights (parameters) that
define the shape of the movement, and $\boldsymbol{v}$ are meta-parameters.
The exact definition of the meta-parameters $\boldsymbol{v}$ depends on the
DMP type, but most types allow to set the initial state $\boldsymbol{x}_0$,
the final state $\boldsymbol{g}$, and the duration of the movement $\tau$. A
DMP generates a trajectory in state space so that a controller that translates
states $\boldsymbol{x}_t, \boldsymbol{x}_{t+1}$ to control commands is required.

DMPs have been used for imitation learning, in which one demonstration is
enough to learn a DMP. DMPs can also be used in a reinforcement learning
setting, in which the weights of the DMP or the meta-parameters can be
learned.
@Saveriano2023 provide a survey of DMPs and how they can be used.

In the *movement_primitives* library, we implement several types that are
important for Cartesian movement generation: an extension that includes the
final velocity as a meta-parameter [@Muelling2013], DMPs for Cartesian poses
in three dimensions with unit quaternions [@Ude2014], and DMPs that define
bimanual movements by introducing a coupling term that controls the relative
motion of two arms [@Gams2013].

Another type of movement primitives implemented in this library are
Probabilistic Movement Primitives (ProMPs) [@Paraschos2013] that capture
the distribution of multiple demonstrations. Their probabilistic formulation
allows to modify movements by conditioning, for instance, on viapoints.

# Implementations of Movement Primitives

The *movement_primitives* library is a reimplementation and extension of the
movement primitive features of BOLeRo [@Fabisch2020]. BOLeRo is a C++/Python
framework for behavior learning and optimization. However, the focus is very
broad and more on reinforcement learning and behavior parameter optimization
than on imitation learning.

Another similar library is dmpbbo [@Stulp2019], which has a general DMP
implementation and additional components to optimize the parameters of DMPs in
reinforcement learning settings. The library is designed to train DMPs in
Python and execute them in C++.
Both implementations are not well-suited for imitation learning because
additional tooling for data analysis and deployment is required. Switching
between C++ and Python is also not convenient for various reasons: building
and installing these packages is complicated, continuous integration is hard
to set up, code maintenance is complicated, and it does not integrate easily
with the Python scientific ecosystem.

There are more implementations listed by @Saveriano2023 (available at
https://gitlab.com/dmp-codes-collection/third-party-dmp). A lot of these are
examplary Matlab scripts and not maintained anymore, or only implementations
of specific papers.
Other libraries do not support Cartesian movement primitives, which are only
available in BOLeRo and *movement_primitives*. The latter also supports
bimanual movements through dual Cartesian DMPs.

# Design and Features
The main contributions of *movement_primitives* are (1) a fast Python-only
library for movement primitives, and (2) robust implementations of several
types of movement primitives (see \autoref{tab:features}).
Our focus is on Cartesian movement primitives that are used to control one or
two robotic arms and offer exemplary implementations of coupling terms for
Cartesian (bimanual) DMPs. These can be used for obstacle avoidance and to
constrain dual arm motions to relative positions and/or orientations.

:Overview of implemented movement primitives.\label{tab:features}

| Class                | Description                | Publication    |
|----------------------|----------------------------|----------------|
| DMP                  | Standard DMP               | @Ijspeert2013  |
| DMP                  | Smooth spatial scaling     | @Pastor2009    |
| DMPWithFinalVelocity | Allows final velocity      | @Muelling2013  |
| CartesianDMP         | DMP of Cartesian poses     | @Ude2014       |
| DualCartesianDMP     | DMP of two Cartesian poses | @Gams2013      |
| ProMP                | Standard ProMP             | @Paraschos2013 |

Furthermore, *movement_primitives* supports the whole imitation learning
pipeline, including data analysis through plotting and visualization (based on
pytransform3d [@Fabisch2019] and Open3D [@Zhou2018]), data preprocessing for
imitation learning, good integration with the scientific ecosystem in Python,
simulation of learned movement primitives (in PyBullet [@Coumans2021]), export
to permanent data formats (pickle, JSON, YAML), and analysis of kinematic
feasibility.
Although it has several dependencies and requires compilation because of its
Cython [@Dalcin2011] components, it is possible to simply install it with pip
from PyPI.

# Example: Rotating a Compact Solar Panel with a Humanoid

\autoref{fig:rh5} and \autoref{fig:rh5_viz} show a humanoid robot
rotating an object with two hands. The movement is generated by a dual
Cartesian DMP trained on a demonstrated rotation movement. The width of the
object is known. Hence, it can easily be adapted for similar objects with a
different size through a coupling term defined by [@Gams2013].

![RH5 Manus [@Boukheddimi2022] rotating a compact solar panel.\label{fig:rh5}](rh5_panel_0){ width=50% }

![Visualization of similar rotation trajectory with another humanoid robot.\label{fig:rh5_viz}](dual_cart_dmps_rh5_open3d){ width=50% }

A similar task has been solved by @Mronga2021 with two Kuka iiwa arms. They
record a dataset for different panel sizes via kinesthetic teaching and use
Gaussian mixture regression to represent the distribution of solutions and
condition it on the object width to generalize. This is easier with ProMPs:
for each demonstration, we compute ProMP weights, concatenate them with the
task parameters over which we want to generalize, and learn a Gaussian mixture
model, which we can condition on task parameters to generate ProMPs that
define trajectory distributions to solve these tasks (\autoref{fig:kuka_mean}
and \autoref{fig:kuka_ellipsoids}).

![Mean trajectories for conditional ProMPs and panel widths 30/40/50 cm.\label{fig:kuka_mean}](contextual_promps_kuka_panel_width_open3d){ width=80% }

![At each step, the position distribution defined by the conditioned ProMP is indicated by an equiprobable ellipsoid. The arms are at the mean start position for width 50 cm.\label{fig:kuka_ellipsoids}](contextual_promps_kuka_panel_width_open3d2){ width=50% }

# Benchmark of DMP Implementations

Since execution speed of DMPs is relevant in robotics, we compare several DMP
implementations from dmpbbo and *movement_primitives*. For this purpose, we
create a minimum jerk trajectory of $N$ dimensions that moves from
$\boldsymbol{0} \in \mathbb{R}^N$ to $\boldsymbol{1} \in \mathbb{R}^N$ in one
second, train a DMP on it, and execute the DMP step by step. We use $M$
weights per dimension, and step through the DMP with $\Delta t = 0.001 s$. The
concept of dmpbbo is to train in Python and run DMPs in C++. We still analyze
the Python version and the C++ version of dmpbbo as well as
*movement_primitives* with various implementations of the integration (Euler
integration with $h=0.1 \cdot \Delta t$ and RK4 integration, both in Python
and Cython). The default integration method of dmpbbo is RK4. Results for
varying configurations of $N$ and $M$ are summarized in
\autoref{fig:timing_execution_weights}, \autoref{fig:timing_execution_dims}
and \autoref{tab:benchmark_results}. While the number of weights per dimension and the
number of dimensions have a considerable influence on the runtime of dmpbbo,
the influence on the runtime of *movement_primitives* is negligible because
NumPy [@Harris2020] vectorization is used.  More specifically, computing all
steps of a DMP with 1 s duration at 1 kHz ($\Delta t = 0.001 s$) with $N=50$
dimensions and $M=60$ weights per  dimension takes $0.0822 \pm 0.0015 s$ with
the *movement_primitives* library and RK4 integration in Cython, which means
$8.51\%$ of the DMP's runtime is spent on computing steps. This allows online
adaptation of the trajectory. dmpbbo's C++ implementation is the best
candidate for a low number of dimensions and weights per dimension. In this
domain it outperforms all other implementations by a considerable margin.
However, it scales linearly with these numbers. Hence, it is considerably
slower for $N=50$ and $M=60$ than any RK4 implementation of
*movement_primitives*. The Python version of dmpbbo is not able to run some
configurations in real time. For example, $N=6, M=30$ needs
$5.9292 \pm 0.0955 s$ to compute.

![Benchmark of execution speed for various DMP implementations and
configurations. Each bar shows an average over 100 stepwise executions of a
DMP. Varying number of weights per dimension $M$, number of dimensions
$N=6$.\label{fig:timing_execution_weights}](timing_execution_weights.pdf)

![Benchmark of execution speed for various DMP implementations and
configurations. Each bar shows an average over 100 stepwise executions of a
DMP. Varying number of dimensions $N$, number of weights per dimension
$M=30$.\label{fig:timing_execution_dims}](timing_execution_dimensions.pdf)

:Benchmark results for DMP execution. Best performance per setup in **bold**.\label{tab:benchmark_results}

| Library             | Implementation | $N$ | $M$ | Time $\mu \pm \sigma$ [s] |
|---------------------|----------------|-----|-----|---------------------------|
| dmpbbo              | C++            | 3   | 10  | **0.0027** $\pm$ 0.0001   |
|                     |                |     | 30  | **0.0077** $\pm$ 0.0001   |
|                     |                |     | 60  | **0.0144** $\pm$ 0.0004   |
|                     |                | 6   | 10  | **0.0049** $\pm$ 0.0001   |
|                     |                |     | 30  | **0.0146** $\pm$ 0.0002   |
|                     |                |     | 60  | **0.0300** $\pm$ 0.0052   |
|                     |                | 15  | 10  | **0.0129** $\pm$ 0.0028   |
|                     |                |     | 30  | **0.0376** $\pm$ 0.0059   |
|                     |                |     | 60  | **0.0729** $\pm$ 0.0103   |
|                     |                | 50  | 10  | **0.0401** $\pm$ 0.0068   |
|                     |                |     | 30  | 0.1236 $\pm$ 0.0174       |
|                     |                |     | 60  | 0.2405 $\pm$ 0.0308       |
| dmpbbo              | Python         | 3   | 10  | 0.8137 $\pm$ 0.0164       |
|                     |                |     | 30  | 1.6986 $\pm$ 0.0319       |
|                     |                |     | 60  | 3.0244 $\pm$ 0.0454       |
|                     |                | 6   | 10  | 1.3946 $\pm$ 0.0228       |
|                     |                |     | 30  | 3.1676 $\pm$ 0.0746       |
|                     |                |     | 60  | 5.9292 $\pm$ 0.0955       |
|                     |                | 15  | 10  | 3.2079 $\pm$ 0.0593       |
|                     |                |     | 30  | 7.4972 $\pm$ 0.1366       |
|                     |                |     | 60  | 14.2590 $\pm$ 0.2811      |
|                     |                | 50  | 10  | 9.7134 $\pm$ 0.0448       |
|                     |                |     | 30  | 24.6018 $\pm$ 2.0579      |
|                     |                |     | 60  | 47.4420 $\pm$ 2.0075      |
| movement_primitives | euler-cython   | 3   | 10  | 0.1946 $\pm$ 0.0019       |
|                     |                |     | 30  | 0.2223 $\pm$ 0.0070       |
|                     |                |     | 60  | 0.2234 $\pm$ 0.0031       |
|                     |                | 6   | 10  | 0.1912 $\pm$ 0.0033       |
|                     |                |     | 30  | 0.2301 $\pm$ 0.0043       |
|                     |                |     | 60  | 0.2306 $\pm$ 0.0060       |
|                     |                | 15  | 10  | 0.2117 $\pm$ 0.0067       |
|                     |                |     | 30  | 0.2334 $\pm$ 0.0041       |
|                     |                |     | 60  | 0.2310 $\pm$ 0.0013       |
|                     |                | 50  | 10  | 0.2260 $\pm$ 0.0009       |
|                     |                |     | 30  | 0.2547 $\pm$ 0.0273       |
|                     |                |     | 60  | 0.2529 $\pm$ 0.0044       |
| movement_primitives | rk4-cython     | 3   | 10  | 0.0447 $\pm$ 0.0006       |
|                     |                |     | 30  | 0.0737 $\pm$ 0.0018       |
|                     |                |     | 60  | 0.0760 $\pm$ 0.0003       |
|                     |                | 6   | 10  | 0.0471 $\pm$ 0.0036       |
|                     |                |     | 30  | 0.0733 $\pm$ 0.0003       |
|                     |                |     | 60  | 0.0761 $\pm$ 0.0003       |
|                     |                | 15  | 10  | 0.0468 $\pm$ 0.0022       |
|                     |                |     | 30  | 0.0754 $\pm$ 0.0005       |
|                     |                |     | 60  | 0.0776 $\pm$ 0.0002       |
|                     |                | 50  | 10  | 0.0752 $\pm$ 0.0002       |
|                     |                |     | 30  | **0.0794** $\pm$ 0.0063   |
|                     |                |     | 60  | **0.0822** $\pm$ 0.0015   |

# Conclusion

Although movement primitives are a popular tool in robot learning, there is a
lack of well maintained implementations in particular for bimanual and
Cartesian movements. *movement_primitives* provides a well-tested, robust
implementation of various movement primitives with the goal of generating
Cartesian robot movements. It integrates well with the existing Python
scientific ecosystem.

# Acknowledgements

This work was supported by a grant of the German Federal Ministry of Economic
Affairs and Energy (BMWi, FKZ 50 RA 1701) and by the European Commission under
the Horizon 2020 framework program for Research and Innovation (project
acronym: APRIL, project number: 870142).

# References