# Movement Primitives

## Features

* Dynamical Movement Primitives (DMPs) for
    * positions (with fast Runge-Kutta integration)
    * Cartesian position and orientation (with fast Cython implementation)
    * Dual Cartesian position and orientation (with fast Cython implementation)
* Coupling terms for synchronization of position and/or orientation of dual Cartesian DMPs
* Propagation of DMP weight distribution to state space distribution
* Probabilistic Movement Primitives (ProMPs)

## Install Dependencies

```bash
# untested: pip install git+https://git.hb.dfki.de/dfki-interaction/experimental/mocap.git
git clone git@git.hb.dfki.de:dfki-interaction/experimental/mocap.git
cd mocap
pip install -e .
cd ..

# optional: install pytransform3d from source
git clone https://github.com/rock-learning/pytransform3d.git
cd pytransform3d
pip install -e .
cd ..
```

## Install Library

I recommend to install the library via pip in editable mode:

```
pip install -e .
```

## Get URDFs

```
# UR5
git clone git@git.hb.dfki.de:models-robots/ur5_fts300_2f-140.git

# RH5
git clone git@git.hb.dfki.de:models-robots/rh5_models/pybullet-only-arms-urdf.git --branch develop --recursive

# Kuka
git clone git@git.hb.dfki.de:models-robots/kuka_lbr.git

# Solar panel
git clone git@git.hb.dfki.de:models-objects/solar_panels.git

# RH5 Gripper
git clone git@git.hb.dfki.de:motto/abstract-urdf-gripper.git --recursive
```

## Optional: Build Cython extensions

```bash
python setup.py build_ext --inplace
```

## Data

I assume that your data is located in the folder `data/` in most scripts.
You should put a symlink there to point to your actual data folder.

<img src="doc/source/_static/contextual_promps_kuka_panel_width_open3d.png" width="400px" />

<img src="doc/source/_static/contextual_promps_kuka_panel_width_open3d2.png" width="400px" />

<img src="doc/source/_static/coupled_dual_cart_dmps_gripper_open3d.png" width="400px" />

<img src="doc/source/_static/coupled_dual_cart_dmps_rh5_pybullet.png" width="400px" />

<img src="doc/source/_static/dmp_state_space_distribution_kuka_peginhole_open3d.png" width="400px" />

<img src="doc/source/_static/dual_cart_dmps_rh5_open3d.png" width="400px" />

<img src="doc/source/_static/dual_cart_dmps_rh5_pybullet.png" width="400px" />
