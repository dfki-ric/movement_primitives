# Install Dependencies

```bash
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

# Install Library

I recommend to install the library via pip in editable mode:

```
pip install -e .
```

# Get URDFs

```
git clone git@git.hb.dfki.de:models-robots/ur5_fts300_2f-140.git

git clone git@git.hb.dfki.de:models-robots/rh5_models/pybullet-only-arms-urdf.git --branch develop
cd pybullet-only-arms-urdf.git
git submodule init
git submodule update
cd ..

git clone git@git.hb.dfki.de:models-robots/kuka_lbr.git

git@git.hb.dfki.de:models-objects/solar_panels.git
```

# Optional: Build Cython extensions

```bash
python setup.py build_ext --inplace
```

# Data

I assume that your data is located in the folder `data/` in most scripts.
You should put a symlink there to point to your actual data folder.
