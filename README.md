# Install Dependencies

```bash
git clone https://github.com/AlexanderFabisch/gmr.git --branch feature/unscented_transform
cd gmr
pip install -e .
cd ..

git clone git@git.hb.dfki.de:dfki-interaction/experimental/mocap.git
cd mocap
pip install -e .
cd ..

# optional: install pytransform3d from source
git clone https://github.com/rock-learning/pytransform3d.git --branch feature/batch_conversions
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

git clone git@git.hb.dfki.de:motto/abstract-urdf-gripper.git
cd abstract-urdf-gripper
git submodule init
git submodule update
cd ..
cp rh5_left_arm.urdf abstract-urdf-gripper/urdf/
cp rh5_right_arm.urdf abstract-urdf-gripper/urdf/
cp rh5_fixed.urdf abstract-urdf-gripper/urdf/

git clone git@git.hb.dfki.de:models-robots/kuka_lbr.git
```

# Optional: Build Cython extensions

```bash
python setup.py build_ext --inplace
```
