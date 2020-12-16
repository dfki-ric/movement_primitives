# Install Dependencies

```bash
# Python packages
git clone https://github.com/rock-learning/pytransform3d.git
cd pytransform3d
pip install -e .
cd ..

git clone https://github.com/AlexanderFabisch/gmr.git --branch feature/unscented_transform
cd gmr
pip install -e .
cd ..

git clone git@git.hb.dfki.de:dfki-interaction/experimental/mocap.git
cd mocap
pip install -e .
cd ..

# URDFs
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

# optional:
# install hyrodyn: https://git.hb.dfki.de/skumar/hyrodyn/
```

# Optional: Build Cython extensions

```bash
python setup.py build_ext --inplace
```

# Environment

```bash
export PYTHONPATH=.:$PYTHONPATH

# optional:
export PYTHONPATH=.:hyrodyn_dev/install/lib/python3.7/site-packages/hyrodyn-0.0.0-py3.7-linux-x86_64.egg:$PYTHONPATH  # HACK
source hyrodyn_dev/env.sh
```
