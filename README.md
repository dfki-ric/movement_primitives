# Install Dependencies

```bash
git clone git@git.hb.dfki.de:models-robots/ur5_fts300_2f-140.git

git clone https://github.com/rock-learning/pytransform3d.git
cd pytransform3d
pip install -e .
cd ..

git clone git@git.hb.dfki.de:motto/abstract-urdf-gripper.git
cd abstract-urdf-gripper
# TODO pending MR
#git checkout 49ba35ad
git submodule init
git submodule update
cd ..

cp rh5_left_arm.urdf abstract-urdf-gripper/urdf/
cp rh5_right_arm.urdf abstract-urdf-gripper/urdf/
cp rh5_fixed.urdf abstract-urdf-gripper/urdf/

# install hyrodyn: https://git.hb.dfki.de/skumar/hyrodyn/
```

# Environment

```bash
export PYTHONPATH=.:hyrodyn_dev/install/lib/python3.7/site-packages/hyrodyn-0.0.0-py3.7-linux-x86_64.egg:$PYTHONPATH
source hyrodyn_dev
```
