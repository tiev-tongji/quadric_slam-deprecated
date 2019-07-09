# Quadric SLAM #
This code contains a basic implementation for Cube SLAM. Given RGB and 2D object detection, the algorithm detects 3D cuboids from each frame then formulate an object SLAM to optimize both camera pose and cuboid poses. ```object_slam``` is main package. ```detect_3d_cuboid``` is the C++ version of single image cuboid detection, corresponding to a [matlab version](https://github.com/shichaoy/matlab_cuboid_detect).

**Original Authors:** [Shichao Yang](https://shichaoy.github.io./)

**Authors:** [tiev-tongji]

**Related Paper:**

* **CubeSLAM: Monocular 3D Object SLAM**, IEEE Transactions on Robotics 2019, S. Yang, S. Scherer  [**PDF**](https://arxiv.org/abs/1806.00557)

* **QuadricSLAM: Dual Quadrics From Object Detections as Landmarks in Object-Oriented SLAM**,  IEEE Robotics and Automation Letters ( Volume: 4 , Issue: 1 , Jan. 2019 ), Lachlan Nicholson, Michael Milford, Niko Sünderhauf 

If you use the code in your research work, please cite the above paper. Feel free to contact the authors if you have any further questions.



## Installation

### Prerequisites
This code contains several ros packages. We test it in **ROS indigo/kinetic/Melodic, Ubuntu 14.04/16.04/18.04, Opencv 2/3**. Create or use existing a ros workspace.
```bash
mkdir -p ~/cubeslam_ws/src
cd ~/cubeslam_ws/src
catkin_init_workspace
git clone git@github.com:shichaoy/cube_slam.git
cd cube_slam
```

### Compile dependency g2o
```bash
sh install_dependenices.sh
```

### Compile
```bash
cd ~/cubeslam_ws
catkin_make -j4
```

## Running #
```bash
source devel/setup.bash
roslaunch object_slam object_slam_example.launch
```
You will see results in Rviz. Default rviz file is for ros indigo. A kinetic version is also provided.

## Change Log

### v0.1

2019/07/09

- Finished basic structure

- Update README

### v0.0

2019/07/07

- Initial Commit

## TODO List
- [ ] Generate and update quadrics (Quadric_landmark.quadric_detection()
- [ ] Score quadrics
- [ ] Data Association
- [ ] Visualization the quadrics and camera pose(publish_all_poses())
- [ ] Offline mode

