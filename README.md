<div align="center">   

# Extended Kalman Filter State Estimation for Autonomous Robots
</div>

<h3 align="center">
  <a href="https://www.jsr.org/hs/index.php/path/article/view/5578">JSR</a> |
  <a href="https://www.youtube.com/watch?v=u2EER8b3shA">Video</a> |
  <a href="docs/ekf.pdf">Slides</a> |
  <a href="http://arxiv.org/abs/2310.04459">arXiv</a>
</h3>

![teaser](sources/ekf.png)

## Table of Contents:
1. [Abstract](#abstract)
2. [Anaysis](#analysis)
3. [News](#news)
4. [TODO](#todos)
5. [License](#license)
6. [Citation](#citation)
7. [Resource](#resource)

## Abstract <a name="high"></a>
- :robot: **Localization**: Autonomous mobile robot competitions judge based on a robot’s ability to quickly and accurately navigate the game field. This means accurate localization is crucial for creating an autonomous competition robot. Two common localization methods are odometry and computer vision landmark detection. Odometry provides frequent velocity measurements, while landmark detection provides infrequent position measurements. The state can also be predicted with a physics model. These three types of localization can be “fused” to create a more accurate state estimate using an Extended Kalman Filter (EKF). The EKF is a nonlinear full-state estimator that approximates the state estimate with the lowest covariance error when given the sensor measurements, the model prediction, and their variances.
- :trophy: **EKF**: In this research, we demonstrate the effectiveness of the EKF by implementing it on a 4-wheel mecanum-drive robot simulation. The position and velocity accuracy of fusing together various combinations of these three data sources are compared. We also discuss the assumptions and limitations of an EKF.

## Analysis <a name="analysis"></a>
- We successfully apply an Extended Kalman Filter to “fuse'' odometry and computer vision landmark measurements. 
- The EKF definitely provides good results for a mecanum drivetrain, as quantified by the decrease in RMSE when data is fused, and can be applied to the FTC competition.
- EKF greatly increases localization accuracy while not requiring much computation power

## News <a name="news"></a>
- **`Jan-Apr 2024`** EKF v2.0 for [FIRST Robotics Competition](https://www.firstinspires.org/robotics/frc) and [FIRST Tech Challenge](https://www.firstinspires.org/robotics/ftc) deployment.
- **`Sep-Dec 2023`** Start upgrading EKF for [FIRST Robotics Competition](https://www.firstinspires.org/robotics/frc) autonomous driving.
- **`Jan-Oct 2023`** EKF paper [Journal of Student Research](https://www.jsr.org/hs/index.php/path/article/view/5578) publication with minor update.
- **`Jan-Mar 2023`** Present EKF at [Polygence Symposium](https://www.youtube.com/watch?v=u2EER8b3shA) and [Synopsys Science and Technology Championship](https://science-fair.org).
- **`Oct-Dec 2022`** EKF v1.0 code implementation and system simulation with bugs' fix.
- **`Jun-Dec 2022`** Polygence research for [FIRST Tech Challenge](https://www.firstinspires.org/robotics/ftc) development and arXiv preparation.

## TODO <a name="todos"></a>
- [ ] Implement an EKF on a physical robot with a more realistic model that includes acceleration and other factors such as friction and motor voltages.
- [ ] A more accurate physics model can be obtained by training a machine learning model with human driving.
- [ ] Another state estimation algorithm to look into is the particle filter, as it can perform state estimation on systems with non- Gaussian noise.
- [x] Bugs fix

## License <a name="license"></a>
All assets and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.

## Citation <a name="citation"></a>
Please consider citing our paper if the project helps your research with the following information:
```
Kou, E., & Haggenmiller, A. (2023).
Extended Kalman Filter State Estimation for Autonomous Competition Robots.
Journal of Student Research, 12(1).
https://doi.org/10.47611/jsrhs.v12i1.5578
```
As well as citation in IEEE/ACS/ABNT formats as reference:
```
E. Kou and A. Haggenmiller, “Extended Kalman Filter State Estimation for Autonomous Competition Robots”, J Stud Res, vol. 12, no. 1, Feb. 2023.
Kou, E.; Haggenmiller, A. Extended Kalman Filter State Estimation for Autonomous Competition Robots. J Stud Res 2023, 12.
KOU, E.; HAGGENMILLER, A. Extended Kalman Filter State Estimation for Autonomous Competition Robots. Journal of Student Research, v. 12, n. 1, 28 Feb. 2023.
```

## Resource
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
- [PID](https://github.com/BubblyBingBong/PID) ( :rocket:Ours!)
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems.
- Franklin, W. Kalman Filter Explained Simply. The Kalman Filter.
- FIRST® (2021). 2021-2022 FIRST® Tech Challenge Game Manual Part 2 – Traditional Events.
- Thrun, S. (2002). Probabilistic robotics. Communications of the ACM, 45(3),52-57.
- Taheri, H., Qiao, B., & Ghaeminezhad, N.(2015). Kinematic model of a four mecanum wheeled mobile robot. International journal of computer applications, 113(3), 6-9.
- Olson, E. (2004). A primer on odometry and motor control. Electronic Group Discuss, 12.
- Coulter, R. C. (1992). Implementation of the pure pursuit path tracking algorithm. Carnegie-Mellon UNIV Pittsburgh PA Robotics INST.
