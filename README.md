<div align="center">   
  
# Extended Kalman Filter State Estimation for Autonomous Robots
</div>

<h3 align="center">
  <a href="https://arxiv.org">arXiv</a> |
  <a href="https://www.youtube.com/watch?v=u2EER8b3shA">Video</a> |
  <a href="docs/ekf.pdf">Slides</a>
</h3>

![teaser](sources/ekf.png)

## Table of Contents:
1. [Highlights](#high)
2. [News](#news)
3. [Getting Started](#start)
4. [Anaysis](#analysis)
5. [TODO](#todos)
6. [License](#license)
7. [Citation](#citation)
8. [Resource](#resource)

## Highlights <a name="high"></a>

- 🤖 **Planning-oriented philosophy**: Autonomous mobile robot competitions judge based on a robot’s ability to quickly and accurately navigate the game field. This means accurate localization is crucial for creating an autonomous competition robot. Two common localization methods are odometry and computer vision landmark detection. Odometry provides frequent velocity measurements, while landmark detection provides infrequent position measurements. The state can also be predicted with a physics model. These three types of localization can be “fused” to create a more accurate state estimate using an Extended Kalman Filter (EKF). The EKF is a nonlinear full-state estimator that approximates the state estimate with the lowest covariance error when given the sensor measurements, the model prediction, and their variances. In this paper, we demonstrate the effectiveness of the EKF by implementing it on a 4-wheel mecanum-drive robot simulation. The position and velocity accuracy of fusing together various combinations of these three data sources are compared. We also discuss the assumptions and limitations of an EKF.

## News <a name="news"></a>

- **`2023/09`** EKF paper is available on arXiv.

## Getting Started <a name="start"></a>

## Analysis <a name="analysis"></a>

## TODO <a name="todos"></a>
- [ ] All configs & checkpoints
- [x] Bug fixes

## License <a name="license"></a>

All assets and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.

## Citation <a name="citation"></a>

Please consider citing our paper if the project helps your research with the following BibTex:

```bibtex
```

## Resource

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
