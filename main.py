# main loop
dtValues = []

avgXYError = []
avgXError = []
avgYError = []
avgAngleError = []
avgXVelError = []
avgYVelError = []
avgAngleVelError = []
mpc_dTime = 0.1

endTime = 15.0  # default = 15.0
actual_dTime = 0.001  # default = 0.001
startDt = 0.01

iters = 1
LOCALIZATION_MODE = 0  # 0 is EKF, 1 is prediction, 2 is measurement
initial_state = np.array([0, 0, 0, 0, 0, 0])
# initial_state = np.array([13.75/2, 130, np.pi/2, 0, 0, 0])

# endDt = 5.00001
ACCURACY_PATH = np.array(
    [
        [0, 0, np.pi / 2],
        [0, 25, np.pi / 6],
        [50, 50, np.pi / 6],
        [50, 75, np.pi / 2],
        [75.3, 71.7, 99999],
        [87.4, 66.7, 99999],
        [95.2, 61.7, 99999],
        [100.7, 56.7, 99999],
        [104.5, 51.7, 99999],
        [107.2, 46.7, 99999],
        [108.7, 31.7, 99999],
        [107.2, 26.7, 99999],
        [104.5, 21.7, 99999],
        [100.7, 16.7, 99999],
        [95.2, 11.7, 99999],
        [87.4, 6.7, 99999],
        [75.3, 1.7, np.pi],
    ]
)
current_path = ACCURACY_PATH

showFakeField = True
showField = True
showObstacles = True

endDt = 0.01
dtChange = 0.001
multiplier = 1.7
total_iter = 0
error_avg_list = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
while startDt <= endDt:
    xErrors.clear()
    yErrors.clear()
    xyErrors.clear()
    angleErrors.clear()
    xVelErrors.clear()
    yVelErrors.clear()
    angleVelErrors.clear()

    dtValues.append(startDt)

    # for _ in range (iters):
    #     images = []
    #     (estimatedOdometryStates, actualOdometryStates, error_list) = runIteration(startDt, 0, True)
    # error_avg_list += error_list/20.0

    images = []
    (estimatedOdometryStates, actualOdometryStates, error_list) = runIteration(
        startDt, 0, True
    )
    print("error list:" + str(error_list))
    # (estimatedOdometryStates, actualOdometryStates) = runIteration(startDt, 0, True)
    # (estimatedModelStates, actualModelStates) = runIteration(startDt, 1, True)
    # (estimatedOnlyOdometryStates, actualOnlyOdometryStates) = runIteration(startDt, 2, True)

    images = [image1Sim, image2Sim]
    # (estimatedCameraStates, actualCameraStates) = runIteration(startDt, 1, True)
    (estimatedBothMeasurementsStates, actualBothMeasurementsStates, error_list) = (
        runIteration(startDt, 0, True)
    )

    avgXYError.append(sum(xyErrors) / (iters * 1.0))
    avgXError.append(sum(xErrors) / (iters * 1.0))
    avgYError.append(sum(yErrors) / (iters * 1.0))
    avgAngleError.append(sum(angleErrors) / (iters * 1.0))
    avgXVelError.append(sum(xVelErrors) / (iters * 1.0))
    avgYVelError.append(sum(yVelErrors) / (iters * 1.0))
    avgAngleVelError.append(sum(angleVelErrors) / (iters * 1.0))

    startDt += dtChange
    dtChange *= multiplier
startDt -= dtChange / multiplier
print("cycle error average:" + str(error_avg_list))
print("-------------------------------------")
print("AVERAGE RMSE VALUES:")
print("-------------------------------------")
print("XY ERROR: " + str(statistics.mean(avgXYError)))
print("X ERROR: " + str(statistics.mean(avgXError)))
print("Y ERROR: " + str(statistics.mean(avgYError)))
print("ANGLE ERROR: " + str(statistics.mean(avgAngleError)))
print("-------------------------------------")
print("X VEL ERROR: " + str(statistics.mean(avgXVelError)))
print("Y VEL ERROR: " + str(statistics.mean(avgYVelError)))
print("ANGLE VEL ERROR: " + str(statistics.mean(avgAngleVelError)))
print("-------------------------------------")
# print("ANGLE AVERAGES: " + str(averageAngleErrorList))
# print("ANGLE MEAN: " + str(statistics.mean(angleErrors)))
# #print("ANGLE STDEV: " + str(statistics.stdev(angleErrors)))
# print("-----------------------")
# print("TRANSLATIONAL AVERAGES: " + str(averageTransErrorList))
# print("TRANSLATIONAL MEAN: " + str(statistics.mean(translationalErrors)))
# #print("TRANSLATIONAL STDEV: " + str(statistics.stdev(translationalErrors)))

plt.figure(14)
plt.title("localization type comparison")
plt.xlabel("x position")
plt.ylabel("y position")
plt.plot(current_path[:, 0], current_path[:, 1])
# plt.plot(actualModelStates[:, 0], actualModelStates[:, 1])
# plt.plot(actualOnlyOdometryStates[:, 0], actualOnlyOdometryStates[:, 1])
plt.plot(actualOdometryStates[:, 0], actualOdometryStates[:, 1])
# plt.plot(actualCameraStates[:, 0], actualCameraStates[:, 1])
plt.plot(actualBothMeasurementsStates[:, 0], actualBothMeasurementsStates[:, 1])
if showFakeField:
    plt.plot([-10, 130, 130, -10, -10], [-20, -20, 100, 100, -20], color="black")
    plt.plot(
        [image1Sim[0][0], image1Sim[1][0]],
        [image1Sim[0][1], image1Sim[1][1]],
        color="#00b900",
        linewidth=3,
    )
    plt.plot(
        [image2Sim[0][0], image2Sim[1][0]],
        [image2Sim[0][1], image2Sim[1][1]],
        color="#00b900",
        linewidth=3,
    )
plt.legend(["Target Path", "odo + model", "camera + odo + model"])
# plt.legend(['Target Path', 'model', 'odo', 'odo + model', 'camera + model', 'camera + odo + model'])


plt.figure(12)
plt.title("dt vs translation error")
plt.xlabel("dt")
plt.ylabel("RMSE translational error")
plt.xscale("log")
plt.plot(dtValues, avgXYError)

plt.figure(22)
plt.xlabel("cycle")
plt.ylabel("distance error")
odo_errors = [
    6.2699843,
    6.51979658,
    8.75518228,
    11.05853413,
    11.46729051,
    14.98201586,
    11.33114132,
    16.39483793,
    14.78910938,
    20.78359689,
]
cam_errors = (
    np.array(
        [
            0.84273461,
            7.67439305,
            1.15268751,
            13.24965737,
            1.38891814,
            12.55375929,
            1.24337586,
            11.27671568,
            1.76402486,
            14.60445325,
        ]
    )
    / 2.0
)
plt.plot([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], odo_errors)
plt.plot([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], cam_errors)
plt.legend(["model + odo", "camera + odo + model"])


# plt.figure(13)
# plt.title("accuracy test path")
# plt.xlabel("x position")
# plt.ylabel("y position")
# plt.plot(current_path[:, 0], current_path[:, 1])

# print("state length: " + str(len(state)))

# print("estimated states: " + str(state))

# BEGINNING COMMENT
# print("graphing")
# plt.figure(1)
# plt.xlabel('time')
# plt.ylabel('vel')
# plt.plot(actual_dTime * np.array(range(int(endTime / actual_dTime))), actual_states[:, 4])
# plt.plot(startDt * np.array(range(iterations)), state[:, 4])
# plt.legend(['Actual Y Vel', 'Estimated Y Vel'])

# plt.figure(2)
# plt.title('x velocity vs time')
# plt.xlabel('time')
# plt.ylabel('vel')

# plt.plot(startDt * np.array(range(iterations + 1)), ekf_robot.predicted_states[:, 3])
# plt.plot(startDt * np.array(range(iterations + 1)), actual_robot.measured_states[:, 3])
# plt.plot(actual_dTime * np.array(range(int(endTime / actual_dTime))), actual_states[:, 3])
# plt.plot(startDt * np.array(range(iterations)), state[:, 3])
# #print("x vel measurements: " + str(actual_robot.measured_states[:, 3]))
# plt.legend(['prediction', 'measurement', 'actual', 'estimation'])

# HIHIHIHIHI
# iterations = int(endTime/startDt)
# plt.figure(3)
# plt.xlabel('time')
# plt.ylabel('theta vel')
# plt.plot(actual_dTime * np.array(range(int(endTime / actual_dTime))), actualModelStates[:, 5], color = '#ff7f0e')
# plt.plot(startDt * np.array(range(iterations)), estimatedOnlyOdometryStates[:, 5], color = '#1f77b4')
# plt.plot(startDt * np.array(range(iterations)), estimatedOdometryStates[:, 5], color = '#d62728')
# plt.plot(startDt * np.array(range(iterations)), estimatedModelStates[:, 5], color = '#2ca02c')
# plt.legend(['actual', 'odo', 'odo + model', 'model'])

plt.figure(5)
plt.xlabel("x position")
plt.ylabel("y position")
plt.axis("equal")
plt.plot(current_path[:, 0], current_path[:, 1])
# plt.plot(current_path[:, 0], current_path[:, 1], color = 'blue')
# plt.plot(actualOdometryStates[:, 0], actualOdometryStates[:, 1])
plt.plot(actualOdometryStates[:, 0], actualOdometryStates[:, 1])
# barrier width = 5.56,  normal gap = 13.7,  shared gap = 13.75, hub radius = 9
if showField:
    plt.plot([0, 72, 72, 0, 0], [0, 0, 144, 144, 0], color="black")
    plt.plot(
        [image1[0][0], image1[1][0]],
        [image1[0][1], image1[1][1]],
        color="#00b900",
        linewidth=3,
    )
    plt.plot(
        [image2[0][0], image2[1][0]],
        [image2[0][1], image2[1][1]],
        color="#00b900",
        linewidth=3,
    )
if showObstacles:
    hub = plt.Circle((48, 60), 9, fill=False, linewidth=1.5)
    plt.gca().add_artist(hub)
    plt.plot(
        [13.7, 13.7, 72, 72, 13.7],
        [96 - 5.56 / 2, 96 + 5.56 / 2, 96 + 5.56 / 2, 96 - 5.56 / 2, 96 - 5.56 / 2],
        color="black",
    )
    plt.plot(
        [48 - 5.56 / 2, 48 - 5.56 / 2, 48 + 5.56 / 2, 48 + 5.56 / 2, 48 - 5.56 / 2],
        [96 + 5.56 / 2, 144 - 13.75, 144 - 13.75, 96 + 5.56 / 2, 96 + 5.56 / 2],
        color="black",
    )
plt.legend(["Target Path", "Actual Position", "Field", "Images"])

# for i in range(len(state)):
#     if (i % int(0.25/startDt) == 0):
#         plt.arrow(state[i][0], state[i][1], 4*np.cos(state[i][2]), 4*np.sin(state[i][2]), head_width = 0.5, color = 'red', fill = True)
#         #print()

# plt.figure(6)
# plt.xlabel('x position')
# plt.ylabel('y position')
# plt.title('estimated vs actual position')

# #plt.plot(ekf_robot.predicted_states[:, 0], ekf_robot.predicted_states[:, 1])
# #plt.plot(actual_robot.measured_states[:, 0], actual_robot.measured_states[:, 1])
# plt.plot(actual_states[:, 0], actual_states[:, 1])
# plt.plot(state[:, 0], state[:, 1])
# plt.legend(['Actual Position', 'Estimated Position'])
# #plt.legend(['Predicted Position', 'Measured Position','Estimated Position', 'Actual Position'])

# plt.plot([0, 72, 72, 0, 0], [0, 0, 144, 144, 0])
# plt.plot()
# plt.plot()
# plt.plot([image1[0][0], image1[1][0]], [image1[0][1], image1[1][1]], color = 'red', linewidth = 2)
# plt.plot([image2[0][0], image2[1][0]], [image2[0][1], image2[1][1]], color = 'red', linewidth = 2)


# plt.figure(7)
# plt.axes().set_aspect('equal')
# plt.xlabel('x velocity')
# plt.ylabel('y velocity')

# plt.plot(actual_robot.measured_states[:, 3], actual_robot.measured_states[:, 4])
# plt.plot(actual_states[:, 3], actual_states[:, 4])
# plt.plot(state[:, 3], state[:, 4])
# plt.plot(ekf_robot.predicted_states[:, 3], ekf_robot.predicted_states[:, 4])

# plt.legend(['Measured Velocity', 'Actual Velocity', 'Estimated Velocity', 'Predicted Velocity'])

# plt.figure(8)
# plt.xlabel('time')
# plt.ylabel('encoder')
# plt.plot(dTime * np.array(range(iterations + 1)), ekf_robot.raw_measurements[:, 5])
# plt.plot(dTime * np.array(range(iterations + 1)), ekf_robot.predicted_measurements[:, 5])
# plt.legend(['Actual Measurement', 'Predicted Measurement'])
# END COMMENT

# plt.figure(7)
# plt.xlabel("time")
# plt.ylabel("covar trace")
# plt.plot(startDt * np.array(range(len(measured_var))), measured_var)
# plt.plot(startDt * np.array(range(len(measured_var))), predicted_var)
# plt.legend(['measurement', 'prediction'])
# plt.plot(x_axis, stats.norm.pdf(x_axis, measured_vels[250][0], measure_stddev))
# plt.legend(['Measurement', 'Prediction', 'Estimate'])

# plt.figure(8)
# plt.xlabel('xPos')
# plt.ylabel('yPos')
# plt.plot(np.array(predicted_trajectory)[:, 0], np.array(predicted_trajectory)[:, 1])
# plt.plot([0, 20], [0, 20])
# plt.legend(['optimized', 'target'])

# plt.figure(9)
# plt.xlabel('time')
# plt.ylabel('theta optimization')
# plt.plot(0.2 * np.array(range(len(predicted_trajectory))), np.array(predicted_trajectory)[:, 2])

# plt.figure(10)
# plt.xlabel('x pos')
# plt.ylabel('y pos')
# print(targets)
# plt.plot((np.array(targets))[:, 0], (np.array(targets))[:, 1])
# plt.legend(['Target Positions'])


# MPC optimization trajectory
# opt_x = []
# opt_y = []
# for i in range(len(MPC_trajectory)):
#     if (i % 6 == 0):
#         opt_x.append(MPC_trajectory[i])
#     if (i % 6 == 1):
#         opt_y.append(MPC_trajectory[i])
# plt.figure(11)
# plt.plot(opt_x, opt_y)
# plt.xlabel('x pos')
# plt.ylabel('y pos')
# plt.legend(['Optimization Positions'])

plt.show()
