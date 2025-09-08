import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from satellite import Satellite, ControllerMap
from controllers import *
from igrf_wrapper import *
from frames import *
from quaternions import get_random_unit_quaternion, rotation_matrix_to_quaternion
from orbit import Orbit, RADIUS_EARTH
from coordinates import xyz_to_spherical

np.set_printoptions(precision=2, formatter={'float_kind': lambda x: "%.2e" % x})

"""
NOTES

- with pid the control loop needs to run with at least 10 Hz with the current parameters
    @todo
    * play around with the gains
    * play around with the derivative gain low pass filter time constant


"""

# SIMULATION SETTINGS
# -------------------

SIMULATION_TIME = 1  # hours
SIMULATION_STEP = 0.1  # seconds
OMEGA_MAX = 0.1  # rad/s
OMEGA_MAX_DETUMBLED = 0.01  # rad/s
INERTIA_DIAG_MIN = 0.001
INERTIA_DIAG_MAX = 0.0035  # kg * m^2
PRINT = False  # enables real-time status prints
MIN_ALTITUDE = 300  # km
MAX_ALTITUDE = 600  # km
RANDOM = True  # if orbit parameters are generated randomly
SKIP_DETUMBLE = True  # skips the detumbling part, initial angular velocity already low
UPDATE_TARGET = True
UPDATE_TARGET_DT = 1  # s

# POINT TO BE TRACKED
# -------------------

# @todo implement this functionality
LATITUDE_POINT = ...
LONGITUDE_POINT = ...


# CONTROLLER SETTINGS
# -------------------

# BDOT
BDOT_GAIN = 7e-5

# PID
KP = 8e-2
KI = 0
KD = 6e-2

# KP = 1.6e-2
# KD = 6e-2

# LQR
Q_omega = np.eye(3) * 5
Q_attitude = np.eye(3) * 5
Q = np.block([[Q_omega, np.zeros(shape=(3,3))], [np.zeros(shape=(3,3)), Q_attitude]])
# R = Q_attitude
R = np.eye(3) * 8

# -------------------

def print_status(date, altitude, latitude, longitude, attitude, omega):
    print(f"date: {date}\
            \nlatitude: {latitude:.2f} deg\
            \nlongitude: {longitude:.2f} deg\
            \naltitude: {altitude:.2f} km\
            \nomega: {omega} (x, y, z) rad/s\
            \nattitude: {np.rad2deg(attitude.to_euler())} (roll, pitch, yaw) deg")


def get_position_geodetic(r_ecef) -> tuple:
    """
    expects position vector [x, y, z] in ECEF

    returns altitude, latitude, longitude (km, degrees)
    """
    altitude, latitude, longitude = xyz_to_spherical(r_ecef)
    altitude -= RADIUS_EARTH
    latitude_deg = np.rad2deg(latitude)
    longitude_deg = np.rad2deg(longitude)
    return altitude, latitude_deg, longitude_deg


def quaternion_max_unflip(q_prev: Quaternion, q: Quaternion) -> Quaternion:
    # approach suggest on stack overflow for quaternion flip avoidance
    i = np.argmax(np.abs(q.q))
    if q_prev.q[i] * q.q[i] < 0:
        q *= -1
    return q


def main():
    date = datetime.now()
    
    inertia_tensor = np.zeros(shape=(3, 3))

    # generate random parameters for orbit
    if RANDOM:
        eccentricity = np.random.uniform(0.001, 0.05)
        inclination = np.random.uniform(20, 98)
        argument_of_periapsis = np.random.uniform(0, 360)
        raan = np.random.uniform(0, 360)
        initial_true_anomaly = np.random.uniform(0, 360)
        earth_rotation_angle = np.random.uniform(0, 360)
        semi_major_axis = np.random.uniform(MIN_ALTITUDE, MAX_ALTITUDE) + RADIUS_EARTH
        np.fill_diagonal(inertia_tensor, np.random.uniform(INERTIA_DIAG_MIN, INERTIA_DIAG_MAX, 3))
    else:
        eccentricity = 0.001
        inclination = 97.4
        argument_of_periapsis = 90
        raan = 0
        initial_true_anomaly = 0
        earth_rotation_angle = 0
        semi_major_axis = (MIN_ALTITUDE + MAX_ALTITUDE) / 2 + RADIUS_EARTH
        np.fill_diagonal(inertia_tensor, (INERTIA_DIAG_MIN + INERTIA_DIAG_MAX) / 2)

    orbit = Orbit(semi_major_axis,
                    inclination,
                    eccentricity,
                    argument_of_periapsis,
                    raan,
                    initial_true_anomaly,
                    earth_rotation_angle)

    # generate random initial parameters for the satellite
    if SKIP_DETUMBLE:
        omega = np.random.uniform(-OMEGA_MAX_DETUMBLED, OMEGA_MAX_DETUMBLED, 3)
        state = "COARSE_POINT_NADIR"
    else:
        omega = np.random.uniform(-OMEGA_MAX, OMEGA_MAX, 3)
        state = "DETUMBLE"

    q_body_to_eci = get_random_unit_quaternion()
    inertia_tensor = np.zeros(shape=(3, 3))
    np.fill_diagonal(inertia_tensor, np.random.uniform(INERTIA_DIAG_MIN, INERTIA_DIAG_MAX, 3))

    # get initial B-field
    r_eci, v_eci = orbit.propagate(0)
    r_ecef = orbit.vector_eci_to_ecef(0, r_eci)
    altitude, latitude, longitude = get_position_geodetic(r_ecef)
    b_field =  body_to_ned(get_b_field_NED(latitude, longitude, altitude, date), q_body_to_eci)

    bdot_controller = BDot(BDOT_GAIN, b_field)
    pid_controller = PID(kp=KP, ki=KI, kd=KD, tau_derivative=0.1)

    # @todo understand why the LQR currently only works with these units
    lqr_controller = LQR_Yang(R=R, Q=Q, J=inertia_tensor)

    # map controllers to states
    controllers: ControllerMap = {
        "DETUMBLE": bdot_controller,
        "COARSE_POINT_NADIR": pid_controller,
        "FINE_POINT_NADIR": lqr_controller
    }

    satellite = Satellite(q_body_to_eci, omega, inertia_tensor, controllers, OMEGA_MAX_DETUMBLED, b_field, state)

    total_time = SIMULATION_TIME * 3600 # total simulation time
    dt = SIMULATION_STEP
    num_steps = int(total_time / dt)

    # lists for plotting
    angular_speeds = {
        "DETUMBLE": [],
        "COARSE_POINT_NADIR": [],
        "FINE_POINT_NADIR": []
    }

    commanded_torques = []
    applied_torques = []
    target_q_vectors = []
    error_q_vectors = []
    attitude_q_vectors = [] # New list for actual attitude quaternions
    omega_vectors = []
    b_field_x = []
    b_field_y = []
    b_field_z = []
    pid_proportional_terms_x = []
    pid_proportional_terms_y = []
    pid_proportional_terms_z = []
    pid_integral_terms_x = []
    pid_integral_terms_y = []
    pid_integral_terms_z = []
    pid_derivative_terms_x = []
    pid_derivative_terms_y = []
    pid_derivative_terms_z = []
    
    print(f"Simulation starting at")
    print_status(date, altitude, latitude, longitude, q_body_to_eci, omega)

    # q_body_to_eci_target = get_random_unit_quaternion()
    q_body_to_eci_target = Quaternion()
    q_body_to_eci_target_prev = Quaternion()

    print(f'initial attitude is {q_body_to_eci}')
    print(f'target attitude is {q_body_to_eci_target}')

    # simulation loop
    t_update_target = 0
    for step in range(1, num_steps):
        date += timedelta(seconds=dt)
        t = dt * step
        r_eci, v_eci = orbit.propagate(t)

        r_ecef = orbit.vector_eci_to_ecef(t, r_eci)
        altitude, latitude, longitude = get_position_geodetic(r_ecef)
        
        # update satellite B field
        satellite.B_field_gauss = body_to_ned(get_b_field_NED(latitude, longitude, altitude, date), satellite.q_body_to_eci)
        
        x_lvlh, y_lvlh, z_lvlh = eci_to_lvlh(r_eci, v_eci)
        R_eci_to_lvlh = get_rotation_matrix(x_lvlh, y_lvlh, z_lvlh)
        q_eci_to_lvlh = rotation_matrix_to_quaternion(R_eci_to_lvlh)
        q_eci_to_lvlh.normalize()

        omega_lvlh_in_eci = orbit.angular_rate * y_lvlh
        omega_lvlh_in_body = satellite.q_body_to_eci.to_rotation_matrix().T @ omega_lvlh_in_eci

        if UPDATE_TARGET and (t_update_target == 0 or (t - t_update_target >= UPDATE_TARGET_DT) and (satellite.state == "FINE_POINT_NADIR" or satellite.state == "COARSE_POINT_NADIR")):
            q_body_to_lvlh_target = Quaternion(1, 0, 0, 0)  # unit quaternion for nadir pointing
            q_body_to_eci_target = q_eci_to_lvlh.get_conjugate() * q_body_to_lvlh_target
            q_body_to_eci_target.normalize()
            q_body_to_eci_target = quaternion_max_unflip(q_body_to_eci_target_prev, q_body_to_eci_target)
            t_update_target = t
            q_body_to_eci_target_prev = q_body_to_eci_target


        if np.any(np.isnan(satellite.omega)):
            print(f"\nSIMULATION BLEW UP ON STEP {step}!\n")
            break

        commanded_torque, applied_torque = satellite.update(dt, orbit.angular_rate, q_body_to_eci_target)  # this is the main iteration call

        commanded_torques.append(commanded_torque)
        applied_torques.append(applied_torque)
        angular_speeds[satellite.state].append(np.linalg.norm(satellite.omega))
        error_q_vectors.append(satellite.q_body_to_eci_error.vector)
        omega_vectors.append(satellite.omega)
        target_q_vectors.append(q_body_to_eci_target.vector)
        attitude_q_vectors.append(satellite.q_body_to_eci.vector)

        b_field_x.append(satellite.B_field_gauss[0])
        b_field_y.append(satellite.B_field_gauss[1])
        b_field_z.append(satellite.B_field_gauss[2])

        if satellite.state == "COARSE_POINT_NADIR":
            pid_proportional_terms_x.append(pid_controller.proportional_terms[-1][0])
            pid_proportional_terms_y.append(pid_controller.proportional_terms[-1][1])
            pid_proportional_terms_z.append(pid_controller.proportional_terms[-1][2])
            pid_integral_terms_x.append(pid_controller.integral_terms[-1][0])
            pid_integral_terms_y.append(pid_controller.integral_terms[-1][1])
            pid_integral_terms_z.append(pid_controller.integral_terms[-1][2])
            pid_derivative_terms_x.append(pid_controller.derivative_terms[-1][0])
            pid_derivative_terms_y.append(pid_controller.derivative_terms[-1][1])
            pid_derivative_terms_z.append(pid_controller.derivative_terms[-1][2])
        else:
            pid_proportional_terms_x.append(0)
            pid_proportional_terms_y.append(0)
            pid_proportional_terms_z.append(0)
            pid_integral_terms_x.append(0)
            pid_integral_terms_y.append(0)
            pid_integral_terms_z.append(0)
            pid_derivative_terms_x.append(0)
            pid_derivative_terms_y.append(0)
            pid_derivative_terms_z.append(0)

        if PRINT:
            print(f"step {step}")
            print_status(date, altitude, latitude, longitude, satellite.q_body_to_eci, satellite.omega)

    print(f"Simulation finishing at")
    print_status(date, altitude, latitude, longitude, satellite.q_body_to_eci, satellite.omega)

    T = np.linspace(0, total_time, num_steps - 1)

    commanded_torques = np.array(commanded_torques)
    applied_torques = np.array(applied_torques)
    error_q_vectors = np.array(error_q_vectors)
    omega_vectors = np.array(omega_vectors)
    target_q_vectors = np.array(target_q_vectors)
    attitude_q_vectors = np.array(attitude_q_vectors)
    b_field_x = np.array(b_field_x)
    b_field_y = np.array(b_field_y)
    b_field_z = np.array(b_field_z)

    pid_proportional_terms_x = np.array(pid_proportional_terms_x)
    pid_proportional_terms_y = np.array(pid_proportional_terms_y)
    pid_proportional_terms_z = np.array(pid_proportional_terms_z)
    pid_integral_terms_x = np.array(pid_integral_terms_x)
    pid_integral_terms_y = np.array(pid_integral_terms_y)
    pid_integral_terms_z = np.array(pid_integral_terms_z)
    pid_derivative_terms_x = np.array(pid_derivative_terms_x)
    pid_derivative_terms_y = np.array(pid_derivative_terms_y)
    pid_derivative_terms_z = np.array(pid_derivative_terms_z)

    # Plot 1: Angular Speed Progression
    fig1 = plt.figure(figsize=(10, 6))
    fig1.canvas.manager.set_window_title("Angular Speed Progression")
    idx = 0
    for key, array in angular_speeds.items():
        n = len(array)
        plt.plot(T[idx:idx + n], array, label=key)
        idx += n
    plt.ylabel("Angular Speed [rad/s]")
    plt.xlabel("Time [s]") # Changed to seconds
    plt.legend()
    plt.title("Angular Speed Progression")
    plt.ylim(bottom=0)
    plt.grid(True)

    # Plot 2: Quaternion Vector Elements Progression
    fig2 = plt.figure(figsize=(10, 6))
    fig2.canvas.manager.set_window_title("Attitude Error Vector Elements Progression")
    plt.plot(T, error_q_vectors[:, 0], label="q1", linestyle='-')
    plt.plot(T, error_q_vectors[:, 1], label="q2", linestyle='--')
    plt.plot(T, error_q_vectors[:, 2], label="q3", linestyle=':')
    plt.ylabel("Vector Element Value")
    plt.xlabel("Time [s]") # Changed to seconds
    plt.legend()
    plt.title("Attitude Error Vector Elements Progression")
    plt.grid(True)

    # Plot 3: Target Attitude Quaternion Vector Elements Progression
    fig3 = plt.figure(figsize=(10, 6))
    if fig3.canvas.manager:
        fig3.canvas.manager.set_window_title("Target Attitude Quaternion Vector Elements Progression")
    plt.plot(T, target_q_vectors[:, 0], label="q1", linestyle='-')
    plt.plot(T, target_q_vectors[:, 1], label="q2", linestyle='--')
    plt.plot(T, target_q_vectors[:, 2], label="q3", linestyle=':')
    plt.ylabel("Vector Element Value")
    plt.xlabel("Time [s]") # Changed to seconds
    plt.legend()
    plt.title("Target Attitude Quaternion Vector Elements Progression")
    plt.grid(True)

    # Plot 4: Actual Attitude Quaternion Vector Elements Progression
    fig4 = plt.figure(figsize=(10, 6))
    if fig4.canvas.manager:
        fig4.canvas.manager.set_window_title("Actual Attitude Quaternion Vector Elements Progression")
    plt.plot(T, attitude_q_vectors[:, 0], label="q1", linestyle='-')
    plt.plot(T, attitude_q_vectors[:, 1], label="q2", linestyle='--')
    plt.plot(T, attitude_q_vectors[:, 2], label="q3", linestyle=':')
    plt.ylabel("Vector Element Value")
    plt.xlabel("Time [s]") # Changed to seconds
    plt.legend()
    plt.title("Actual Attitude Quaternion Vector Elements Progression")
    plt.grid(True)

    # Plot 5: Angular Velocity Components Progression
    fig5 = plt.figure(figsize=(10, 6))
    if fig5.canvas.manager:
        fig5.canvas.manager.set_window_title("Angular Velocity Components Progression")
    plt.plot(T, omega_vectors[:, 0], label="omega_x", linestyle='-')
    plt.plot(T, omega_vectors[:, 1], label="omega_y", linestyle='--')
    plt.plot(T, omega_vectors[:, 2], label="omega_z", linestyle=':')
    plt.ylabel("Angular Velocity [rad/s]")
    plt.xlabel("Time [s]") # Changed to seconds
    plt.legend()
    plt.title("Angular Velocity Components Progression")
    plt.grid(True)

    # Plot 6: Applied Control Torque Components Progression
    fig6 = plt.figure(figsize=(10, 6))
    if fig6.canvas.manager:
        fig6.canvas.manager.set_window_title("Control Torque Components Progression")
    plt.plot(T, applied_torques[:, 0], label="Applied Torque X", linestyle='-')
    plt.plot(T, applied_torques[:, 1], label="Applied Torque Y", linestyle='--')
    plt.plot(T, applied_torques[:, 2], label="Applied Torque Z", linestyle=':')
    plt.ylabel("Control Torque [Nm]")
    plt.xlabel("Time [s]") # Changed to seconds
    plt.legend()
    plt.title("Applied Control Torque Components Progression")
    plt.ylim(bottom=0)
    plt.grid(True)

    # Plot 7: B-field Components Progression
    fig7 = plt.figure(figsize=(10, 6))
    if fig7.canvas.manager:
        fig7.canvas.manager.set_window_title("B-field Components Progression")
    plt.plot(T, b_field_x, label="B_x", linestyle='-')
    plt.plot(T, b_field_y, label="B_y", linestyle='--')
    plt.plot(T, b_field_z, label="B_z", linestyle=':')
    plt.ylabel("B-field [Gauss]")
    plt.xlabel("Time [s]") # Changed to seconds
    plt.legend()
    plt.title("B-field Components Progression")
    plt.grid(True)

    # Plot 8: PID Proportional Terms Progression
    fig8 = plt.figure(figsize=(10, 6))
    if fig8.canvas.manager:
        fig8.canvas.manager.set_window_title("PID Proportional Terms Progression")
    plt.plot(T, pid_proportional_terms_x, label="Proportional X", linestyle='-')
    plt.plot(T, pid_proportional_terms_y, label="Proportional Y", linestyle='--')
    plt.plot(T, pid_proportional_terms_z, label="Proportional Z", linestyle=':')
    plt.ylabel("Proportional Term Value")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.title("PID Proportional Terms Progression")
    plt.grid(True)

    # Plot 9: PID Integral Terms Progression
    fig9 = plt.figure(figsize=(10, 6))
    if fig9.canvas.manager:
        fig9.canvas.manager.set_window_title("PID Integral Terms Progression")
    plt.plot(T, pid_integral_terms_x, label="Integral X", linestyle='-')
    plt.plot(T, pid_integral_terms_y, label="Integral Y", linestyle='--')
    plt.plot(T, pid_integral_terms_z, label="Integral Z", linestyle=':')
    plt.ylabel("Integral Term Value")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.title("PID Integral Terms Progression")
    plt.grid(True)

    # Plot 10: PID Derivative Terms Progression
    fig10 = plt.figure(figsize=(10, 6))
    if fig10.canvas.manager:
        fig10.canvas.manager.set_window_title("PID Derivative Terms Progression")
    plt.plot(T, pid_derivative_terms_x, label="Derivative X", linestyle='-')
    plt.plot(T, pid_derivative_terms_y, label="Derivative Y", linestyle='--')
    plt.plot(T, pid_derivative_terms_z, label="Derivative Z", linestyle=':')
    plt.ylabel("Derivative Term Value")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.title("PID Derivative Terms Progression")
    plt.grid(True)


    plt.show()


if __name__ == "__main__":
    main()
