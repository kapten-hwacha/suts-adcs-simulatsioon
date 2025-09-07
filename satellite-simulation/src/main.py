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

# SIMULATION SETTINGS
# -------------------

SIMULATION_TIME = 1  # hours
SIMULATION_STEP = 1  # seconds
OMEGA_MAX = 0.1  # rad/s
OMEGA_MAX_DETUMBLED = 0.0002  # rad/s
INERTIA_DIAG_MIN = 0.001
INERTIA_DIAG_MAX = 0.0035  # kg * m^2
PRINT = True  # enables real-time status prints
MIN_ALTITUDE = 300  # km
MAX_ALTITUDE = 600  # km
RANDOM = False  # if orbit parameters are generated randomly
SKIP_DETUMBLE = True  # skips the detumbling part, initial angular velocity already low
UPDATE_TARGET = True

# POINT TO BE TRACKED
# -------------------

LATITUDE_POINT = ...
LONGITUDE_POINT = ...


# CONTROLLER SETTINGS
# -------------------

# BDOT
BDOT_GAIN = 7e-5

# PID
KP = 1e-5
KI = 0
KD = 3e-3
"""
KD is nicely dampened with 1e-3 when target is not updated
"""

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


def main():
    date = datetime.now()

    # generate random parameters for orbit
    if RANDOM:
        eccentricity = np.random.uniform(0.001, 0.05)
        inclination = np.random.uniform(20, 98)
        argument_of_periapsis = np.random.uniform(0, 360)
        raan = np.random.uniform(0, 360)
        initial_true_anomaly = np.random.uniform(0, 360)
        earth_rotation_angle = np.random.uniform(0, 360)
        semi_major_axis = np.random.uniform(MIN_ALTITUDE, MAX_ALTITUDE) + RADIUS_EARTH
    else:
        eccentricity = 0.001
        inclination = 97.4
        argument_of_periapsis = 90
        raan = 0
        initial_true_anomaly = 0
        earth_rotation_angle = 0
        semi_major_axis = (MIN_ALTITUDE + MAX_ALTITUDE) / 2 + RADIUS_EARTH

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

    attitude = get_random_unit_quaternion()
    inertia_tensor = np.zeros(shape=(3, 3))
    np.fill_diagonal(inertia_tensor, np.random.uniform(INERTIA_DIAG_MIN, INERTIA_DIAG_MAX, 3))

    # get initial B-field
    r_eci, v_eci = orbit.propagate(0)
    r_ecef = orbit.vector_eci_to_ecef(0, r_eci)
    altitude, latitude, longitude = get_position_geodetic(r_ecef)
    b_field =  body_to_ned(get_b_field_NED(latitude, longitude, altitude, date), attitude)

    bdot_controller = BDot(BDOT_GAIN, b_field)
    pid_controller = PID(kp=KP, ki=KI, kd=KD)
    lqr_controller = LQR(R=R, Q=Q, J=inertia_tensor)

    # map controllers to states
    controllers: ControllerMap = {
        "DETUMBLE": bdot_controller,
        "COARSE_POINT_NADIR": pid_controller,
        "FINE_POINT_NADIR": lqr_controller
    }

    satellite = Satellite(attitude, omega, inertia_tensor, controllers, b_field, state)

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
    error_q_vectors = []
    omega_vectors = []
    b_field_x = []
    b_field_y = []
    b_field_z = []
    
    print(f"Simulation starting at")
    print_status(date, altitude, latitude, longitude, attitude, omega)

    # simulation loop
    for step in range(1, num_steps):
        date += timedelta(seconds=dt)
        t = dt * step
        r_eci, v_eci = orbit.propagate(t)

        r_ecef = orbit.vector_eci_to_ecef(t, r_eci)
        altitude, latitude, longitude = get_position_geodetic(r_ecef)
        
        # update satellite B field
        satellite.B_field_gauss = body_to_ned(get_b_field_NED(latitude, longitude, altitude, date), satellite.q_body_to_eci)

        q_body_to_eci_target = Quaternion(1, 0, 0, 0)
        
        x_lvlh, y_lvlh, z_lvlh = eci_to_lvlh(r_eci, v_eci)
        R_eci_to_lvlh = get_rotation_matrix(x_lvlh, y_lvlh, z_lvlh)
        q_eci_to_lvlh = rotation_matrix_to_quaternion(R_eci_to_lvlh)
        omega_lvlh_in_eci = orbit.angular_rate * y_lvlh
        omega_lvlh_in_body = satellite.q_body_to_eci.to_rotation_matrix().T @ omega_lvlh_in_eci
        # q_body_to_lvlh = satellite.q_body_to_eci * q_eci_to_lvlh

        if UPDATE_TARGET and (satellite.state == "FINE_POINT_NADIR" or satellite.state == "COARSE_POINT_NADIR"):
            q_body_to_lvlh_target = Quaternion(1, 0, 0, 0)  # unit quaternion for nadir pointing
            q_body_to_eci_target = q_eci_to_lvlh.get_conjugate() * q_body_to_lvlh_target
            q_body_to_eci_target.normalize()

        if np.any(np.isnan(satellite.omega)):
            print(f"\nSIMULATION BLEW UP ON STEP {step}!\n")
            break

        commanded_torque, applied_torque = satellite.update(dt, orbit.angular_rate, q_body_to_eci_target)  # this is the main iteration call

        commanded_torques.append(commanded_torque)
        applied_torques.append(applied_torque)
        angular_speeds[satellite.state].append(np.linalg.norm(satellite.omega))
        error_q_vectors.append(satellite.q_body_to_eci_error.vector)
        omega_vectors.append(satellite.omega)

        b_field_x.append(b_field[0])
        b_field_y.append(b_field[1])
        b_field_z.append(b_field[2])

        if PRINT:
            print(f"step {step}")
            print_status(date, altitude, latitude, longitude, satellite.q_body_to_eci, satellite.omega)

    print(f"Simulation finishing at")
    print_status(date, altitude, latitude, longitude, satellite.q_body_to_eci, satellite.omega)

    T = np.linspace(0, SIMULATION_TIME, num_steps - 1)

    commanded_torques = np.array(commanded_torques)
    applied_torques = np.array(applied_torques)
    error_q_vectors = np.array(error_q_vectors)
    omega_vectors = np.array(omega_vectors)
    b_field_x = np.array(b_field_x)
    b_field_y = np.array(b_field_y)
    b_field_z = np.array(b_field_z)

    # Plot 1: Angular Speed Progression
    fig1 = plt.figure(figsize=(10, 6))
    fig1.canvas.manager.set_window_title("Angular Speed Progression")
    idx = 0
    for key, array in angular_speeds.items():
        n = len(array)
        plt.plot(T[idx:idx + n], array, label=key)
        idx += n
    plt.ylabel("Angular Speed [rad/s]")
    plt.xlabel("Time [h]")
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
    plt.xlabel("Time [h]")
    plt.legend()
    plt.title("Attitude Error Vector Elements Progression")
    plt.grid(True)

    # Plot 3: Angular Velocity Components Progression
    fig3 = plt.figure(figsize=(10, 6))
    fig3.canvas.manager.set_window_title("Angular Velocity Components Progression")
    plt.plot(T, omega_vectors[:, 0], label="omega_x", linestyle='-')
    plt.plot(T, omega_vectors[:, 1], label="omega_y", linestyle='--')
    plt.plot(T, omega_vectors[:, 2], label="omega_z", linestyle=':')
    plt.ylabel("Angular Velocity [rad/s]")
    plt.xlabel("Time [h]")
    plt.legend()
    plt.title("Angular Velocity Components Progression")
    plt.grid(True)

    # Plot 4: Applied Control Torque Components Progression
    fig4 = plt.figure(figsize=(10, 6))
    fig4.canvas.manager.set_window_title("Control Torque Components Progression")
    plt.plot(T, applied_torques[:, 0], label="Applied Torque X", linestyle='-')
    plt.plot(T, applied_torques[:, 1], label="Applied Torque Y", linestyle='--')
    plt.plot(T, applied_torques[:, 2], label="Applied Torque Z", linestyle=':')
    plt.ylabel("Control Torque [Nm]")
    plt.xlabel("Time [h]")
    plt.legend()
    plt.title("Applied Control Torque Components Progression")
    plt.ylim(bottom=0)
    plt.grid(True)

    # Plot 5: B-field Components Progression
    fig5 = plt.figure(figsize=(10, 6))
    fig5.canvas.manager.set_window_title("B-field Components Progression")
    plt.plot(T, b_field_x, label="B_x", linestyle='-')
    plt.plot(T, b_field_y, label="B_y", linestyle='--')
    plt.plot(T, b_field_z, label="B_z", linestyle=':')
    plt.ylabel("B-field [Gauss]")
    plt.xlabel("Time [h]")
    plt.legend()
    plt.title("B-field Components Progression")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
