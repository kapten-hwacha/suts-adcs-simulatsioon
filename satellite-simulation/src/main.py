import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from satellite import Satellite, ControllerMap
from controllers import *
from igrf_wrapper import *
from frames import *
from quaternions import get_random_unit_quaternion
from orbit import Orbit, RADIUS_EARTH

np.set_printoptions(precision=2, formatter={'float_kind': lambda x: "%.2e" % x})

# SIMULATION SETTINGS
# -------------------

SIMULATION_TIME = 4  # hours
SIMULATION_STEP = 2  # seconds
OMEGA_MAX = 0.1  # rad/
INERTIA_DIAG_MIN = 1
INERTIA_DIAG_MAX = 3.5  # kg * m^2
PRINT = True  # enables real-time status prints
MIN_ALTITUDE = 300  # km
MAX_ALTITUDE = 600  # km

# -------------------


# CONTROLLER SETTINGS
# -------------------

# BDOT
BDOT_GAIN = 0.07

# PID
KP = 1e-8
KI = 0
KD = 5e-11

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


def main():
    date = datetime.now()

    # generate random parameters for orbit
    eccentricity = np.random.uniform(0.001, 0.05)
    inclination = np.random.uniform(20, 98)
    argument_of_periapsis = np.random.uniform(0, 360)
    raan = np.random.uniform(0, 360)
    initial_true_anomaly = np.random.uniform(0, 360)
    earth_rotation_angle = np.random.uniform(0, 360)
    semi_major_axis = np.random.uniform(MIN_ALTITUDE, MAX_ALTITUDE) + RADIUS_EARTH
    
    orbit = Orbit(semi_major_axis,
                    inclination,
                    eccentricity,
                    argument_of_periapsis,
                    raan,
                    initial_true_anomaly,
                    earth_rotation_angle)

    # generate random initial parameters for the satellite
    omega = np.random.uniform(-OMEGA_MAX, OMEGA_MAX, 3)
    attitude = get_random_unit_quaternion()
    inertia_tensor = np.zeros(shape=(3, 3))
    np.fill_diagonal(inertia_tensor, np.random.uniform(INERTIA_DIAG_MIN, INERTIA_DIAG_MAX, 3))
    altitude, latitude, longitude = orbit.propagate(0)

    # calculate initial magnetic field
    b_field = ned_to_body(get_b_field_NED(latitude, longitude, altitude, date), attitude)
    bdot_controller = BDot(BDOT_GAIN, b_field)
    pid_controller = PID(kp=KP, ki=KI, kd=KD)
    lqr_controller = LQR(R=R, Q=Q, J=inertia_tensor * 1e-3)

    # map controllers to states
    controllers: ControllerMap = {
        "DETUMBLE": bdot_controller,
        "POINT": lqr_controller
    }

    satellite = Satellite(attitude, omega, inertia_tensor, controllers, b_field)

    total_time = SIMULATION_TIME * 3600 # total simulation time
    dt = SIMULATION_STEP
    num_steps = int(total_time / dt)

    # lists for plotting
    angular_speeds = []
    torques = []
    quaternion_vectors = [] # New list
    omega_vectors = []      # New list
    detumbled_idx = num_steps
    
    print(f"Simulation starting at")
    print_status(date, altitude, latitude, longitude, attitude, omega)

    # simulation loop
    for step in range(1, num_steps):
        date += timedelta(seconds=dt) 
        
        altitude, latitude, longitude = orbit.propagate(dt * step)
        
        # update satellite B field
        satellite.B_field_gauss = ned_to_body(get_b_field_NED(latitude, longitude, altitude, date), satellite.attitude_q)

        if np.any(np.isnan(satellite.omega)):
            print(f"\nSIMULATION BLEW UP ON STEP {step}!\n")
            break

        torque, detumbled = satellite.update(dt, orbit.angular_rate)  # this is the main iteration call

        if detumbled:
            detumbled_idx = step

        torques.append(np.linalg.norm(torque))
        angular_speeds.append(np.linalg.norm(satellite.omega))
        quaternion_vectors.append(satellite.attitude_q.vector) # New line
        omega_vectors.append(satellite.omega)                  # New line
        
        if PRINT:
            print(f"step {step}")
            print_status(date, altitude, latitude, longitude, satellite.attitude_q, satellite.omega)

    print(f"Simulation finishing at")
    print_status(date, altitude, latitude, longitude, satellite.attitude_q, satellite.omega)
    
    # the percentage should be close to 200% for good B-dot performance
    print(f"final omega is {angular_speeds[-1]:.3e},\
            \nwhich is {angular_speeds[-1] / orbit.angular_rate * 100:.1f}% of the orbiting angular velocity\
            \nfinal attitude is {(attitude).to_euler()} (roll, pitch, yaw) deg")

    T = np.linspace(0, SIMULATION_TIME, num_steps - 1)

    quaternion_vectors = np.array(quaternion_vectors)
    omega_vectors = np.array(omega_vectors)

    fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    fig.suptitle("Satellite Control System Performance", fontsize=16)

    # Plot 1: Angular Speed Progression
    axs[0].plot(T[:detumbled_idx], angular_speeds[:detumbled_idx], label="detumbling", color="blue")
    axs[0].plot(T[detumbled_idx:], angular_speeds[detumbled_idx:], label="pointing", color="orange")
    axs[0].set_ylabel("Angular Speed [rad/s]")
    axs[0].legend()
    axs[0].set_title("Angular Speed Progression")
    axs[0].set_ylim(bottom=0)
    axs[0].grid(True)

    # Plot 2: Quaternion Vector Elements Progression
    axs[1].plot(T, quaternion_vectors[:, 0], label="q1", linestyle='-')
    axs[1].plot(T, quaternion_vectors[:, 1], label="q2", linestyle='--')
    axs[1].plot(T, quaternion_vectors[:, 2], label="q3", linestyle=':')
    axs[1].set_ylabel("Vector Element Value")
    axs[1].legend()
    axs[1].set_title("Attitude Vector Elements Progression")
    axs[1].grid(True)

    # Plot 3: Angular Velocity Components Progression
    axs[2].plot(T, omega_vectors[:, 0], label="omega_x", linestyle='-')
    axs[2].plot(T, omega_vectors[:, 1], label="omega_y", linestyle='--')
    axs[2].plot(T, omega_vectors[:, 2], label="omega_z", linestyle=':')
    axs[2].set_ylabel("Angular Velocity [rad/s]")
    axs[2].set_xlabel("Time [h]")
    axs[2].legend()
    axs[2].set_title("Angular Velocity Components Progression")
    axs[2].grid(True)

    # Plot 4: Control Torque Norm Progression
    axs[3].plot(T, torques, label="Torque", color="green")
    axs[3].set_ylabel("Control Torque [Nm]")
    axs[3].set_xlabel("Time [h]")
    axs[3].legend()
    axs[3].set_title("Control Torque Progression")
    axs[3].set_ylim(bottom=0)
    axs[3].grid(True)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95)) # Adjust layout to prevent title overlap
    plt.show()

if __name__ == "__main__":
    main()
