import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from satellite import Satellite
from controllers import *
from igrf_wrapper import *
from frames import *
from quaternions import get_random_unit_quaternion
from orbit import Orbit
from satellite import Satellite, calculate_alpha, calculate_q_dot

SIMULATION_TIME = 10  # hours
SIMULATION_STEP = 10  # seconds
OMEGA_MAX = 0.01  # rad/s
INERTIA_DIAG_MAX = 1.5  # kg * m^2
PRINT = True


# for numerical solvers
def get_derivatives(t, x, satellite: "Satellite"):
    torque = satellite.controller.get_control(satellite, dt=t)
    omega = x[0:3]
    attitude = x[3:7]

    return np.concat((calculate_alpha(torque, omega, satellite.inertia_tensor), 
                    calculate_q_dot(omega, attitude)))


def main():
    date = datetime.now()

    bdot_gain = 0

    latitude = 20
    longitude = 60
    altitude = 400.0
    inclination = 51.6
    orbit = Orbit(altitude, inclination, latitude, longitude)
    omega = np.random.uniform(-OMEGA_MAX, OMEGA_MAX, 3)
    attitude = get_random_unit_quaternion()
    inertia_tensor = np.zeros(shape=(3, 3))
    np.fill_diagonal(inertia_tensor, np.random.uniform(0, INERTIA_DIAG_MAX, 3))

    B_field = ned_to_body(get_b_field_NED(latitude, longitude, altitude, date), attitude)
    bdot_controller = BDot(bdot_gain, B_field)
    satellite = Satellite(attitude, omega, bdot_controller, inertia_tensor)

    total_time = SIMULATION_TIME * 3600 # total simulation time
    dt = SIMULATION_STEP
    num_steps = int(total_time / dt)

    angular_speeds = []
    torques = []
    
    print(f"Simulation starting at:\
            \ndate: {date}\
            \nlatitude: {latitude} deg\
            \nlongitude: {longitude} deg\
            \naltitude: {altitude} km\
            \nomega: {satellite.omega} (x, y, z) rad/s\
            \nattitude: {np.rad2deg(attitude.to_euler())} (roll, pitch, yaw) deg")

    # simulation loop
    for step in range(num_steps):
        date += timedelta(seconds=dt) 
        
        latitude, longitude = orbit.propagate(dt * num_steps)
        
        # update satellite B field
        satellite.B_field_gauss = ned_to_body(get_b_field_NED(latitude, longitude, altitude, date), satellite.attitude_q)

        if np.any(np.isnan(satellite.omega)):
            print(f"\nSIMULATION BLEW UP ON STEP {step}!\n")
            break

        # update satellite state
        torque = satellite.update(dt)
        torques.append(np.linalg.norm(torque))
        angular_speeds.append(np.linalg.norm(satellite.omega))
        
        if PRINT:
            print(f"step {step}\
                    \ndate: {date}\
                    \nlatitude: {latitude} deg\
                    \nlongitude: {longitude} deg\
                    \naltitude: {altitude} km\
                    \nomega: {omega} (x, y, z) rad/s\
                    \nattitude: {np.rad2deg(attitude.to_euler())} (roll, pitch, yaw) deg\
                    \ntorque: {torque}")

    print(f"Simulation finishing at:\
            \ndate: {date}\
            \nlatitude: {latitude} deg\
            \nlongitude: {longitude} deg\
            \naltitude: {altitude} km\
            \nomega: {omega} (x, y, z) rad/s\
            \nattitude: {np.rad2deg(attitude.to_euler())} (roll, pitch, yaw) deg")
    
    # the percentage should be close to 200% for good B-dot performance
    print(f"final omega is {angular_speeds[-1]}, which is {angular_speeds[-1] / orbit.omega * 100:.1f}% of the orbiting angular velocity")

    plt.plot(angular_speeds, label="omega")
    plt.legend()
    plt.title("angular speed progression")
    plt.show()

if __name__ == "__main__":
    main()
