from datetime import datetime, timezone
# import numpy as np
# from time_function import time_step_gen, vector_julian_to_julian

def save_quaternion_aem(times_seconds, quaternions, start_mjd, dt, filename="suts_aem_quaternion.txt"):
    
    """
    Save quaternions in AEM format, using time_utils.time_step_gen
    and time_utils.julian_to_vector_julian to avoid duplicating vector ↔ float logic.

    Each data line will be:
      <MJD_day> <MJD_second> <x> <y> <z> <w>
    """
    # Compute total duration in days
    # duration_days = times_seconds[-1]/(24 * 3600)
    # end_mjd = start_mjd + duration_days

    # # Build “vector” inputs for time_step_gen: [ integer_day, seconds_in_day ]
    # start_vec = np.array([int(start_mjd), (start_mjd % 1) * 86400.0])
    # end_vec = np.array([int(end_mjd), (end_mjd % 1) * 86400.0])

    # julian_vec_matrix = time_step_gen(time_interval=dt, start_mjd=start_vec, end_mjd=end_vec, time_type="mjd", input_type="vector", output_type="vector")

    # # Convert that N×2 array into an (N,) float array of MJD
    # julian_floats = vector_julian_to_julian(julian_vec_matrix)

    with open(filename, 'w') as file:
        file.write("CIC_AEM_VERS = 1.0\n")
        file.write(f"CREATION_DATE = {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')}\n")
        file.write("ORIGINATOR = SUTS\n\n")
        file.write("META_START\n\n")
        file.write("COMMENT Detumbling simulation output using quaternion\n\n")
        file.write("OBJECT_NAME = SUTS\n")
        file.write("OBJECT_ID = SUTS\n\n")
        file.write("REF_FRAME_A = EME2000\n")
        file.write("REF_FRAME_B = SC_BODY_1\n")
        file.write("ATTITUDE_DIR = A2B\n")
        file.write("TIME_SYSTEM = UTC\n")
        file.write("ATTITUDE_TYPE = QUATERNION\n\n")
        file.write("META_STOP\n\n")

        # Each line: day second x y z w
        for t, (w, x, y, z) in zip(times_seconds, quaternions):
            mjd_full = start_mjd + (t / 86400.0)
            day = int(mjd_full)
            sec = (mjd_full - day) * 86400.0

            file.write(f"{day} {sec:10.5f} {w:.6f} {x:.6f} {y:.6f} {z:.6f}\n")

    print(f"Quaternion AEM file saved as '{filename}'")


def save_oem_file(times_seconds, positions, start_mjd, dt, filename="suts_oem_position.txt"):
    """
    Save satellite positions in OEM format, reusing time_utils.julian_to_vector_julian
    so there is no duplication of day/second splitting.

    Each data line will be:
      <MJD_day> <MJD_second> <x> <y> <z>
    """
    # Compute total duration in days
    # duration_days = times_seconds[-1]/(24 * 3600)
    # end_mjd = start_mjd + duration_days

    # Convert start_mjd and end_mjd floats into [day, second] vectors
    # start_vec = np.array([int(start_mjd), (start_mjd % 1) * 86400.0])
    # end_vec = np.array([int(end_mjd), (end_mjd % 1) * 86400.0])

    # julian_vec_matrix = time_step_gen(time_interval=dt, start_mjd=start_vec, end_mjd=end_vec, time_type="mjd", input_type="vector", output_type="vector")

    # julian_floats = vector_julian_to_julian(julian_vec_matrix)

    with open(filename, "w") as file:
        file.write("CIC_OEM_VERS = 2.0\n")
        file.write(f"CREATION_DATE  = {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')}\n")
        file.write("ORIGINATOR     = SUTS\n\n")
        file.write("META_START\n\n")
        file.write("OBJECT_NAME = SUTS\n")
        file.write("OBJECT_ID = SUTS\n\n")
        file.write("CENTER_NAME = EARTH\n")
        file.write("REF_FRAME   = EME2000\n")
        file.write("TIME_SYSTEM = UTC\n\n")
        file.write("META_STOP\n\n")

        # Each line: day second x y z
        for t, (x, y, z) in zip(times_seconds, positions):
            mjd_full = start_mjd + (t / 86400.0)
            day = int(mjd_full)
            sec = (mjd_full - day) * 86400.0
            file.write(f"{day} {sec:10.5f} {x:.6f} {y:.6f} {z:.6f}\n")

    print(f"OEM file saved as '{filename}'")