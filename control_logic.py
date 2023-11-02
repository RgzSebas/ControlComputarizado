# Sebastian Rodriguez - A01700378

import numpy as np

def get_user_coefficients(prompt, num_coeffs):
    coefficients = []
    print(prompt)
    for i in range(num_coeffs):
        coeff = float(input(f"Enter coefficient {i + 1}: "))
        coefficients.append(coeff)
    return np.array(coefficients)

def get_user_amplitude(prompt):
    return float(input(prompt))


# Get user inputs for a and b coefficients
num_a_coeffs = 4
num_b_coeffs = 4
a_coeffs = get_user_coefficients("Enter ARX 'a' coefficients:", num_a_coeffs)
b_coeffs = get_user_coefficients("Enter ARX 'b' coefficients:", num_b_coeffs)

# Get user input for perturbation and input amplitudes
perturbation_amplitude = get_user_amplitude("Enter perturbation amplitude: ")
input_amplitude = get_user_amplitude("Enter input amplitude: ")

# Define the square wave parameters
square_wave_frequency = 1.0  # Frequency of the square wave (Hz)

# Define the sampling interval
sampling_interval = 1.0  # Change this as needed

# Initialize the time and output
t = 0
y = np.zeros(num_a_coeffs)

# Simulate the ARX control function
while True:
    # Simulate the input as a square wave
    u = input_amplitude * (2 * int(t * square_wave_frequency) % 2 - 1)

    # Calculate the perturbation as a square wave
    perturbation = perturbation_amplitude * (2 * int(t * square_wave_frequency) % 2 - 1)

    # Calculate the output at the current time step
    y_next = np.dot(a_coeffs, y) + np.dot(b_coeffs, [u] + y[-(num_b_coeffs - 1):]) + perturbation

    # Print the current output
    print(f"Time: {t:.2f}, Input: {u}, Output: {y_next}")

    # Update the output and time for the next step
    y = np.roll(y, shift=-1)
    y[-1] = y_next
    t += sampling_interval

    # Check if the user wants to stop
    user_input = input("Press Enter to continue or 'q' to quit: ")
    if user_input == 'q':
        break
