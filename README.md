#!/usr/bin/env python3
"""
Homework Assignment ELE2038 – PID Tuning for Question 2

Plant:
    G(s) = (0.5s + 1) / ((s + 1)**3 * (0.1s + 1))

This script tunes a PID controller for the plant using two methods:
    1. Ziegler–Nichols First Method (using step response data)
    2. Ziegler–Nichols Ultimate Sensitivity Method

For each method, the closed-loop step response is plotted.
"""

import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# Define Laplace variable and plant transfer function
s = ctrl.TransferFunction.s
plant = (0.5*s + 1) / ((s + 1)**3 * (0.1*s + 1))

# ===============================================================
# Ziegler–Nichols First Method (Step Response Method)
# ===============================================================
# Assume we have measured these values from the open-loop step response:
L = 0.2   # apparent time delay in seconds
T = 1.0   # time constant in seconds

# According to the textbook formulas (Section 10.3.2),
# we set the PID parameters as follows:
Kp_zn1 = 1.2 * (T / L)
Ti_zn1 = 2 * L
Td_zn1 = 0.5 * L

# Form the PID controller using the standard ideal form:
# C(s) = Kp * (1 + 1/(Ti*s) + Td*s)
PID_zn1 = Kp_zn1 * (1 + 1/(Ti_zn1*s) + Td_zn1*s)

# Create the closed-loop system (unity feedback)
CL_zn1 = ctrl.feedback(PID_zn1 * plant, 1)

# Compute the step response for the first method
time1, response1 = ctrl.step_response(CL_zn1)

plt.figure(figsize=(8, 4))
plt.plot(time1, response1, label='ZN First Method')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.title('Step Response (Ziegler–Nichols First Method)')
plt.legend()
plt.grid(True)

# ===============================================================
# Ziegler–Nichols Ultimate Sensitivity Method
# ===============================================================
# Find the phase crossover frequency of the open-loop system
omega_range = np.logspace(-2, 2, 1000)
mag, phase, omega = ctrl.bode(plant, omega_range, Plot=False)

# Convert phase to degrees for easier interpretation
phase_deg = np.degrees(phase)

# Find frequency where phase is closest to -180 degrees
index = np.argmin(np.abs(phase_deg + 180))
w_pc = omega[index]  # phase crossover frequency (rad/s)

# Ultimate gain Ku is the reciprocal of the magnitude at the phase crossover
Ku = 1 / np.abs(ctrl.evalfr(plant, 1j * w_pc))
Tu = 2 * np.pi / w_pc  # ultimate period

# Using the Z-N ultimate sensitivity formulas (Section 10.3.3):
Kp_znu = 0.6 * Ku
Ti_znu = 0.5 * Tu
Td_znu = 0.125 * Tu

# Build the PID controller for the ultimate sensitivity method
PID_znu = Kp_znu * (1 + 1/(Ti_znu*s) + Td_znu*s)

# Closed-loop system with this controller
CL_znu = ctrl.feedback(PID_znu * plant, 1)

# Compute the step response for the ultimate sensitivity method
time2, response2 = ctrl.step_response(CL_znu)

plt.figure(figsize=(8, 4))
plt.plot(time2, response2, 'r', label='ZN Ultimate Method')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.title('Step Response (Ziegler–Nichols Ultimate Sensitivity Method)')
plt.legend()
plt.grid(True)

# ===============================================================
# Print out tuning parameters for reference
# ===============================================================
print("Ziegler-Nichols First Method Parameters:")
print(f"  Kp = {Kp_zn1:.3f}, Ti = {Ti_zn1:.3f} s, Td = {Td_zn1:.3f} s")
print("\nZiegler-Nichols Ultimate Sensitivity Method:")
print(f"  Ultimate Gain Ku = {Ku:.3f}")
print(f"  Ultimate Period Tu = {Tu:.3f} s")
print(f"  Kp = {Kp_znu:.3f}, Ti = {Ti_znu:.3f} s, Td = {Td_znu:.3f} s")

plt.show()
