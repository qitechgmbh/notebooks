import numpy as np
import matplotlib.pyplot as plt


class TemperatureSimulation:
    def __init__(
        self,
        control_function,
        # Element A parameters (measured element)
        element_a_mass=1.0,
        element_a_specific_heat_capacity=490,
        # Element B parameters (heating element)
        element_b_mass=0.4,
        element_b_specific_heat_capacity=400,
        # Shared parameters
        target_temp=150,
        ambient_temperature=25,
        # Heat transfer coefficients
        element_a_b_contact_area=0.5,
        element_a_b_heat_transfer_coefficient=100,
        element_b_air_contact_area=0.5,
        element_b_air_heat_transfer_coefficient=10,
        # Simulation parameters
        total_time=3600,
        dt=1,
        max_power=1000,
        max_acceleration=10,
        delay_steps=5,
    ):
        # Store the control function
        self.control_function = control_function

        # Element A properties (the measured element)
        self.element_a_mass = element_a_mass
        self.element_a_specific_heat_capacity = element_a_specific_heat_capacity

        # Element B properties (the heating element)
        self.element_b_mass = element_b_mass
        self.element_b_specific_heat_capacity = element_b_specific_heat_capacity

        # Heat transfer properties between A and B
        self.element_a_b_contact_area = element_a_b_contact_area
        self.element_a_b_heat_transfer_coefficient = (
            element_a_b_heat_transfer_coefficient
        )

        # Heat transfer properties between B and air
        self.element_b_air_contact_area = element_b_air_contact_area
        self.element_b_air_heat_transfer_coefficient = (
            element_b_air_heat_transfer_coefficient
        )

        # Environment parameters
        self.ambient_temperature = ambient_temperature

        # Simulation parameters
        self.target_temp = target_temp
        self.total_time = total_time
        self.dt = dt
        self.time_points = np.arange(0, self.total_time, self.dt)
        self.num_steps = len(self.time_points)

        # Power constraints
        self.max_power = max_power
        self.max_acceleration = max_acceleration

        # Time delay simulation
        self.delay_steps = delay_steps

        # Initialize arrays
        self.target_temp_values = np.zeros(self.num_steps)
        self.element_a_temperature = np.zeros(self.num_steps)
        self.element_b_temperature = np.zeros(self.num_steps)
        # Start at room temperature
        self.element_a_temperature[0] = self.ambient_temperature
        self.element_b_temperature[0] = self.ambient_temperature
        self.target_power_values = np.zeros(self.num_steps)
        self.actual_power_values = np.zeros(self.num_steps)
        self.power_history = np.zeros(self.delay_steps)

        # Define target temperature function (constant target)
        self.target_temp_function = lambda t: self.target_temp

    def update_temperatures(self, element_a_temp, element_b_temp, applied_power):
        """
        Update temperatures of both elements based on heat transfer and applied power.
        """
        # Heat added to element B during this time step
        heat_added_to_b = applied_power * self.dt  # Joules

        # Heat transferred between element A and B
        temp_diff_a_b = element_b_temp - element_a_temp
        heat_transferred_a_b = (
            self.element_a_b_heat_transfer_coefficient
            * self.element_a_b_contact_area
            * temp_diff_a_b
            * self.dt
        )

        # Heat lost from element B to environment
        heat_lost_b_to_air = (
            self.element_b_air_heat_transfer_coefficient
            * self.element_b_air_contact_area
            * (element_b_temp - self.ambient_temperature)
            * self.dt
        )

        # Add some random noise to simulate real-world variations
        noise_a = np.random.normal(0, 0.01)
        noise_b = np.random.normal(0, 0.01)

        # Calculate temperature changes
        # Element B gains heat from power input, loses heat to element A and air
        net_heat_b = heat_added_to_b - heat_transferred_a_b - heat_lost_b_to_air
        element_b_temp_change = (
            net_heat_b / (self.element_b_mass * self.element_b_specific_heat_capacity)
            + noise_b
        )

        # Element A gains heat from element B
        net_heat_a = heat_transferred_a_b
        element_a_temp_change = (
            net_heat_a / (self.element_a_mass * self.element_a_specific_heat_capacity)
            + noise_a
        )

        # Return new temperatures
        new_element_a_temp = element_a_temp + element_a_temp_change
        new_element_b_temp = element_b_temp + element_b_temp_change

        return new_element_a_temp, new_element_b_temp

    def run_simulation(self):
        """Run the complete temperature simulation"""
        for i in range(1, self.num_steps):
            time = self.time_points[i]

            # Calculate target temperature using the provided function
            self.target_temp_values[i] = self.target_temp_function(time)

            # Get control signal using the controller function - using element A temperature as measurement
            # Pass measured temp, target temp, ambient temp, and dt to the controller
            control_signal = self.control_function(
                self.element_a_temperature[i - 1],
                self.target_temp_values[i],
                self.ambient_temperature,
                self.dt,
            )

            # Ensure control signal is between 0 and 1
            control_signal = max(0, min(1, control_signal))

            # Convert control signal to power
            self.target_power_values[i] = self.max_power * control_signal

            # Apply acceleration limit
            power_diff = self.target_power_values[i] - self.actual_power_values[i - 1]
            if power_diff > self.max_acceleration:
                power_diff = self.max_acceleration
            elif power_diff < -self.max_acceleration:
                power_diff = -self.max_acceleration

            self.actual_power_values[i] = self.actual_power_values[i - 1] + power_diff

            # Simulate delay
            self.power_history = np.roll(self.power_history, 1)
            self.power_history[0] = self.actual_power_values[i]
            applied_power = self.power_history[-1] if i >= self.delay_steps else 0

            # Update temperatures using the provided function
            self.element_a_temperature[i], self.element_b_temperature[i] = (
                self.update_temperatures(
                    self.element_a_temperature[i - 1],
                    self.element_b_temperature[i - 1],
                    applied_power,
                )
            )

    def plot_results(self):
        """Plot the simulation results"""
        plt.figure(figsize=(12, 12))

        # Temperature plot showing both elements
        plt.subplot(3, 1, 1)
        plt.plot(
            self.time_points,
            self.element_a_temperature,
            "r-",
            label="Core Temperature",
        )
        plt.plot(
            self.time_points,
            self.element_b_temperature,
            "m-",
            label="Heating Temperature",
        )
        plt.plot(
            self.time_points, self.target_temp_values, "b--", label="Target Temperature"
        )
        plt.title("Temperature of Elements Over Time (With Control)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.grid(True)

        # Power input plot
        plt.subplot(3, 1, 2)
        plt.plot(
            self.time_points, self.actual_power_values, "b--", label="Actual Power"
        )
        plt.plot(self.time_points, self.target_power_values, "g-", label="Target Power")
        plt.axhline(y=self.max_power, color="r", linestyle="-.", label="Max Power")
        plt.title("Power Input Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Power (Watts)")
        plt.legend()
        plt.grid(True)

        # Error plot
        plt.subplot(3, 1, 3)
        error_values = self.target_temp_values - self.element_a_temperature
        plt.plot(self.time_points, error_values, "m-")
        plt.title("Temperature Error Over Time (Target - Element A)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Error (°C)")
        plt.ylim(-2.5, 1.5)
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def get_statistics(self):
        # diff_error is the difference between target and actual element A temperature in °C
        diff_errors = self.target_temp_values - self.element_a_temperature
        # Skip first 600 data points to ignore initial heating phase
        start_idx = 1800 if len(diff_errors) > 600 else 0
        diff_error_mean = np.mean(diff_errors[start_idx:])
        diff_error_std = np.std(diff_errors[start_idx:])

        # overshoot_error is the maximum negative error (temperature exceeding target)
        overshoot_error = -(np.min(diff_errors) if np.min(diff_errors) < 0 else 0)

        # Find the first time the error was within ±0.5°C for at least 600 consecutive seconds
        time_within_0_5 = None
        consecutive_count = 0
        required_consecutive = (
            600 // self.dt
        )  # Convert 600 seconds to number of time steps

        for i, error in enumerate(diff_errors):
            if abs(error) <= 0.5:
                consecutive_count += 1
                if (
                    consecutive_count >= required_consecutive
                    and time_within_0_5 is None
                ):
                    # Found the first occurrence - record the time when this started
                    time_within_0_5 = self.time_points[i - required_consecutive + 1]
                    break
            else:
                consecutive_count = 0

        # If we never reached the required consecutive time within bounds
        if time_within_0_5 is None:
            time_within_0_5 = float("inf")  # Or some other indicator like -1

        # Calculate maximum temperature difference between Element A and B
        max_temp_diff = np.max(
            np.abs(self.element_b_temperature - self.element_a_temperature)
        )

        return {
            "mean_error": diff_error_mean,
            "std_error": diff_error_std,
            "overshoot_error": overshoot_error,
            "time_within_0_5": time_within_0_5,
            "max_temp_diff_a_b": max_temp_diff,
        }

    def print_statistics(self):
        """Print the statistics of the simulation"""
        stats = self.get_statistics()
        print("Simulation Statistics:")
        print(f"- Mean Error (after 1800s): {stats['mean_error']:.2f} °C")
        print(f"- Error STD (after 1800s): {stats['std_error']:.2f} °C")
        print(f"- Overshoot Error: {stats['overshoot_error']:.2f} °C")
        print(f"- Final Element A Temperature: {self.element_a_temperature[-1]:.2f} °C")
        print(f"- Final Element B Temperature: {self.element_b_temperature[-1]:.2f} °C")
        print(
            f"- Maximum A-B Temperature Difference: {stats['max_temp_diff_a_b']:.2f} °C"
        )
        print(f"- Time to reach ±0.5°C: {stats['time_within_0_5']:.2f} seconds")

    def print_parameters(self):
        """Print the parameters of the simulation"""
        print("Simulation Parameters:")
        print("Element A (measured element):")
        print(f"- Mass: {self.element_a_mass} kg")
        print(
            f"- Specific Heat Capacity: {self.element_a_specific_heat_capacity} J/(kg·°C)"
        )

        print("\nElement B (heating element):")
        print(f"- Mass: {self.element_b_mass} kg")
        print(
            f"- Specific Heat Capacity: {self.element_b_specific_heat_capacity} J/(kg·°C)"
        )

        print("\nHeat Transfer:")
        print(f"- A-B Contact Area: {self.element_a_b_contact_area} m²")
        print(
            f"- A-B Heat Transfer Coefficient: {self.element_a_b_heat_transfer_coefficient} W/(m²·°C)"
        )
        print(f"- B-Air Contact Area: {self.element_b_air_contact_area} m²")
        print(
            f"- B-Air Heat Transfer Coefficient: {self.element_b_air_heat_transfer_coefficient} W/(m²·°C)"
        )

        print("\nEnvironment:")
        print(f"- Ambient Temperature: {self.ambient_temperature} °C")

        print("\nSimulation Settings:")
        print(f"- Total Time: {self.total_time} seconds")
        print(f"- Time Step (dt): {self.dt} seconds")


class SuperTemperatureSimulation:
    def __init__(
        self,
        control_function,
        n_simulations=1000,
        # Element A parameters (measured element)
        element_a_mass_range=(1.0, 2.0),
        element_a_specific_heat_capacity_range=(450, 550),
        # Element B parameters (heating element)
        element_b_mass_range=(0.1, 0.5),
        element_b_specific_heat_capacity_range=(350, 450),
        # Target temperature
        target_temp_range=(100, 200),
        # Environment
        ambient_temperature_range=(20, 30),
        # Heat transfer parameters
        element_a_b_contact_area_range=(0.2, 0.5),
        element_a_b_heat_transfer_coefficient_range=(80, 120),
        element_b_air_contact_area_range=(0.2, 0.5),
        element_b_air_heat_transfer_coefficient_range=(8, 12),
        # Power and control parameters
        max_power_range=(800, 1100),
        max_acceleration_range=(8, 12),
        delay_steps_range=(3, 7),
        total_time=3600,
        dt=1,
    ):
        """
        Initialize the super temperature simulation with randomization ranges for two-element model.

        Parameters:
        control_function: Function that takes measured_temp, target_temp, ambient_temp, dt and returns control signal
        n_simulations: Number of simulations to run
        *_range: Tuples of (min, max) for randomizing each parameter
        total_time: Total simulation time in seconds
        dt: Time step in seconds
        """
        self.control_function = control_function
        self.n_simulations = n_simulations

        # Element A parameters
        self.element_a_mass_range = element_a_mass_range
        self.element_a_specific_heat_capacity_range = (
            element_a_specific_heat_capacity_range
        )

        # Element B parameters
        self.element_b_mass_range = element_b_mass_range
        self.element_b_specific_heat_capacity_range = (
            element_b_specific_heat_capacity_range
        )

        # Target temperature
        self.target_temp_range = target_temp_range

        # Environment
        self.ambient_temperature_range = ambient_temperature_range

        # Heat transfer parameters
        self.element_a_b_contact_area_range = element_a_b_contact_area_range
        self.element_a_b_heat_transfer_coefficient_range = (
            element_a_b_heat_transfer_coefficient_range
        )
        self.element_b_air_contact_area_range = element_b_air_contact_area_range
        self.element_b_air_heat_transfer_coefficient_range = (
            element_b_air_heat_transfer_coefficient_range
        )

        # Power and control parameters
        self.max_power_range = max_power_range
        self.max_acceleration_range = max_acceleration_range
        self.delay_steps_range = delay_steps_range

        self.total_time = total_time
        self.dt = dt

        # Store results from all simulations
        self.simulations = []
        self.stats = []

    def _random_value(self, range_tuple):
        """Generate a random value within the given range"""
        min_val, max_val = range_tuple
        return min_val + (max_val - min_val) * np.random.random()

    def _create_random_simulation(self):
        """Create a simulation with randomized parameters for the two-element model"""
        return TemperatureSimulation(
            control_function=self.control_function,
            # Element A parameters
            element_a_mass=self._random_value(self.element_a_mass_range),
            element_a_specific_heat_capacity=self._random_value(
                self.element_a_specific_heat_capacity_range
            ),
            # Element B parameters
            element_b_mass=self._random_value(self.element_b_mass_range),
            element_b_specific_heat_capacity=self._random_value(
                self.element_b_specific_heat_capacity_range
            ),
            # Target temperature
            target_temp=self._random_value(self.target_temp_range),
            # Environment
            ambient_temperature=self._random_value(self.ambient_temperature_range),
            # Heat transfer parameters
            element_a_b_contact_area=self._random_value(
                self.element_a_b_contact_area_range
            ),
            element_a_b_heat_transfer_coefficient=self._random_value(
                self.element_a_b_heat_transfer_coefficient_range
            ),
            element_b_air_contact_area=self._random_value(
                self.element_b_air_contact_area_range
            ),
            element_b_air_heat_transfer_coefficient=self._random_value(
                self.element_b_air_heat_transfer_coefficient_range
            ),
            # Power and control parameters
            total_time=self.total_time,
            dt=self.dt,
            max_power=self._random_value(self.max_power_range),
            max_acceleration=self._random_value(self.max_acceleration_range),
            delay_steps=int(self._random_value(self.delay_steps_range)),
        )

    def run_simulations(self):
        """Run multiple simulations with randomized parameters"""
        self.simulations = []
        self.stats = []

        for i in range(self.n_simulations):
            print(f"Running simulation {i + 1}/{self.n_simulations}...")
            sim = self._create_random_simulation()
            sim.run_simulation()
            self.simulations.append(sim)
            self.stats.append(sim.get_statistics())

    def get_aggregate_statistics(self):
        """Calculate aggregate statistics across all simulations"""
        if not self.stats:
            return "No simulations have been run yet."

        # Extract metrics from all simulations
        mean_errors = [stat["mean_error"] for stat in self.stats]
        std_errors = [stat["std_error"] for stat in self.stats]
        overshoot_errors = [stat["overshoot_error"] for stat in self.stats]
        times_within_0_5 = [stat["time_within_0_5"] for stat in self.stats]
        max_temp_diffs = [stat["max_temp_diff_a_b"] for stat in self.stats]

        # Calculate mean and standard deviation for each metric
        aggregate_stats = {
            "mean_error": {
                "mean": np.mean(mean_errors),
                "std": np.std(mean_errors),
            },
            "std_error": {
                "mean": np.mean(std_errors),
                "std": np.std(std_errors),
            },
            "overshoot_error": {
                "mean": np.mean(overshoot_errors),
                "std": np.std(overshoot_errors),
            },
            "max_temp_diff_a_b": {
                "mean": np.mean(max_temp_diffs),
                "std": np.std(max_temp_diffs),
            },
            "time_within_0_5": {
                "mean": np.mean([t for t in times_within_0_5 if t != float("inf")])
                if any(t != float("inf") for t in times_within_0_5)
                else float("inf"),
                "std": np.std([t for t in times_within_0_5 if t != float("inf")])
                if any(t != float("inf") for t in times_within_0_5)
                else 0,
                "success_rate": sum(1 for t in times_within_0_5 if t != float("inf"))
                / len(times_within_0_5),
            },
        }

        return aggregate_stats

    def print_aggregate_statistics(self):
        """Print aggregate statistics from all simulations"""
        stats = self.get_aggregate_statistics()
        if isinstance(stats, str):
            print(stats)
            return

        print("\n===== AGGREGATE SIMULATION STATISTICS =====")
        print(f"Based on {self.n_simulations} randomized simulations")
        print("\nMean Error (after 1800s):")
        print(f"  Mean: {stats['mean_error']['mean']:.2f} °C")
        print(f"  Std Dev: {stats['mean_error']['std']:.2f} °C")

        print("\nError Standard Deviation (after 1800s):")
        print(f"  Mean: {stats['std_error']['mean']:.2f} °C")
        print(f"  Std Dev: {stats['std_error']['std']:.2f} °C")

        print("\nOvershoot Error:")
        print(f"  Mean: {stats['overshoot_error']['mean']:.2f} °C")
        print(f"  Std Dev: {stats['overshoot_error']['std']:.2f} °C")

        print("\nMax Temperature Difference between Elements A and B:")
        print(f"  Mean: {stats['max_temp_diff_a_b']['mean']:.2f} °C")
        print(f"  Std Dev: {stats['max_temp_diff_a_b']['std']:.2f} °C")

        print("\nTime to reach ±0.5°C for 600s:")
        if stats["time_within_0_5"]["mean"] == float("inf"):
            print("  Never achieved in any simulation")
        else:
            print(f"  Mean: {stats['time_within_0_5']['mean']:.2f} seconds")
            print(f"  Std Dev: {stats['time_within_0_5']['std']:.2f} seconds")
            print(
                f"  Success Rate: {stats['time_within_0_5']['success_rate'] * 100:.1f}%"
            )

    def plot_temperature_comparison(self):
        """Plot temperature curves from all simulations using relative values"""
        if not self.simulations:
            print("No simulations have been run yet.")
            return

        plt.figure(figsize=(15, 10))

        # Plot Element A temperatures
        plt.subplot(2, 1, 1)
        # Plot normalized temperature curves for all simulations - Element A
        for i, sim in enumerate(self.simulations):
            # Calculate relative temperature (as percentage of target)
            relative_temp = sim.element_a_temperature / sim.target_temp
            relative_target = np.ones_like(sim.time_points)  # 100% is the target

            plt.plot(
                sim.time_points,
                relative_temp,
                alpha=0.3,
                label=f"Sim {i + 1} (Target: {sim.target_temp:.0f}°C)" if i < 5 else "",
            )

        # Plot the average relative temperature curve for Element A
        avg_relative_temp_a = np.mean(
            [sim.element_a_temperature / sim.target_temp for sim in self.simulations],
            axis=0,
        )
        plt.plot(
            self.simulations[0].time_points,
            avg_relative_temp_a,
            "k-",
            linewidth=2,
            label="Average Element A",
        )

        # Plot the 100% target line
        plt.plot(
            self.simulations[0].time_points,
            relative_target,
            "r--",
            linewidth=2,
            label="Target (100%)",
        )

        plt.title("Relative Temperature Curves (Element A) Across All Simulations")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Temperature (% of target)")
        plt.grid(True)
        plt.legend(loc="best")

        # Plot Element B temperatures
        plt.subplot(2, 1, 2)
        # Plot normalized temperature curves for all simulations - Element B
        for i, sim in enumerate(self.simulations):
            # Calculate relative temperature (as percentage of target)
            relative_temp = sim.element_b_temperature / sim.target_temp
            relative_target = np.ones_like(sim.time_points)  # 100% is the target

            plt.plot(
                sim.time_points,
                relative_temp,
                alpha=0.3,
                label=f"Sim {i + 1} (Target: {sim.target_temp:.0f}°C)" if i < 5 else "",
            )

        # Plot the average relative temperature curve for Element B
        avg_relative_temp_b = np.mean(
            [sim.element_b_temperature / sim.target_temp for sim in self.simulations],
            axis=0,
        )
        plt.plot(
            self.simulations[0].time_points,
            avg_relative_temp_b,
            "k-",
            linewidth=2,
            label="Average Element B",
        )

        # Plot the 100% target line
        plt.plot(
            self.simulations[0].time_points,
            relative_target,
            "r--",
            linewidth=2,
            label="Target (100%)",
        )

        plt.title("Relative Temperature Curves (Element B) Across All Simulations")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Temperature (% of target)")
        plt.grid(True)
        plt.legend(loc="best")

        plt.tight_layout()
        plt.show()

    def plot_error_distribution(self):
        """Plot distribution of key error metrics"""
        if not self.stats:
            print("No simulations have been run yet.")
            return

        mean_errors = [stat["mean_error"] for stat in self.stats]
        std_errors = [stat["std_error"] for stat in self.stats]
        overshoot_errors = [stat["overshoot_error"] for stat in self.stats]
        times_within_0_5 = [stat["time_within_0_5"] for stat in self.stats]

        plt.figure(figsize=(15, 10))

        # Plot mean error distribution
        plt.subplot(2, 2, 1)
        plt.hist(mean_errors, bins=10, alpha=0.7)
        plt.axvline(
            np.mean(mean_errors),
            color="r",
            linestyle="--",
            label=f"Mean: {np.mean(mean_errors):.2f}",
        )
        plt.title("Mean Error Distribution")
        plt.xlabel("Mean Error (°C)")
        plt.ylabel("Frequency")
        plt.legend()

        # Plot std error distribution
        plt.subplot(2, 2, 2)
        plt.hist(std_errors, bins=10, alpha=0.7)
        plt.axvline(
            np.mean(std_errors),
            color="r",
            linestyle="--",
            label=f"Mean: {np.mean(std_errors):.2f}",
        )
        plt.title("Error Std Dev Distribution")
        plt.xlabel("Error Std Dev (°C)")
        plt.ylabel("Frequency")
        plt.legend()

        # Plot overshoot error distribution
        plt.subplot(2, 2, 3)
        plt.hist(overshoot_errors, bins=10, alpha=0.7)
        plt.axvline(
            np.mean(overshoot_errors),
            color="r",
            linestyle="--",
            label=f"Mean: {np.mean(overshoot_errors):.2f}",
        )
        plt.title("Overshoot Error Distribution")
        plt.xlabel("Overshoot Error (°C)")
        plt.ylabel("Frequency")
        plt.legend()

        # Plot time to reach ±0.5°C distribution
        plt.subplot(2, 2, 4)
        # Filter out infinity values
        finite_times = [t for t in times_within_0_5 if t != float("inf")]
        if finite_times:
            # Use only finite values for the histogram
            plt.hist(finite_times, bins=10, alpha=0.7)
            plt.axvline(
                np.mean(finite_times),
                color="r",
                linestyle="--",
                label=f"Mean: {np.mean(finite_times):.2f}s",
            )
            success_rate = len(finite_times) / len(times_within_0_5) * 100
            plt.title(f"Time to reach ±0.5°C (Success rate: {success_rate:.1f}%)")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Frequency")
            plt.legend()
        else:
            plt.text(
                0.5,
                0.5,
                "No simulations reached ±0.5°C",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title("Time to reach ±0.5°C")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    def plot_relative_error_comparison(self):
        """Plot relative error curves from all simulations"""
        if not self.simulations:
            print("No simulations have been run yet.")
            return

        plt.figure(figsize=(12, 8))

        # Plot relative error curves for all simulations (using element A temperature)
        for i, sim in enumerate(self.simulations):
            # Calculate relative error (as percentage of target)
            relative_error = (
                100
                * (sim.element_a_temperature - sim.target_temp_values)
                / sim.target_temp
            )

            plt.plot(
                sim.time_points,
                relative_error,
                alpha=0.3,
                label=f"Sim {i + 1} (Target: {sim.target_temp:.0f}°C)" if i < 5 else "",
            )

        # Plot the average relative error curve
        avg_relative_error = np.mean(
            [
                100
                * (sim.element_a_temperature - sim.target_temp_values)
                / sim.target_temp
                for sim in self.simulations
            ],
            axis=0,
        )
        plt.plot(
            self.simulations[0].time_points,
            avg_relative_error,
            "k-",
            linewidth=2,
            label="Average",
        )

        # Plot the zero error line
        plt.axhline(y=0, color="r", linestyle="--", label="Zero Error")

        plt.title("Relative Error Curves Across All Simulations (Element A)")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Error (% of target)")
        plt.grid(True)
        plt.ylim(-20, 20)  # Limit y-axis for better visibility

        # Only show legend for first few simulations to avoid clutter
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def plot_temperature_difference(self):
        """Plot temperature difference between Element A and Element B"""
        if not self.simulations:
            print("No simulations have been run yet.")
            return

        plt.figure(figsize=(12, 8))

        # Plot temperature difference for all simulations
        for i, sim in enumerate(self.simulations):
            temp_diff = sim.element_b_temperature - sim.element_a_temperature
            plt.plot(
                sim.time_points,
                temp_diff,
                alpha=0.3,
                label=f"Sim {i + 1}" if i < 5 else "",
            )

        # Plot the average temperature difference
        avg_temp_diff = np.mean(
            [
                sim.element_b_temperature - sim.element_a_temperature
                for sim in self.simulations
            ],
            axis=0,
        )
        plt.plot(
            self.simulations[0].time_points,
            avg_temp_diff,
            "k-",
            linewidth=2,
            label="Average Difference",
        )

        plt.title("Temperature Difference Between Elements B and A")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Temperature Difference (°C)")
        plt.grid(True)

        # Only show legend for first few simulations to avoid clutter
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
