# Satellite Detumbling and Attitude Control System Simulation

This project simulates a simple satellite detumbling and attitude control system using reduced quaternion theory. The simulation aims to provide insights into the dynamics of satellite attitude control and the effectiveness of various control strategies.

## Project Structure

```
satellite-simulation
├── src
│   ├── __init__.py
│   ├── main.py
│   ├── satellite.py
│   ├── controllers.py
│   ├── quaternions.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Installation

To set up the project, ensure you have Python installed on your system. Then, you can install the required dependencies using pip. 

1. Clone the repository:
   ```
   git clone <repository-url>
   cd satellite-simulation
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the simulation, execute the `main.py` file. This file initializes the satellite system, sets up the simulation parameters, and runs the main simulation loop.

```
python src/main.py
```

## Components

- **src/satellite.py**: Defines the `Satellite` class, which includes properties like `quaternion` and `angular_velocity`, and methods for state updates and control input applications.
  
- **src/controllers.py**: Contains the `Controller` class with methods to compute control inputs based on the satellite's current state.

- **src/quaternions.py**: Provides functions and a class for quaternion operations essential for attitude representation and calculations.

- **src/utils.py**: Includes utility functions for logging, data visualization, and numerical methods to assist in the simulation.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.