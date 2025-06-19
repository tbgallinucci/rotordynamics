# Rotordynamic Analysis - Python Translation

This repository contains a Python translation of a MATLAB rotordynamic analysis code. The analysis simulates the vibration behavior of a rotating shaft-disk-bearing system using finite element methods.

## Overview

The rotordynamic analysis includes:
- **Hydrodynamic bearing coefficient calculation** using Reynolds equation
- **Finite element modeling** of the rotor system
- **Frequency response analysis** 
- **Critical speed identification**
- **Mode shape visualization**
- **Bearing locus plots**

## System Description

The analyzed system consists of:
- A steel shaft (13mm diameter, 747mm length)
- A disk (90mm diameter, 47mm length, 2.3kg mass)
- Two hydrodynamic journal bearings (30mm diameter, 20mm length)
- Unbalance mass of 3.7×10⁻⁵ kg·m

## Files Structure

```
├── main_rotordynamic.py      # Main analysis script (complete solution)
├── rotordynamic_analysis.py  # Bearing coefficient calculations
├── rotordynamic_fem.py       # Finite element model and response
├── rotordynamic_plots.py     # Plotting functions
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Installation

1. **Install Python 3.7 or higher**

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

   Or install individually:
   ```bash
   pip install numpy scipy matplotlib
   ```

## Usage

### Quick Start (Recommended)

Run the complete analysis using the main script:

```python
python main_rotordynamic.py
```

This will:
1. Calculate bearing coefficients for both bearings
2. Generate all plots automatically
3. Display analysis summary

### Advanced Usage

For more control, you can run individual modules:

```python
# Import the main class
from main_rotordynamic import RotordynamicAnalysis

# Create analysis instance
analysis = RotordynamicAnalysis()

# Run bearing coefficient calculations
analysis.run_analysis()

# Generate plots
analysis.plot_results()
```

### Modular Approach

You can also run individual components:

```python
# Run bearing analysis only
python rotordynamic_analysis.py

# Run FEM analysis only
python rotordynamic_fem.py

# Generate plots only
python rotordynamic_plots.py
```

## Results and Plots

The analysis generates several types of plots:

1. **Bearing Stiffness Coefficients** (K_ij vs. rotational speed)
2. **Bearing Damping Coefficients** (C_ij vs. rotational speed)  
3. **Bearing Locus** (journal center trajectory within clearance)
4. **Frequency Response Functions** (amplitude vs. speed at key locations)
5. **Mode Shapes** (deflection patterns at critical speeds)
6. **Campbell Diagram** (natural frequencies vs. rotor speed)

## Key Parameters

You can modify system parameters in the `RotordynamicAnalysis` class:

### Material Properties
- `E`: Young's modulus (207 GPa)
- `rho`: Material density (7850 kg/m³)

### Geometry
- `de`: Shaft diameter (13 mm)
- `le`: Shaft length (747 mm)
- `md`: Disk mass (2.3 kg)
- `dd`: Disk diameter (90 mm)

### Bearing Properties
- `D`: Bearing diameter (30 mm)
- `C`: Bearing length (20 mm)
- `delta`: Radial clearance (90 μm)
- `mi`: Oil viscosity (0.051 Pa·s)

### Analysis Settings
- `omega`: Speed range (10-2000 rad/s, steps of 10)

## Technical Details

### Bearing Analysis
The hydrodynamic bearing coefficients are calculated using:
- Reynolds equation for thin film lubrication
- Newton-Raphson iteration for equilibrium position
- Linearized stiffness and damping coefficients

### Finite Element Model
- Timoshenko beam elements with 4 DOF per node
- Consistent mass and stiffness matrices
- Gyroscopic effects included
- Proportional damping

### Solution Method
- Complex impedance matrix formulation
- Direct solution at each frequency point
- Unbalance force excitation

## Validation

The Python translation has been validated against the original MATLAB code for:
- Bearing coefficient convergence
- System natural frequencies
- Response amplitudes
- Critical speed locations

## Limitations

1. Linear analysis only (small displacements assumed)
2. Constant bearing coefficients per speed (no orbit dependency)
3. Simplified disk model (rigid body)
4. No bearing temperature effects

## References

This code is based on classical rotordynamic theory:
- Childs, D.W. "Turbomachinery Rotordynamics"
- Vance, J.M. "Rotordynamics of Turbomachinery"
- API 684 "Rotordynamic Tutorial"

## License

This code is provided for educational and research purposes. Please ensure proper attribution when using or modifying.

## Support

For questions or issues with the Python translation, please check:
1. Verify all dependencies are installed correctly
2. Ensure Python 3.7+ is being used
3. Check that input parameters are within reasonable ranges

## Example Output

When run successfully, you should see output similar to:
```
Starting rotordynamic analysis...
Calculating bearing 1 coefficients...
Calculating bearing 2 coefficients...
Bearing coefficients calculated successfully!
Generating plots...
All plots generated successfully!

==================================================
ROTORDYNAMIC ANALYSIS COMPLETED SUCCESSFULLY
==================================================
Speed range: 10 - 2000 rad/s
Number of speed points: 200
Bearing 1 reaction: 12.34 N  
Bearing 2 reaction: 10.22 N
==================================================
``` 