````
# Hand-Eye Calibration Project

## Overview
This repository implements a standard **eye-in-hand hand-eye calibration** pipeline based on OpenCV.
The pipeline completes automatic chessboard corner detection, camera parameter calibration, and extrinsic transformation solving between the camera and robot end-effector coordinate systems.

The generated transformation matrix serves as a fundamental prerequisite for vision-based robotic perception, 6D object pose estimation, and sim-to-real deployment of robotic manipulation tasks.

## Environment Dependencies
### Recommended Python Version
Python 3.10

### Required Packages
```bash
pip install opencv-python numpy open3d

````

## Project Structure

plaintext

```
hand-eye-calibration/
├── images/            # Storage of chessboard calibration image sequences
├── eyehand.py         # Main program for calibration computation
├── poses.txt          # Recorded robot end-effector poses
├── raw_data.txt       # Saved camera intrinsics, distortions and calibration results
└── README.md          # Project documentation

```

## Hardware Setup

- **Camera**: Monocular camera mounted on robot end-effector (eye-in-hand)
- **Calibration Target**: Standard chessboard calibration plate
- **Robotic Manipulator**: Six-degree-of-freedom collaborative robot

## Usage

1. Fix the camera installation position and keep the hardware stable.
2. Execute the calibration script:

bash

运行

```
python eyehand.py

```

1. Acquire image data from diverse viewing angles.
2. The program automatically solves and persists the hand-eye transformation parameters.
3. Apply the calibrated matrix to downstream visual localization and manipulation algorithms.

## Troubleshooting

- **Chessboard detection failure**

  &#x20;Avoid overexposure, shadow occlusion and reflective interference. Ensure the calibration board is flat and occupies a dominant area in the field of view.
- **Low calibration accuracy**

  &#x20;Increase the number of sampling frames, enrich perspective diversity, and avoid highly similar shooting poses. Keep the device static during data acquisition.
- **Invalid pose matching**

  &#x20;Maintain stable robot motion during sampling and eliminate external vibration interference.

## Applications

- Vision-guided robotic grasping and manipulation
- Real-scene 6D object pose estimation
- Sim-to-real domain adaptation for robotic simulation systems
- Visual positioning, sorting and high-precision robotic tasks

