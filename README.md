# Footfall Detection System using YOLOv8

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Mathematical Concepts](#mathematical-concepts)
- [Contributing](#contributing)
- [License](#license)

## Introduction
We developed a footfall detection system using Python and YOLOv8 to monitor and record the number of people entering and exiting a specific area in real-time. This system utilizes object detection tracking and geometric algorithms to determine movement direction across a predefined line. The data is stored in a database for further analysis.

## Features
- Real-time monitoring of foot traffic
- Object detection using YOLOv8
- Direction determination using geometric algorithms
- Data storage in SQLite3 for further analysis

## Requirements

### Hardware
- CCTV camera with RTSP streaming
- Computer with GPU (optional for better performance)

### Software
- Python 3.x
- OpenCV
- NumPy
- SQLite3
- YOLOv8 (ultralytics package)
- Configuration file (configurations.json)

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/footfall-detection-system.git
    cd footfall-detection-system
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Set up the configuration file:
    - Create a `configurations.json` file based on the provided template and update it with your specific settings.

## Usage
1. Run the main script to start the footfall detection system:
    ```bash
    python CCTV_IN_OUT.py
    ```
2. The system will start processing the video stream from the CCTV camera and record the number of people entering and exiting the area.

## Mathematical Concepts

### Line Crossing
To detect line crossing, we use the line equation `ax + by + c = 0`. The sign of this equation indicates a point's position relative to the line. A change in sign between frames indicates crossing.

### Direction Determination
We calculate the dot product between the user-defined direction vector and the object's movement vector. The cosine of the angle between these vectors determines if the movement aligns with the defined direction.

## Contributing
Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
