# AI Project-PPE Compliance Recognition

## Project Description
This project is designed to run on a Raspberry Pi 5 and uses a USB-connected camera to track eye positions with Haar Cascade, calculate the facial region, and detect whether a face mask is being worn correctly. It targets a frame rate of approximately 12 FPS and a latency of about 50ms.

## Hardware Requirements
- **Raspberry Pi 5**: This project must be run on a Raspberry Pi 5.
- **Camera**: A USB camera must be connected to the Raspberry Pi to capture video input.

## Software Requirements
### TensorFlow Lite Environment
This project requires the TensorFlow Lite environment to be installed on the Raspberry Pi. For installation instructions and more details, visit the official TensorFlow Lite documentation here: [TensorFlow Lite](https://www.tensorflow.org/lite/guide).

### Setup and Installation
1. Ensure your Raspberry Pi 5 is set up and running.
2. Connect a USB camera to the Raspberry Pi.
3. Install TensorFlow Lite by following the guidelines provided in the official documentation linked above.

## Running the Project
To run the project, follow these steps:
1. Clone or download this project to your Raspberry Pi.
2. Open a terminal and navigate to the project directory:
   ```bash
   cd path/to/project-directory
   ```
3. Run the program using the following command:
   ```bash
   python Main.py
   ```
## Additional Information
This system uses the Haar Cascade algorithm to track eye positions and subsequently detects the face mask compliance based on the calculated facial region.

## Contributions
Contributions to this project are welcome. Please ensure to follow the best practices and coding standards.

## License
Specify your license or state that the project is unlicensed.
