# Rubik's Cube Solver

**Live Demo:** [cubesolverv2.onrender.com](https://cubesolverv2.onrender.com)

A computer vision pipeline and full-stack web application designed to solve standard Rubik's Cubes. The solver actively detects cube faces from webcam feeds or image uploads, classifies the facelet colors, and generates an optimal solution sequence.

## Technical Architecture & Vision Pipeline

The application processes user-uploaded or live-captured webcam frames through a custom OpenCV pipeline before routing the normalized state to Kociemba's algorithm.

**1. Contour-Based Facelet Detection**
Rather than assuming a rigid 3x3 grid (which fails if the camera angle is slightly skewed), the system dynamically detects the facelets in 3D space:
* It applies a bilateral filter to smooth textures and lighting glares while preserving hard edges.
* It combines Canny edge detection with Adaptive Thresholding to find geometric boundaries.
* Contours are filtered by area, perimeter approximation (quadrilaterals), and aspect ratio (0.65 to 1.55) to isolate sticker-like shapes.
* The system ranks bounding boxes by area consistency, picks the 9 best matches, and sorts them into a logical 3x3 top-to-bottom, left-to-right grid.

**2. Color Extraction & Classification**
Once the 9 facelets are isolated, the system extracts the color data to rebuild the digital state:
* The region of interest (ROI) for each facelet is converted from BGR to the HSV (Hue, Saturation, Value) color space to better handle variations in room lighting and shadows.
* A central pixel patch is sampled from each facelet.
* The pixels are passed through tuned heuristic thresholds to classify them into one of the 6 standard Rubik's Cube colors (White, Yellow, Green, Blue, Red, Orange).

**3. Kociemba Resolution & Frontend State Management**
Once all 54 facelets are detected (or manually corrected via the interactive 2D UI net), the string is validated (ensuring exactly 9 of each color and 6 unique centers). The backend utilizes the Kociemba two-phase algorithm to calculate an optimal solution (typically under 22 moves), which the Vanilla JS frontend translates into a step-by-step interactive stepper.

## Tech Stack

* **Backend:** Python 3, FastAPI, Uvicorn (ASGI web server)
* **Computer Vision:** OpenCV (`opencv-python-headless` for server deployment), NumPy
* **Solver Logic:** `kociemba` (Two-Phase algorithm)
* **Frontend:** Vanilla JavaScript, HTML5, CSS3 (No external frameworks to ensure minimal latency)
* **Hosting:** Render (Web Service deployment)

## Local Development Setup

To run this project locally on your machine:

1. **Clone the repository and navigate into it.**
2. **Install the dependencies:**
   ```bash
   pip install fastapi uvicorn opencv-python numpy python-multipart kociemba