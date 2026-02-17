# Depth-Anything-SAM2
3D Point Cloud reconstruction from a single 2D image using **SAM 2** and **Depth Anything V2**.

## Results
<table>
  <tr>
    <td><b>Original Image</b></td>
    <td><b>SAM 2 Mask</b></td>
    <td><b>3D Point Cloud</b></td>
  </tr>
  <tr>
    <td><img src="data/campanile.jpeg" width="300"></td>
    <td><img src="assets/sam2_check.jpg" width="300"></td>
    <td><img src="assets/MeshLabReconstruction.jpg" width="300"></td>
  </tr>
</table>

## Project Structure
- `src/`: Main Python scripts for depth estimation and projection.
- `data/`: Input images.
- `assets/`: Result images and screenshots.
- `checkpoints/`: Model weights (e.g., `sam2_hiera_small.pt`).

## How it Works
1. **Segmentation**: Using SAM 2 to isolate the specific object (e.g., the tower).
2. **Depth Map**: Running Depth Anything V2 to get a monocular depth estimation.
3. **Projection**: Applying a pinhole camera model to project 2D pixels into 3D space based on the depth values.

## Setup
1. Clone the repo.
2. Install SAM 2 and Depth Anything V2 requirements.
3. Place model checkpoints in the `/checkpoints/` folder.
4. Run: `python src/run_reconstruction.py`
