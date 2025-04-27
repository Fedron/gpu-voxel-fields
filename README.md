# Fast GPU Generation of Distance Fields from a Voxel Grid

<p align="center">
  <img src="assets/demo1.gif" width="45%" style="display: inline-block; margin-right: 10px;" />
  <img src="assets/demo2.gif" width="45%" style="display: inline-block;" />
</p>


## 🚀 Overview
Fast GPU Generation of Distance Fields from a Voxel Grid is a real-time voxel painting application built in Rust, designed to explore and optimize techniques for generating distance fields directly on the GPU.
The project compares several existing algorithms and proposes a new optimized approach suitable for interactive applications.

Whether you're painting in voxels or pushing the limits of real-time computation, this project showcases how GPU acceleration can make complex field generation *blazingly fast*.

## ✨Features
- Interactive voxel painting interface
- Real-time distance field generation using GPU shaders
- A comparison of multiple distance field algorithms in the report
- Optimized distance field generation pipeline for minimal latency

## 🛠️ Running the Application
You can either **download a prebuilt executable** for your system **or** built it yourself from source.

### 📦 Download
Prebuilt executables are available on the [Releases](https://github.com/Fedron/gpu-voxel-fields/releases) page for:
- Windows
- macOS
- Linux

### 🔨 Building from Source
If you prefer building yourself a nightly version of rust is required for the following features:
- `generic_const_exprs`
- `duration_millis_float`
- `map_try_insert`
- `variant_count`


To build the project:
1. Install Rust Nightly:
```bash
rustup install nightly
rustup default nightly
```
2. Clone the repository:
```bash
git clone https://github.com/Fedron/gpu-voxel-fields.git
cd gpu-voxel-fields/implementation
```
3. Run the application using Cargo:
```bash
cargo run --release
```

## 📈 Results Summary
Through a comparison of existing distance field generation techniques, this project develops an optimized GPU-based method that achieves real-time performance, significantly outperforming naive approaches especially on larger voxel grids.

**Key improvements:**
- Reduction in computation time by 99% compared to a brute force approach
- Improved scalability to larger voxel volumes

**Limitations:**
- Large memory usage as memory representation and compression was not a focus of the paper
- Ray marcher computation time bottleneck at large voxel volumes

## 📂 Repository Structure

```
/implementation - Rust source code for the voxel painting application and distance field generation compute shader
/report         - LaTeX source files for the full dissertation report, and a complete PDF
```

## 📜 License
This project is licensed under the [MIT License](LICENSE).
