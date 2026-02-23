# YOLO26n on Hailo-10H (40 TOPS) - Raspberry Pi 5

> **Real-time object detection powered by the Hailo-10H NPU on Raspberry Pi 5**

![Hailo-10H](https://img.shields.io/badge/Hailo-10H%2040%20TOPS-blue)
![Python](https://img.shields.io/badge/Python-3.13+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ⚠️ **CRITICAL: Hardware Compatibility Warning**

**This repository is ONLY compatible with the Hailo-10H (AI HAT+ 2).** 

❌ **DO NOT use with Hailo-8L (AI Kit v1)** — incompatible drivers will cause firmware mismatch errors.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Software Installation](#software-installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Performance Metrics](#performance-metrics)
- [Usage Guide](#usage-guide)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [ROS2 Integration](#ros2-integration)
- [Training Your Own Model](#training-your-own-model)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This repository provides a complete implementation of **YOLO26n** running on the **Hailo-10H NPU** (40 TOPS) via the Raspberry Pi AI HAT+ 2. It includes:

- ✅ Optimized `.hef` model file (compiled for Hailo-10H)
- ✅ Python inference scripts (single image & live camera)
- ✅ NMS-free architecture (end-to-end detection)
- ✅ Hardware-accelerated preprocessing pipeline
- ✅ Benchmark utilities

**Use Cases:**
- Robotics vision systems
- ROS2 integration for autonomous navigation
- Real-time surveillance
- Proof-of-concept for edge AI deployment

---

## 🛠 Hardware Requirements

| Component | Specification |
|-----------|---------------|
| **SBC** | Raspberry Pi 5 (8GB or 16GB recommended) |
| **NPU** | Hailo-10H AI HAT+ 2 (40 TOPS) |
| **Camera** | Raspberry Pi Camera Module 3 or compatible |
| **Storage** | 32GB+ microSD card (for model and dependencies) |

### Verifying Your Hardware

After installation, verify your Hailo chip:

```bash
hailortcli fw-control identify
```

**Expected Output:**
```
Control Protocol Version: 2
Firmware Version: 5.1.1 (release,app)
Logger Version: 0
Device Architecture: HAILO10H
```
---

## 💻 Software Installation

### 1️⃣ Install Hailo Drivers (Hailo-10H Specific)

**If you previously installed Hailo-8L drivers, purge them first:**

```bash
sudo apt purge hailo-all hailofirmware
sudo apt autoremove
```

**Now install the Hailo-10H stack:**

```bash
sudo apt update
sudo apt install hailo-h10-all
sudo reboot
```

### 2️⃣ Create Python Virtual Environment

```bash
cd yolo26_test
python3 -m venv hailo_venv --system-site-packages
source hailo_venv/bin/activate
pip install numpy opencv-python picamera2
```

**Why `--system-site-packages`?**  
This allows access to system-installed Hailo libraries while keeping your dependencies isolated.

### 3️⃣ Verify Installation

```bash
python3 -c "from hailo_platform import VDevice; print('Hailo SDK: OK')"
```

If this prints `Hailo SDK: OK`, you're ready to proceed.

---

## 🚀 Quick Start

### Run Inference on a Single Image

```bash
source hailo_venv/bin/activate
python3 inference.py
```

**Expected Output:**
```
Running NPU Inference...
Decoding bounding boxes...
Detected: bus: 87.3%
Detected: person: 72.1%
[SUCCESS] Image saved to output_image.jpg
```

### Run Live Camera Detection

```bash
python3 live_camera.py
```

Press `Ctrl+C` to stop. The video will be saved to `headless_output.avi`.

### Run FPS Benchmark

```bash
python3 live_bench.py
```

This will output the **raw NPU throughput** (typically 200-240 FPS).

---

## 📂 Project Structure

```
yolo26_test/
├── inference.py          # Single image detection script
├── live_camera.py        # Live camera inference with recording
├── live_bench.py         # FPS benchmarking tool
├── yolo26n.hef           # Compiled Hailo model (40 TOPS optimized)
├── custom_yolo.json      # Model metadata
├── test_image.jpg        # Sample input image
├── output_image.jpg      # Detection output (gitignored)
├── .gitignore            # Git exclusion rules
└── README.md             # This file
```

---

## ⚡ Performance Metrics

| Metric | Value |
|--------|-------|
| **NPU Speed (raw)** | 243 FPS |
| **Live Pipeline (Python)** | 14-15 FPS |
| **Model Size** | 4.7 MB |
| **Input Resolution** | 640×640 RGB |
| **Power Draw** | ~3W (NPU only) |

### Bottleneck Analysis

The current Python implementation is CPU-bound due to:
1. OpenCV frame capture overhead
2. NumPy array conversions
3. Box decoding in Python

**Planned Optimization:** GStreamer-based pipeline (target: 60+ FPS)

---

## 📖 Usage Guide

### `inference.py` - Single Image Detection

**What it does:**
- Loads a `.jpg` image
- Runs inference on the Hailo NPU
- Decodes bounding boxes
- Applies Non-Maximum Suppression (NMS)
- Saves annotated output

**Modifying the model path:**
```python
HEF_PATH = "yolo26n.hef"
IMAGE_PATH = "test_image.jpg"
OUTPUT_PATH = "output_image.jpg"
```

### `live_camera.py` - Live Detection with Recording

**Features:**
- Hardware-accelerated ISP resizing (640×640)
- Real-time bounding box overlay
- Headless recording to `.avi` file
- FPS counter

**Key Parameters:**
```python
threshold=0.45  # Detection confidence threshold
nms_threshold=0.5  # Overlap suppression threshold
```

### `live_bench.py` - Raw NPU Benchmark

**Purpose:**
- Tests the **theoretical maximum throughput** of the NPU
- Uses `picamera2.devices.Hailo` for zero-copy operation
- No post-processing overhead

---

## 🧠 Technical Details

### Model Architecture

**YOLO26n** is an NMS-free, end-to-end detection model with:

- **Input:** `(1, 640, 640, 3)` RGB image
- **Outputs:** 6 tensors across 3 scales (80×80, 40×40, 20×20)
  - 3 bounding box tensors (shape: `[res, res, 4]`)
  - 3 classification tensors (shape: `[res, res, 80]`)

### Conversion Pipeline

```
YOLO26 (.pt) → ONNX → Hailo Dataflow Compiler → .hef (NPU-ready)
```

**Tools used:**
- Hailo Model Zoo
- Hailo Dataflow Compiler v3.29+
- Calibration dataset: 64 images (proof-of-concept)

### Coordinate Decoding

The model outputs **distance-from-cell-center** coordinates:

$$
\begin{align*}
c_x &= (x + 0.5) \times \text{stride} \\
c_y &= (y + 0.5) \times \text{stride} \\
x_{\min} &= c_x - (l \times \text{stride}) \\
y_{\min} &= c_y - (t \times \text{stride}) \\
w &= (r + l) \times \text{stride} \\
h &= (b + t) \times \text{stride}
\end{align*}
$$

Where `[l, t, r, b]` are the raw box outputs and `stride ∈ {8, 16, 32}`.

---

## 🔧 Troubleshooting

### Error: "Device not found"

**Cause:** PCIe connection issue or driver mismatch.

**Fix:**
```bash
lspci | grep Hailo
# Should show: "Hailo Technologies Ltd. Hailo-10H AI Processor"
```

If nothing appears:
1. Reseat the HAT+
2. Check ribbon cable connection
3. Verify PCIe is enabled in `raspi-config`

### Error: "Firmware version mismatch"

**Cause:** Wrong driver version for Hailo-10H.

**Fix:** Reinstall drivers as shown in [Software Installation](#software-installation).

### Low FPS (~2-5 FPS)

**Possible causes:**
- Running outside virtual environment (use `source hailo_venv/bin/activate`)
- Camera running at wrong resolution
- Thermal throttling (check with `vcgencmd measure_temp`)

---

## 🤖 ROS2 Integration

To use this model in a ROS2 navigation stack:

1. Install `hailo_ros_common`:
   ```bash
   sudo apt install ros-humble-hailo-ros-common
   ```

2. Create a ROS2 node that:
   - Subscribes to `/camera/image_raw`
   - Runs inference via `hailo_platform`
   - Publishes detections to `/detections`

**Reference Package:** [hailo_ros_common](https://github.com/hailo-ai/hailo-rpi5-examples)

---

## 🎓 Training Your Own Model

### Current Model Status

⚠️ **This is a proof-of-concept model** trained on only 64 images. Accuracy is low (false positives expected).

### To Train a Production Model:

1. **Collect Data:** 1,000+ labeled images
2. **Train YOLO26:** Use Ultralytics or official YOLO26 repo
3. **Export to ONNX:**
   ```python
   model.export(format='onnx', dynamic=False, imgsz=640)
   ```
4. **Convert to `.hef`:**
   - Use Hailo Dataflow Compiler
   - Provide calibration dataset (representative images)
   - Optimize for INT8 quantization

**Expected training time:** 2-4 hours on RTX 3090.

---

**Please open an issue before starting major work.**

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

**Model weights:** Check YOLO26 repository for licensing.

---

## 📞 Support

If this repo helped your project, please ⭐ **star it**!

For issues specific to:
- **Hailo drivers:** [Hailo Community Forum](https://community.hailo.ai)
- **This code:** Open a GitHub issue
- **ROS2 integration:** Ask on [ROS Answers](https://answers.ros.org)

---

