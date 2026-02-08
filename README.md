# ScreenVLM: AI Screen Assistant

ScreenVLM is a desktop application that captures your screen and allows you to ask questions about it using a fine-tuned Vision-Language Model (VLM). It combines screen capture, local VLM inference, and an optional RAG (Retrieval-Augmented Generation) system.

## Features

-   **Screen Capture**: Instantly captures your screen for analysis.
-   **Visual Q&A**: Ask questions about the content on your screen.
-   **Local Inference**: Runs a fine-tuned SmolVLM model locally.
-   **RAG Support**: Optional integration with document retrieval.
-   **Cross-Platform**: Designed for Windows (with CUDA support) and compatible with other platforms.

## Installation

### Prerequisites

-   Python 3.10+
-   (Optional) NVIDIA GPU with CUDA drivers for faster inference.

### 1. Set up a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate
```

### 2. Install Dependencies

Install the core dependencies using `pip`:

```powershell
pip install -r requirements.txt
```

### 3. (Optional) Enable CUDA Support

If you have an NVIDIA GPU, you should install the CUDA-enabled version of PyTorch for significantly better performance. Run the provided script:

**Command Prompt (cmd):**
```batch
install_cuda.bat
```

**PowerShell:**
```powershell
.\install_cuda.ps1
```

> **Note:** This script will uninstall the default CPU-only `torch` version and install the CUDA 12.6 version.

## Usage

The application is run via the command-line interface (CLI). Ensure your virtual environment is activated.

### Launch the Application

To start the UI and the VLM worker:

```powershell
python -m screenvlm.cli run
```

### System Health Check

Run the `doctor` command to verify your installation and environment:

```powershell
python -m screenvlm.cli doctor
```

### Other Commands

-   **Ingest Documents (RAG)**: `python -m screenvlm.cli ingest --docs <path_to_docs>`
-   **Merge Adapter**: `python -m screenvlm.cli merge --out <output_dir>`
-   **Help**: `python -m screenvlm.cli --help`

### Project Structure

-   `screenvlm/`: Main application source code.
-   `vlm_qlora/`: Fine-tuned LoRA adapter files.
-   `requirements.txt`: Python package dependencies.
-   `install_cuda.bat/ps1`: Scripts for CUDA installation.
