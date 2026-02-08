Write-Host "Uninstalling existing PyTorch installation..."
pip uninstall -y torch torchvision
Write-Host "Installing PyTorch with CUDA 12.6 support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
Read-Host -Prompt "Press Enter to exit"
