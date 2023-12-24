#!/bin/bash
# CUDA Version: 11.2 
#GPU: Tesla V100-SXM2-32GB
#if you have torch with GPU version, you can skip this step
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install datasets
pip install peft==0.5.0
pip install accelerate
pip install bitsandbytes
pip install scipy
pip install sentencepiece
pip install transformers==4.33.1
pip install protobuf
