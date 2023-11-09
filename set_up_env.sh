#!/bin/bash
# CUDA Version: 11.2 
#GPU: Tesla V100-SXM2-32GB
#if you have torch with GPU version, you can skip this step
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install datasets
pip install peft
pip install scipy
pip install -q peft transformers 
pip install sentencepiece
pip install git+https://github.com/huggingface/transformers
pip install accelerate