#!/bin/bash

mkdir weights
cd ./weights

# SALAD (~ 340 MiB)
echo "Downloading SALAD weights (~ 340 MiB) ..."
SALAD_URL="https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
curl -L "$SALAD_URL" -o "./dino_salad.ckpt"


# DA3NESTED-GIANT-LARGE-1.1
echo "Downloading DA3NESTED-GIANT-LARGE-1.1 weights and config (~ 6.76 GiB)..."
BASE_URL="https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1/resolve/main"

# download config.json (~ 3.1 KiB)
curl -L "$BASE_URL/config.json" -o "./config.json"

# download model.safetensors (~ 6.76 GiB)
curl -L "$BASE_URL/model.safetensors" -o "./model.safetensors"
