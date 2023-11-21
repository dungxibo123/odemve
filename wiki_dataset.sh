#!/bin/bash

# Create a directory to store the datasets
mkdir -p datasets
cd datasets

# Download the WikiMatrix dataset
echo "Downloading WikiMatrix dataset..."
wget https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/en-vi.txt.zip
unzip en-vi.txt.zip
rm en-vi.txt.zip

# Download the Wikipedia English-Vietnamese dataset
echo "Downloading Wikipedia English-Vietnamese dataset..."
wget https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-vi.txt.zip
unzip en-vi.txt.zip
rm en-vi.txt.zip

echo "Download complete."
