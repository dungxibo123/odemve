mkdir -p datasets
cd datasets

# Download the VQAv2 dataset - questions
echo "Downloading VQAv2 Questions..."
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip v2_Questions_Train_mscoco.zip
unzip v2_Questions_Val_mscoco.zip
rm v2_Questions_Train_mscoco.zip
rm v2_Questions_Val_mscoco.zip

# Download the VQAv2 dataset - annotations
echo "Downloading VQAv2 Annotations..."
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip v2_Annotations_Train_mscoco.zip
unzip v2_Annotations_Val_mscoco.zip
rm v2_Annotations_Train_mscoco.zip
rm v2_Annotations_Val_mscoco.zip

echo "Download complete."
