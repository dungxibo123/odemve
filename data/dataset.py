import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO


class CustomDataset(Dataset):
    def __init__(self, data_list, coco_annotations_path, coco_images_directory, transform=None):
        self.data_list = data_list
        self.coco_annotations_path = coco_annotations_path
        self.coco_images_directory = coco_images_directory
        self.transform = transform

        # Load COCO dataset
        self.coco = COCO(coco_annotations_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]

        # Load image
        image_id = sample['img_id']
        image = self.load_image(image_id)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Return data sample
        return {
            'image': image,
            'question': sample['question'],
            'answer': sample['answer']
        }

    def load_image(self, image_id):
        # Load image information
        image_info = self.coco.loadImgs(image_id)[0]

        # Construct the image file path
        image_path = self.coco_images_directory + image_info['file_name']

        # Load the image using PIL
        image = Image.open(image_path).convert('RGB')

        return image

