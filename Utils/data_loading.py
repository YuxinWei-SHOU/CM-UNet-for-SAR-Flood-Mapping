import logging  # For logging
from os import listdir  # For file operations
from os.path import splitext  # For file operations
from pathlib import Path  # For handling file paths
import random  # For random operations
import numpy as np  # For array operations
from osgeo import gdal  # For handling image files
from torch.utils.data import Dataset  # Basic dataset class in pytorch
import albumentations as A  # For data augmentation
from albumentations.pytorch import ToTensorV2  # To convert image to tensor


class BasicDataset(Dataset):  # BasicDataset class for training, evaluation, and testing datasets
    """ Basic dataset for train, evaluation, and test.

    Attributes:
        images_dir(str): path of images.
        labels_dir(str): path of labels.
        train(bool): ensure creating a train dataset or other dataset.
        ids(list): name list of images.
        train_transforms_all(class): data augmentation applied to image and label.
    """

    def __init__(self, images_dir: str, labels_dir: str, train: bool):
        """ Init of basic dataset.

        Parameter:
            images_dir(str): path of images.
            labels_dir(str): path of labels.
            train(bool): ensure creating a train dataset or other dataset.
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.train = train

        # image name without suffix (Get list of filenames without extension by iterating through all files in images_dir)
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        self.ids.sort()  # Sort the list of filenames

        # If the list of image IDs is empty, raise a runtime error to inform the user that no image files were found
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        # Define data augmentation and transformations
        #self.train_transforms_all = A.Compose([
        #    A.Flip(p=0.5),  # Flip the image horizontally with a 50% chance
        #    A.Transpose(p=0.5),  # Transpose the image with a 50% chance
        #], additional_targets={'image1': 'image'})  # Apply the same augmentation operations to the corresponding label image as well.

        # Standardize the input data (4 bands)
        self.normalize = A.Compose([
            A.Normalize(mean=(0.0, 0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0, 1.0))
        ])

        # Convert the image data to a format suitable for PyTorch deep learning models, i.e., (C,H,W)->(channels, height, width)
        self.to_tensor = A.Compose([
            ToTensorV2()
        ])

    # Return the length of the dataset, i.e., the number of samples in the dataset
    def __len__(self):
        """ Return length of dataset."""
        return len(self.ids)  # self.ids is a list of all image filenames (without extension)

    # Set all non-zero elements in the label array to 1
    @classmethod  # @classmethod allows this method to be called on the class itself, without needing to instantiate the class. Usually named as cls
    def label_preprocess(cls, label):
        """ Binaryzation label."""
        label[label != 0] = 1
        return label

    # This block defines a class method load, which opens image files and converts them to NumPy arrays
    @classmethod
    def load(cls, filename):
        """Open image and convert image to array using GDAL."""
        try:
            filename = str(filename)  # Ensure filename is a string
            dataset = gdal.Open(filename)  # Open the image file
            if dataset is None:
                raise ValueError(f"Cannot open image: {filename}")

            bands = []
            for band_index in range(1, dataset.RasterCount + 1):  # Get data for all bands
                band = dataset.GetRasterBand(band_index)
                band_data = band.ReadAsArray()
                bands.append(band_data)

            img = np.stack(bands, axis=-1)  # Stack all band data along the new last axis to form a multi-channel image
            img = img.astype(np.float32)
            return img
        except Exception as e:
            logging.error(f"Error loading image {filename}: {e}")
            raise

    def __getitem__(self, idx):
        """
        Index the dataset.

        Index the image name list to get the image name, search for the image in the image path,
        open the image and convert it to an array.

        Preprocess the array, apply data augmentation and noise addition (optional), and convert the array to a tensor.

        Parameter:
            idx(int): The index of the dataset.

        Return:
            tensor(tensor): The tensor of the image.
            label_tensor(tensor): The tensor of the label.
            name(str): The name of the image and label.
        """
        name = self.ids[idx]  # Name is the filename of the idx-th sample in the dataset (without extension)

        # img_file and label_file are the full paths to the image and label files that match the filename. glob is a method of pathlib.Path class used to match file paths based on a wildcard pattern. It returns a generator that produces file paths matching the pattern
        # The wildcard .* means matching files with any extension
        img_file = list(self.images_dir.glob(name + '.*'))
        label_file = list(self.labels_dir.glob(name + '.*'))

        # Ensure that each image and label file is unique
        assert len(label_file) == 1, f'Either no label or multiple labels found for the ID {name}: {label_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'

        # Load the unique image and label files, and binarize the label by converting non-zero values to 1
        img = self.load(img_file[0])
        label = self.load(label_file[0])
        label = self.label_preprocess(label)

        # Apply data augmentation to image and label when in training mode
        #if self.train:  # self.train is a boolean attribute
        #   sample = self.train_transforms_all(image=img, mask=label)
        #    img, label = sample['image'], sample['mask']

        # Standardize and convert the image to tensor format
        img = self.normalize(image=img)['image']
        sample = self.to_tensor(image=img, mask=label)
        tensor, label_tensor = sample['image'].contiguous(), sample['mask'].contiguous()

        # Modify part: Remove the last dimension of labels
        if label_tensor.shape[-1] == 1:
            label_tensor = label_tensor.squeeze(-1)

        return tensor, label_tensor, name  # Return the processed image and label tensors along with the image and label filenames
