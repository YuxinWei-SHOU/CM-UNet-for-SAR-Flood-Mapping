import os
import sys
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from osgeo import gdal
from merge import merge_tiffs
import albumentations as A
from albumentations.pytorch import ToTensorV2
from Utils.path_hyperparameter import ph
from Nets.CM_UNet import MambaUNet  # Your model

class InferDataset(Dataset):
    """Only for inference, does not read labels, and reuses the preprocessing steps from training"""
    def __init__(self, images_dir: str):
        self.images_dir = Path(images_dir)
        self.ids = [p.stem for p in self.images_dir.glob("*.tif")]
        self.ids.sort()
        logging.info(f'Creating inference dataset with {len(self.ids)} examples')

        # —— Must be exactly the same as during training ——
        # 1) GDAL reads as float32
        self.normalize = A.Compose([
            A.Normalize(mean=(0.0, 0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0, 1.0))
        ])
        self.to_tensor = A.Compose([ ToTensorV2() ])

    @staticmethod
    def load_float32(path: Path) -> np.ndarray:
        """GDAL reads two bands and returns (H, W, 2) float32"""
        ds = gdal.Open(str(path))
        if ds is None:
            raise FileNotFoundError(f"Cannot open {path}")
        bands = [ds.GetRasterBand(i).ReadAsArray() for i in (1,2,3,4)]
        img = np.stack(bands, axis=-1).astype(np.float32)
        return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_path = self.images_dir / f"{name}.tif"

        # —— Load float32 two bands
        img = self.load_float32(img_path)

        # —— Exactly the same as training: Normalize + ToTensorV2
        img = self.normalize(image=img)['image']
        img = self.to_tensor(image=img)['image']  # Result is a Tensor, shape=(2,H,W)

        return img, name

    def __len__(self):
        return len(self.ids)


def predict_and_save(test_loader, net, device, save_path):
    os.makedirs(save_path, exist_ok=True)
    net.eval()
    with torch.no_grad():
        for batch_img, names in tqdm(test_loader):
            # batch_img: (1, 2, H, W)
            x = batch_img.to(device)
            y = net(x)
            y = torch.sigmoid(y)
            # Inference on single channel, single sample
            mask = (y[0,0].cpu().numpy() > 0.5).astype(np.uint8) * 255

            name = names[0]
            out_fp = os.path.join(save_path, f"{name}.tif")

            # Reuse the original image's Geo information
            src_ds = gdal.Open(str(test_loader.dataset.images_dir / f"{name}.tif"))
            gt = src_ds.GetGeoTransform()
            proj = src_ds.GetProjection()
            src_ds = None

            drv = gdal.GetDriverByName('GTiff')
            out_ds = drv.Create(out_fp, mask.shape[1], mask.shape[0], 1, gdal.GDT_Byte)
            out_ds.SetGeoTransform(gt)
            out_ds.SetProjection(proj)
            out_ds.GetRasterBand(1).WriteArray(mask)
            out_ds.FlushCache()
            out_ds = None

            #logging.info(f"Saved prediction: {out_fp}")

def main():

    # Create necessary directories
    input_image_seg = f'' # Input SAR image slices to predict
    output_pred_image_seg = f'' # Output predicted slices
    output_pred_image = f'' # Output large image after predicted slices are merged

    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device {device}")

    # 1) Dataset
    input_image_seg = InferDataset(images_dir=input_image_seg)
    test_loader = DataLoader(
        input_image_seg, batch_size=1, shuffle=False,
        num_workers=8, prefetch_factor=5, persistent_workers=True
    )

    # 2) Model
    net = MambaUNet(num_classes=1, in_channels=4)
    net.to(device)
    assert ph.load, "Please specify the checkpoint path in ph.load"
    ckpt = torch.load(ph.load, map_location=device)
    net.load_state_dict(ckpt)
    logging.info(f"Model loaded from {ph.load}")

    # 3) Predict and save
    predict_and_save(test_loader, net, device, save_path=output_pred_image_seg)

    # After prediction, call the merge function
    merge_tiffs(output_pred_image_seg, output_pred_image)  # Call merge function
    logging.info(f"Merged TIFF saved as {output_pred_image}")

if __name__ == '__main__':
    try:
        main()  # Call the main function
    except KeyboardInterrupt:
        logging.info('Error')
        sys.exit(0)  # Catch keyboard interrupt signal and exit safely
