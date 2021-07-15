import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pose_est_nets.models.regression_tracker import RegressionTracker
from pose_est_nets.models.heatmap_tracker import HeatmapTracker

from pose_est_nets.datasets.datasets import (
    RegressionDataset,
    HeatmapDataset,
    TrackingDataModule,
)
from typing import Any, Callable, Optional, Tuple, List
import json
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--no_train", action="store_true", help="whether or not to train the model"
    )
    parser.add_argument(
        "--load", action="store_true", help="set true to load model from checkpoint"
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="whether or not to generate predictions on test data",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to model checkpoint if you want to load model from checkpoint",
    )
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--validation_batch_size", type=int, default=16)  # not used now
    parser.add_argument("--test_batch_size", type=int, default=1)  # not used now
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save model outputs",
    )

    args = parser.parse_args()
    return args


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.num_workers != os.cpu_count():
        print(
            "You should set num_workers equal to the number of cpus which is: "
            + str(os.cpu_count())
        )

    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.1636, 0.1636, 0.1636], std=[0.1240, 0.1240, 0.1240]
            ),
        ]
    )

    inverse_normalize = UnNormalize(
        mean=[0.1636, 0.1636, 0.1636], std=[0.1240, 0.1240, 0.1240]
    )

    # TODO: Check out label in top left corner
    dataset = HeatmapDataset(
        root_directory="./data/mouseRunningData/",
        csv_path="CollectedData_.csv",
        header_rows=[1, 2, 3],
        transform=data_transform,
    )

    datamod = TrackingDataModule(
        dataset,
        train_batch_size=args.train_batch_size,
        validation_batch_size=args.validation_batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
    )

    """
    18: models.resnet18,
    34: models.resnet34,
    50: models.resnet50,
    101: models.resnet101,
    152: models.resnet152,
    """

    model = HeatmapTracker(num_targets=17, resnet_version=101, transfer=False)

    if args.load:
        model = model.load_from_checkpoint(
            checkpoint_path=args.ckpt,
            dataset=dataset,
            num_targets=17,
            resnet_version=101,
        )

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=100, mode="min"
    )
    trainer = pl.Trainer(
        gpus=args.num_gpus,
        log_every_n_steps=15,
        callbacks=[early_stopping],
        auto_scale_batch_size=False,
        max_epochs=1000,
    )

    if args.no_train:
        datamod.setup()
    else:
        trainer.fit(model=model, datamodule=datamod)

    if args.predict:
        predict_dl = datamod.test_dataloader()
        for idx, batch in enumerate(predict_dl):
            x, y = batch
            out = model.forward(x)
            x = inverse_normalize(x)
            x = x.squeeze().numpy()
            y = y.squeeze().numpy()

            input_img = np.moveaxis(x, 0, -1) * 255
            input_img = input_img.astype(np.uint8)
            input_img = Image.fromarray(input_img)
            draw = ImageDraw.Draw(input_img)
            r = 5

            for bp_idx in range(y.shape[0]):
                label_coords = np.unravel_index(y[bp_idx].argmax(), y[bp_idx].shape)
                draw.ellipse(
                    (
                        label_coords[1] - r,
                        label_coords[0] - r,
                        label_coords[1] + r,
                        label_coords[0] + r,
                    ),
                    fill=(255, 0, 0, 0),
                )

                out_heatmap = out.squeeze().detach().cpu().numpy()[bp_idx]
                target_coords = np.unravel_index(out_heatmap.argmax(), out_heatmap.shape)
                draw.ellipse(
                    (
                        target_coords[1] - r,
                        target_coords[0] - r,
                        target_coords[1] + r,
                        target_coords[0] + r,
                    ),
                    fill=(0, 255, 0, 0),
                )

                """
                label_heatmap = y[bp_idx] * 255
                label_heatmap = label_heatmap.astype(np.uint8)
                label_heatmap = Image.fromarray(label_heatmap)
                label_heatmap.save(idx_dir / f"{bp_idx}_label.png")

                out_heatmap = out.squeeze().detach().cpu().numpy()
                out_heatmap = out_heatmap[bp_idx] * 255
                out_heatmap = out_heatmap.astype(np.uint8)
                out_heatmap = Image.fromarray(out_heatmap)
                out_heatmap.save(idx_dir / f"{bp_idx}_prediction.png")
                """


            input_img.save(output_dir / f"{idx}_image.png")


    """
    if args.predict:
        preds = {}
        i = 1
        f = open("predictions.txt", "w")
        predict_dl = model.test_dataloader()
        for batch in predict_dl:
            x, y = batch
            plt.clf()
            out = model.forward(x)
            plt.imshow(x[0, 0])
            preds[i] = out.numpy().tolist()
            plt.scatter(out.numpy()[:, 0::2], out.numpy()[:, 1::2], c="blue")
            plt.scatter(y.numpy()[:, 0::2], y.numpy()[:, 1::2], c="orange")
            plt.savefig("preds/testv2_" + str(i) + ".png")
            i += 1
        f.write(json.dumps(preds))
        f.close()
    """
