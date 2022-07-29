"""Data modules split a dataset into train, val, and test modules."""

from nvidia.dali.plugin.pytorch import LastBatchPolicy
import os
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split, Subset
from typeguard import typechecked
from typing import Dict, List, Literal, Optional, Tuple, Union

from lightning_pose.data.dali import PrepareDALI, LitDaliWrapper
from lightning_pose.data.utils import \
    split_sizes_from_probabilities, compute_num_train_frames
from lightning_pose.utils.io import check_video_paths

_TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: add typechecks here


class BaseDataModule(pl.LightningDataModule):
    """Splits a labeled dataset into train, val, and test data loaders."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        use_deterministic: bool = False,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        test_batch_size: int = 1,
        num_workers: int = 8,
        train_probability: float = 0.8,
        val_probability: Optional[float] = None,
        test_probability: Optional[float] = None,
        train_frames: Optional[Union[float, int]] = None,
        torch_seed: int = 42,
        use_deterministic_split: bool = False,
        deterministic_split: Optional[List] = None,
    ) -> None:
        """Data module splits a dataset into train, val, and test data loaders.

        Args:
            dataset: base dataset to be split into train/val/test
            use_deterministic: TODO: use deterministic split of data...?
            use_deterministic_split: use split of data for train/val/test
            train_batch_size: number of samples of training batches
            val_batch_size: number of samples in validation batches
            test_batch_size: number of samples in test batches
            num_workers: number of threads used for prefetching data
            train_probability: fraction of full dataset used for training
            val_probability: fraction of full dataset used for validation
            test_probability: fraction of full dataset used for testing
            train_frames: if integer, select this number of training frames
                from the initially selected train frames (defined by
                `train_probability`); if float, must be between 0 and 1
                (exclusive) and defines the fraction of the initially selected
                train frames
            torch_seed: control data splits

        """
        super().__init__()
        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        # maybe can make the view information more general when deciding on a
        # specific format for csv files
        self.use_deterministic = use_deterministic
        self.use_deterministic_split = use_deterministic_split
        # info about dataset splits
        self.train_probability = train_probability
        self.val_probability = val_probability
        self.test_probability = test_probability
        self.train_frames = train_frames
        self.train_dataset = None  # populated by self.setup()
        self.val_dataset = None  # populated by self.setup()
        self.test_dataset = None  # populated by self.setup()
        self.torch_seed = torch_seed
        self.deterministic_split = deterministic_split

    def setup(self, stage: Optional[str] = None):  # stage arg needed for ptl

        if self.use_deterministic:
            return

        # Use deterministic/explicit split of data
        if self.use_deterministic_split:
            train_frames = self.deterministic_split[0]
            val_frames = self.deterministic_split[1]
            test_frames = self.deterministic_split[2]
            self.train_dataset = Subset(self.dataset, train_frames)
            self.val_dataset = Subset(self.dataset, val_frames)
            self.test_dataset = Subset(self.dataset, test_frames)
            return

        datalen = self.dataset.__len__()
        print(
            "Number of labeled images in the full dataset (train+val+test): {}".format(
                datalen
            )
        )

        # split data based on provided probabilities
        data_splits_list = split_sizes_from_probabilities(
            datalen,
            train_probability=self.train_probability,
            val_probability=self.val_probability,
            test_probability=self.test_probability,
        )

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            data_splits_list,
            generator=torch.Generator().manual_seed(self.torch_seed),
        )

        # further subsample training data if desired
        if self.train_frames is not None:

            n_frames = compute_num_train_frames(
                len(self.train_dataset), self.train_frames)

            if n_frames < len(self.train_dataset):
                # split the data a second time to reflect further subsampling from
                # train_frames
                self.train_dataset.indices = self.train_dataset.indices[:n_frames]

        print(
            "Size of -- train set: {}, val set: {}, test set: {}".format(
                len(self.train_dataset), len(self.val_dataset), len(self.test_dataset)
            )
        )
    
    def setup_video_prediction(self, video: str):
        """ this will depend on context flag in dataset"""
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
        )
    
    def full_labeled_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


class UnlabeledDataModule(BaseDataModule):
    """Data module that contains labeled and unlabled data loaders."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        video_paths_list: Union[List[str], str],
        dali_config: Union[dict, DictConfig],
        use_deterministic: bool = False,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        test_batch_size: int = 1,
        num_workers: int = 8,
        train_probability: float = 0.8,
        val_probability: Optional[float] = None,
        test_probability: Optional[float] = None,
        train_frames: Optional[float] = None,
        torch_seed: int = 42,
    ) -> None:
        """Data module that contains labeled and unlabeled data loaders.

        Args:
            dataset: pytorch Dataset for labeled data
            video_paths_list: absolute paths of videos ("unlabeled" data)
            use_deterministic: TODO: use deterministic split of data...?
            train_batch_size: number of samples of training batches
            val_batch_size: number of samples in validation batches
            test_batch_size: number of samples in test batches
            num_workers: number of threads used for prefetching data
            train_probability: fraction of full dataset used for training
            val_probability: fraction of full dataset used for validation
            test_probability: fraction of full dataset used for testing
            train_frames: if integer, select this number of training frames
                from the initially selected train frames (defined by
                `train_probability`); if float, must be between 0 and 1
                (exclusive) and defines the fraction of the initially selected
                train frames
            torch_seed: control data splits
            torch_seed: control randomness of labeled data loading

        """
        super().__init__(
            dataset=dataset,
            use_deterministic=use_deterministic,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            num_workers=num_workers,
            train_probability=train_probability,
            val_probability=val_probability,
            test_probability=test_probability,
            train_frames=train_frames,
            torch_seed=torch_seed,
        )
        self.video_paths_list = video_paths_list
        self.filenames = check_video_paths(self.video_paths_list)
        self.num_workers_for_unlabeled = num_workers // 2
        self.num_workers_for_labeled = num_workers // 2
        self.dali_config = dali_config
        self.unlabeled_dataloader = None  # initialized in setup_unlabeled
        super().setup()
        self.setup_unlabeled()

    def setup_unlabeled(self):
        """Sets up the unlabeled data loader."""
        # dali prep
        # TODO: currently not controlling context_frames_successive. internally it is
        # set to False.
        dali_prep = PrepareDALI(
            train_stage="train",
            model_type="context" if self.dataset.do_context else "base",
            filenames=self.filenames,
            resize_dims=[self.dataset.height, self.dataset.width],
            dali_config=self.dali_config
        )

        self.unlabeled_dataloader = dali_prep()

    def train_dataloader(self):
        loader = {
            "labeled": DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                num_workers=self.num_workers_for_labeled,
                persistent_workers=True,
            ),
            "unlabeled": self.unlabeled_dataloader,
        }
        return loader

    # TODO: check if necessary
    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers_for_labeled,
        )
