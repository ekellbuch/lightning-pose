from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback


class AnnealWeight(Callback):
    """Callback to change weight value during training."""

    def __init__(
        self,
        attr_name: str,
        init_val: float = 0.0,
        increase_factor: float = 0.01,
        final_val: float = 1.0,
        freeze_until_epoch: int = 0,
    ) -> None:
        super().__init__()
        self.init_val = init_val
        self.increase_factor = increase_factor
        self.final_val = final_val
        self.freeze_until_epoch = freeze_until_epoch
        self.attr_name = attr_name

    def on_train_start(self, trainer, pl_module) -> None:
        # Dan: removed buffer; seems to complicate checkpoint loading
        # pl_module.register_buffer(self.attr_name, torch.tensor(self.init_val))
        setattr(pl_module, self.attr_name, torch.tensor(self.init_val))

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if pl_module.current_epoch <= self.freeze_until_epoch:
            pass
        else:
            eff_epoch: int = pl_module.current_epoch - self.freeze_until_epoch
            value: float = min(
                self.init_val + eff_epoch * self.increase_factor, self.final_val
            )
            # Dan: removed buffer; seems to complicate checkpoint loading
            # pl_module.register_buffer(self.attr_name, torch.tensor(value))
            setattr(pl_module, self.attr_name, torch.tensor(value))


class ActiveLoop(Callback):
    """Callback to identify frames with low confidence after training."""

    def __init__(
            self,
            al_strategy: str = 'confidence_mean',
            al_threshold: float = 0.5,
            #max_al_frames_ploop: Optional[int] = None,

    ):
        super().__init__()
        self.al_strategy = al_strategy
        self.al_threshold = al_threshold
        #self.max_al_frames_ploop = max_al_frames_ploop

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # TODO(kellybuchanan): in predict_step add a flag to the dataloader to get the confidence maps.
        # it appears that we cannot send a flag to predict step directly.
        # raises error running predict step in callaback https://github.com/Lightning-AI/lightning/discussions/10361
        number_unused_frames = len(trainer.datamodule.al_dataset.indices)
        if trainer.current_epoch % 100 == 0 and number_unused_frames > 0:
            print("ActiveLoop @ epoch {}".format(trainer.current_epoch))
            # Get unused frames, now placed in al_dataset
            labeled_preds = trainer.predict(model=pl_module,
                                            dataloaders=trainer.datamodule.al_dataloader(),
                                            return_predictions=True)

            # Select frames to label based on al_strategy
            al_frames_idx = []
            for label_idx, (predicted_keypoints, confidence) in enumerate(labeled_preds):
                # confidence is a tensor of shape (num_frames, num_keypoints)
                # predicted_keypoints is a tensor of shape (num_frames, num_keypoints*2)
                if self.al_strategy == 'confidence_mean':
                    # select the frames in avr
                    frame_confidence = confidence.mean(-1)
                elif self.al_strategy == 'confidence_min':
                    frame_confidence = confidence.min(-1)
                else:
                    raise NotImplementedError('al_strategy {} not implemented'.format(self.al_strategy))
                subset_to_label = torch.where(frame_confidence < self.al_threshold)[0] + label_idx*len(frame_confidence)
                al_frames_idx.append(subset_to_label)
            # select subset of frames to label
            
            al_frames_idx = torch.cat(al_frames_idx).to(dtype=torch.int64)
            al_indices = torch.tensor(trainer.datamodule.al_dataset.indices).to(dtype=torch.int64)
            frames_to_label = al_indices[al_frames_idx]
            # Update dataset class with new indices:
            trainer.datamodule.train_dataset.indices += frames_to_label.tolist()
            # Update active learning pool:
            for elem in frames_to_label.tolist():
                trainer.datamodule.al_dataset.indices.remove(elem)
        else:
            pass
