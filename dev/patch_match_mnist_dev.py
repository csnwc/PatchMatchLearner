# patch_match_mnist_dev.py
import argparse
import json
import os
from tqdm import tqdm
import torch
import torchvision
from utils import pairdatamodule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter, Callback
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

################################################################################################
################################################################################################
## Visualise shifts:
def visualise_shifts(dm: pairdatamodule) -> None:

    visualise_export_path = os.path.join(dm.data_dir, "visualise_shifts")
    os.makedirs(visualise_export_path)

    for batch_cnt, batch in enumerate(tqdm(dm.train_dataloader(), desc="fit visualise shifts")):
        shift_pairs, mvs, imgs, labels, paths = batch
        mont_list = []
        for shift_pair, mv, img, label, path in zip(shift_pairs, mvs, imgs, labels, paths):
            diff = torch.abs(torch.diff(shift_pair, dim=1))
            pdiff = torch.nn.functional.pad(diff, (1, 1, 1, 1))
            mont_list.append(torchvision.utils.make_grid([img, pdiff.squeeze(0)], scale_each=True))

        file_name = os.path.join(visualise_export_path, f"data_view_batch_{batch_cnt:04d}.jpg")
        torchvision.utils.save_image(torchvision.utils.make_grid(mont_list), file_name)

################################################################################################
################################################################################################
## The callbacks:
class HelperCallbacks(Callback):
    def on_fit_start(self, trainer, pl_module):
        print(f"on_fit_start: fit train {len(trainer.datamodule.p_train)} images.")
        print(f"on_fit_start: fit tval {len(trainer.datamodule.p_val)} images.")
        print(f"class_to_idx: {json.dumps(trainer.datamodule.p_train.class_to_idx, indent=2)}")

    def on_predict_start(self, trainer, pl_module):
        print(f"on_predict_start: infernce on {len(trainer.datamodule.p_predict)} images.")
        print(f"class_to_idx: {json.dumps(trainer.datamodule.p_predict.class_to_idx, indent=2)}")

################################################################################################
################################################################################################
## The inference:
class InferenceWriter(BasePredictionWriter):
    """Writes predictions on epoch end."""

    def __init__(self, output_path: str, export_box_overlay: bool=True):
        super().__init__("batch") # "epoch")
        self.output_path = output_path
        self.output_file = os.path.join(output_path, "predictions.pt")
        self.export_box_overlay = export_box_overlay

    def write_on_batch_end(
            self,
            trainer,
            pl_module,
            prediction,
            batch_indices,
            batch,
            batch_idx,
            dataloader_idx
        ) -> None:
        """ batch prediction = [pred_patch, pred_mv, pred_bbox] """
        
        output_path = os.path.join(self.output_path, "topk_diffs")
        os.makedirs(output_path, exist_ok=True)

        for pred_id, (patches, mvs, boxes) in enumerate(zip(prediction[0], prediction[1], prediction[2])):
            file_name = os.path.join(output_path,
                f"topk{len(patches)}_view_{batch_idx:04d}_{pred_id:04d}.jpg"
            )

            if self.export_box_overlay: ## Draw boxes on original frame1 image; green 1 pixel mv, red other.
                cols = [(0, 255, 0) if torch.sqrt(mv[0]*mv[0] + mv[1]*mv[1]) == 1.
                    else (255, 0, 0) for mv in mvs]
                image = (batch[0][pred_id]*255.).to(torch.uint8).repeat(3, 1, 1)
                out = torchvision.utils.draw_bounding_boxes(image, boxes=boxes, colors=cols)
                torchvision.utils.save_image(out.to(torch.float32)/255., file_name)
            else: ## export montage of top_k patches for vis/debug.
                resize_scale = 8
                torchvision.utils.save_image(
                        torchvision.transforms.functional.resize(
                            torchvision.utils.make_grid(patches), size=(patches.shape[-1]*resize_scale)
                        ),
                        file_name
                    )

    def write_on_epoch_end(
        self, trainer, module, predictions, batch_indices
    ):
        assert len(predictions) == 1
        predictions = predictions[0]
        outputs = torch.cat(predictions)
        print(f"Saving output to; {self.output_file}")
        torch.save(outputs, self.output_file)

    def read(self):
        return torch.load(self.output_file)

################################################################################################
################################################################################################
## The derived model for prediction:
from models import patch_matcher
from torch import nn

class PML_pred(patch_matcher.PML):
    def __init__(self, input_dim: int, top_k:int = 8):
        super().__init__(input_dim)
        self.input_dim = input_dim
        self.top_k = top_k
        self.stride = input_dim // 2
        self.diff_weights = torch.ones((1, 1, self.input_dim, self.input_dim), dtype=torch.float32)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        topk_vji = self.extract_max_diff_patches(batch) # x, y, paths = batch[0], batch[1], batch[2]
        pred_patch, pred_mv, pred_bbox = [], [], []
        for diff_id, (v, j, i) in enumerate(zip(topk_vji[0], topk_vji[1], topk_vji[2])):
            boxes = [
                        torch.tensor([x*self.stride, y*self.stride, x*self.stride+self.input_dim, y*self.stride+self.input_dim])
                            for y, x in zip(j, i)
                ] # as xyxy

            pred_bbox.append(torch.stack(boxes))

            topk_patch_pairs = [
                batch[0][diff_id:(diff_id+2), :, box[1]:box[3], box[0]:box[2]] 
                    for box in boxes
                ]
            
            pred_patch.append(
                    [
                    torch.abs(torch.diff(patch_pair, dim=0)).squeeze(0)
                        for patch_pair in topk_patch_pairs
                    ]
                )
            x = torch.stack(topk_patch_pairs).permute((0, 2, 1, 3, 4))
            pred_mv.append(self(x).round())

        return [pred_patch, pred_mv, pred_bbox]

    def extract_max_diff_patches(self, batch):
        sum_diff = nn.functional.conv2d(
            torch.abs(torch.diff(batch[0], dim=0)),
            self.diff_weights,
            stride=self.input_dim//2
        )
        topk_vji = self.get_topk_vji(sum_diff, top_k=self.top_k)        
        return topk_vji

    def get_topk_vji(self, x, top_k: int=1):
        topk_patches = torch.topk(torch.flatten(x.squeeze(1), -2), top_k)
        cols = torch.div(topk_patches.indices, x.shape[-1]).to(torch.int32)
        rows = torch.remainder(topk_patches.indices, x.shape[-1])
        return [topk_patches.values, cols, rows]

################################################################################################
################################################################################################
## The boiler plate:
parser = argparse.ArgumentParser(description="Entry point boiler plate for dev template.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_dir", type=str, default="./data", help="src/dst of data")
parser.add_argument("--predict_dir", type=str, default="./data/predict_images", help="The ImageFolder style location of images for prediction.")
parser.add_argument("--ckpt_path", type=str, default="./path/to/lightning_logs/version_??/checkpoints/xxx.ckpt", help="The location of the checkpoint to be used for prediction.")
parser.add_argument("--stage", type=str, default="fit", help="The phase to be applied: [fit|validate|test|predict].")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers.")
parser.add_argument("--batch_size", type=int, default=64, help="Size of image pair batches.")
parser.add_argument("--accelerator", type=str, default="cpu", help="The computer.")
parser.add_argument("--num_devices", type=int, default=1, help="Number of GPU devices.")
parser.add_argument("--output_path", type=str, default="./data/output", help="The lightning_logs and inference destination.")
parser.add_argument("--max_epochs", type=int, default=20, help="The max epochs to train.")
parser.add_argument("--top_k_dets", type=int, default=16, help="The top k magnitude frame difference patches.")
parser.add_argument("--export_box_overlay", type=bool, default=True, help="On inference export boxes overlaid on image (True) or a montage of top_k diff patches.")
parser.add_argument("--visualize_shifts", type=bool, default=False, help="Confirm correct data module preparation and visualise shifts.")

################################################################################################
################################################################################################
##
def main(args):

    print(f"Args:\n{json.dumps(vars(args), indent=2)}")
    dm = pairdatamodule.PairDataModule(
        data_dir=args.data_dir,
        predict_dir=args.predict_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    if args.visualize_shifts:
        dm.prepare_data() # Run this to make sure download and data-prep/loader are correct.
        dm.setup(stage="fit")
        visualise_shifts(dm)
        return

    if args.stage == "fit":
        num_devices = args.num_devices
        max_epochs = args.max_epochs
        strategy = None if args.accelerator=="cpu" else "ddp_find_unused_parameters_false"
        callbacks = [
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_mse_loss"),
            LearningRateMonitor("epoch")
        ]
        ckpt_path = None
    elif args.stage == "predict":
        num_devices = 1
        max_epochs = -1
        strategy = None
        ckpt_path = args.ckpt_path
        inference_writer = InferenceWriter(args.output_path, args.export_box_overlay)
        predict_helper = HelperCallbacks()
        callbacks = [
            inference_writer,
            predict_helper
        ]

    trainer = pl.Trainer(
        devices=num_devices,
        accelerator=args.accelerator,
        default_root_dir=args.output_path,
        strategy=strategy,
        max_epochs=max_epochs,
        callbacks=callbacks
    )

    if args.stage == "fit":
        pml = patch_matcher.PML(input_dim=26)
        trainer.fit(model=pml, datamodule=dm, ckpt_path=ckpt_path)
    elif args.stage == "test":
        pass
    elif args.stage == "predict":
        pml_pred = PML_pred(input_dim=26, top_k=args.top_k_dets)
        trainer.predict(model=pml_pred, datamodule=dm, ckpt_path=ckpt_path)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)