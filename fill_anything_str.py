import cv2
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt

from grounding_dino import get_grounding_output
from sam_segment import predict_masks_with_sam
from stable_diffusion_inpaint import fill_img_with_sd
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point


def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    # parser.add_argument(
    #     "--coords_type", type=str, required=True,
    #     default="key_in", choices=["click", "key_in"], 
    #     help="The way to select coords",
    # )
    # parser.add_argument(
    #     "--point_coords", type=float, nargs='+', required=True,
    #     help="The coordinate of the point prompt, [coord_W coord_H].",
    # )
    # parser.add_argument(
    #     "--point_labels", type=int, nargs='+', required=True,
    #     help="The labels of the point prompt, 1 or 0.",
    # )
    
    parser.add_argument(
        "--box_threshold", type=float, default=0.3, 
        help="box threshold",
    )
    parser.add_argument(
        "--text_threshold", type=float, default=0.25, 
        help="text threshold",
    )
    parser.add_argument(
        "--src_prompt", type=str, required=True,
        help="prompt of src object",
    )
    parser.add_argument(
        "--dst_prompt", type=str, required=True,
        help="prompt of dst object",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--grounddino_model_type", type=str,
        default="swin_t", choices=['swin_t', 'swin_b'],
        help="The type of Grounding DINO model to load. Default: 'swin_t"
    )
    parser.add_argument(
        "--grounddino_ckpt", type=str, required=True,
        help="The path to the Grounding DINO checkpoint to use for box generation.",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b'],
        help="The type of sam model to load. Default: 'vit_h"
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--seed", type=int,
        help="Specify seed for reproducibility.",
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use deterministic algorithms for reproducibility.",
    )


if __name__ == "__main__":
    """Example usage:
    python fill_anything.py \
        --input_img FA_demo/FA1_dog.png \
        --coords_type key_in \
        --point_coords 750 500 \
        --point_labels 1 \
        --text_prompt "a teddy bear on a bench" \
        --dilate_kernel_size 15 \
        --output_dir ./results \
        --sam_model_type "vit_h" \
        --sam_ckpt sam_vit_h_4b8939.pth 
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = load_img_to_array(args.input_img)

    boxes_filt, pred_phrases = get_grounding_output(
        img, args.grounddino_model_type, args.grounddino_ckpt, 
        args.src_prompt, args.box_threshold, args.text_threshold,
    )
    print(pred_phrases)
    H, W, _ = img.shape
    boxes = boxes_filt * torch.Tensor([W, H, W, H])
    boxes[:,:2] -= boxes[:,2:] / 2
    boxes[:,2:] += boxes[:,:2]
    # import pdb
    # pdb.set_trace()
    masks, mask_scores, _ = predict_masks_with_sam(
        img,
        model_type=args.sam_model_type,
        ckpt_p=args.sam_ckpt,
        device=device,
        box=boxes[0].numpy(),
    )
    masks = masks.astype(np.uint8) * 255
    # import pdb
    # pdb.set_trace()

    # dilate mask to avoid unmasked edge effect
    if args.dilate_kernel_size is not None:
        masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

    # visualize the segmentation results
    img_stem = Path(args.input_img).stem
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks):
        # path to the results
        mask_p = out_dir / f"mask_{idx}.png"
        img_points_p = out_dir / f"with_points.png"
        img_mask_p = out_dir / f"with_{Path(mask_p).name}"

        # save the mask
        save_array_to_img(mask, mask_p)

        # save the pointed and masked image
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        # show_points(plt.gca(), [latest_coords], args.point_labels,
        #             size=(width*0.04)**2)
        plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()

    # fill the masked image
    for idx, mask in enumerate(masks):
        if args.seed is not None:
            torch.manual_seed(args.seed)
        mask_p = out_dir / f"mask_{idx}.png"
        img_filled_p = out_dir / f"filled_with_{Path(mask_p).name}"
        img_filled = fill_img_with_sd(
            img, mask, args.dst_prompt, device=device)
        save_array_to_img(img_filled, img_filled_p)