#!/bin/sh
python fill_anything_str.py \
    --input_img ./example/fill-anything/sample1.png \
    --src_prompt "white dog"  \
    --dst_prompt "bear" \
    --dilate_kernel_size 50 \
    --output_dir ./results \
    --grounddino_model_type "swin_t" \
    --grounddino_ckpt ./pretrained_models/groundingdino_swint_ogc.pth \
    --sam_model_type "vit_h" \
    --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth
