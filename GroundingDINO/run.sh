#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python demo/inference_on_a_image.py \
-c ./groundingdino/config/GroundingDINO_SwinB.cfg.py \
-p ./weights/groundingdino_swinb_cogcoor.pth \
-i ../example/fill-anything/sample1.png \
-o ./output \
-t "bench"

