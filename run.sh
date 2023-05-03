#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python grounding_dino.py \
-c ./GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py \
-p ./GroundingDINO/weights/groundingdino_swinb_cogcoor.pth \
-i ./example/fill-anything/sample1.png \
-o ./GroundingDINO/output \
-t "bench"
