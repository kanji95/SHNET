## Command to train on Referit

### python3 main.py --batch_size 50 --num_workers 4 --optimizer AdamW --dataroot <data_path> --lr 1.2e-4 --weight_decay 9e-5 --image_encoder deeplabv3_plus --loss bce --dropout 0.3 --epochs 50 --gamma 0.7 --num_encoder_layers 2 --image_dim 448 --mask_dim 112 --phrase_len 25 --glove_path <glove_path> --threshold 0.40 --task referit --feature_dim 18 --sfm_dim 512 --channel_dim 512 --save
