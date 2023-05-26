set -e

# *** reproduce results ***
# see https://github.com/Verg-Avesta/CounTR/issues/23



# To set proxy
export http_proxy=http://127.0.0.1:7777 https_proxy=http://127.0.0.1:7777
export socks_proxy=socks5://127.0.0.1:1080

# To run training
torchrun --standalone --nnodes=1 --nproc-per-node=4 FSC_finetune_cross.py --epochs 800 --batch_size 8 --lr 5e-7 --output_dir ./data/train/CLIP-TextOnly-Randshot --title CLIP-TextOnly-Randshot --resume FSC147.pth --blr 1e-4 --wandb CounTR --data_path /tmp/datasets

torchrun --standalone --nnodes=1 --nproc-per-node=4 FSC_finetune_cross.py --epochs 1000 --batch_size 8 --lr 1e-5 --output_dir ./data/train/CLIP-Image-Text --title CLIP-Image-Text

# To run testing
python FSC_test_cross\(few-shot\).py --output_dir ./data/out/aaa --resume ./your-weights/FSC147.pth