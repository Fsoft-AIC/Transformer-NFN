# Equivariant Neural Functional Networks for Transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Conference](https://img.shields.io/badge/ICLR-2025-blue)](https://iclr.cc/Conferences/2025)

Official implementation of the paper "Equivariant Neural Functional Networks for Transformers" (ICLR 2025).


## Install dependencies
Tested with conda, python 3.9, CUDA 12.1.
```
conda create -n nfn python=3.9
conda activate nfn
pip install -r requirement.txt
pip install -e .
```

## Download data
```
wget https://huggingface.co/datasets/anonymized-acamedia/Small-Transformer-Zoo/resolve/main/AG-News-Transformers.zip?download=true
wget https://huggingface.co/datasets/anonymized-acamedia/Small-Transformer-Zoo/resolve/main/MNIST-Transformers.zip?download=true
unzip MNIST-Transformers.zip -d data
unzip AG-News-Transformers.zip -d data
mv data/MNIST-Transformers data/mnist_transformer
mv data/AG-News-Transformers data/ag_news_transformer
```

## Run Transformer-NFN model
```
python nfn_transformer/main.py --enc_mode inv --classifier_nfn_channels 10,10 --transformers_nfn_channels 10 --wandb False --dataset mnist --data_path data/mnist_transformer --cut_off 0
python nfn_transformer/main.py --enc_mode inv --emb_mode no --classifier_nfn_channels 10,10 --transformers_nfn_channels 10  --wandb False --dataset ag_news --data_path data/ag_news_transformer --cut_off 0
```

## Run MLP Baseline
```
python nfn_transformer/main.py --enc_mode mlp --classifier_nfn_channels 50,50 --transformers_nfn_channels 50 --num_out_classify 50 --num_out_embedding 50 --num_out_encoder 50 --wandb False --dataset mnist --data_path data/mnist_transformer --cut_off 0
python nfn_transformer/main.py --enc_mode mlp --emb_mode no --classifier_nfn_channels 50,50 --transformers_nfn_channels 50 --num_out_classify 50 --num_out_encoder 50 --wandb False --dataset ag_news --data_path data/ag_news_transformer --cut_off 0
```

## Run StatNN Baseline
```
python nfn_transformer/main.py --enc_mode statnn --cls_mode statnn --classifier_nfn_channels 256 --transformers_nfn_channels 256 --num_out_classify 256 --num_out_embedding 64 --num_out_encoder 256 --wandb False --dataset mnist --data_path data/mnist_transformer --cut_off 0
python nfn_transformer/main.py --enc_mode statnn --cls_mode statnn --emb_mode no --classifier_nfn_channels 256 --transformers_nfn_channels 256 --num_out_classify 256 --num_out_encoder 256 --wandb False --dataset ag_news --data_path data/ag_news_transformer --cut_off 0
```

## Run XGBoost Baseline
```
python nfn_transformer/xgb.py --wandb False --dataset mnist --data_path data/mnist_transformer --cut_off 0
python nfn_transformer/xgb.py --wandb False --dataset ag_news --data_path data/ag_news_transformer --cut_off 0
```

## Run LightGBM Baseline
```
python nfn_transformer/gbm.py --model gbdt --wandb False --dataset mnist --data_path data/mnist_transformer --cut_off 0
python nfn_transformer/gbm.py --model gbdt  --wandb False --dataset ag_news --data_path data/ag_news_transformer --cut_off 0
# If you encounter GPU issues with LightGBM, please look at https://github.com/microsoft/LightGBM/issues/586#issuecomment-352845980
```

## Run Random Forest Baseline
```
python nfn_transformer/gbm.py --model rf --wandb False --dataset mnist --data_path data/mnist_transformer --cut_off 0
python nfn_transformer/gbm.py --model rf  --wandb False --dataset ag_news --data_path data/ag_news_transformer --cut_off 0
```

## Citation

If you find this code useful in your research, please cite our paper:

```bibtex
@inproceedings{tran2025equivariance,
    title={Equivariant Neural Functional Networks for Transformers},
    author={Viet-Hoang Tran and Thieu N. Vo and An Nguyen The and Tho Tran Huu and Minh-Khoi Nguyen-Nhat and Thanh Tran and Duy-Tung Pham and Tan Minh Nguyen},
    booktitle={International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=uBai0ukstY}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the code or paper, please open an issue in this repository or contact the authors directly.