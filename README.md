# HGN: Hierarchical Graph Network for Multi-hop Question Answering

This is the official repository of [HGN](https://arxiv.org/abs/1911.03631) (EMNLP 2020).

## Requirements

We provide Docker image for easier reproduction. Please use `Dockerfile` or pull image directly.
```bash
docker pull studyfang/hgn:latest
```

To run docker without sudo permission, please refer this documentation [Manage Docker as a non-root user](https://docs.docker.com/install/linux/linux-postinstall/).
Then, you could start docker, e.g.
```bash
docker run --gpus all -it -v /datadrive_c/yuwfan:/ssd -it studyfang/hgn:latest bash
```

## Quick Start

*NOTE*: Please make sure you have set up the environment correctly. 

1. Download raw data and our preprocessed data. 
```bash
bash scripts/download_data.sh
```

2. Inference

We provide roberta-large and albert-xxlarge-v2 finetuned model for inference directly.
Please run
```
python predict.py --config_file configs/predict.roberta.json
```

You may get the following results on the dev set with RoBERTa-large model and ALBERT model respectively:
```
em = 0.6895340985820392
f1 = 0.8220366071156804
sp_em = 0.6310600945307225
sp_f1 = 0.8859230865915771
joint_em = 0.4649561107359892
joint_f1 = 0.7436079971145017
```
and
```
em = 0.7018230925050641
f1 = 0.8344362891739213
sp_em = 0.6317353139770425
sp_f1 = 0.8919316739978699
joint_em = 0.4700877785280216
joint_f1 = 0.7573679775376975
```

Please refer to Preprocess and training section if you want to reproduce other steps.

## Preprocess

Please set DATA_ROOT in preprocess.sh for your usage, otherwise data will be downloaded to the currect directory.

To download and preprocess the data, please run
```bash
bash run.sh download,preprocess
```

After you download the data, you could also optionally to preprocess the data only:
```bash
bash run.sh preprocess
```

## Training

Please set your home data folder `HOME_DATA_FOLDER` in envs.py.

And then run
```
python train.py --config_file configs/train.roberta.json
```

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Acknowledgment

Our code makes a heavy use of Huggingface's [PyTorch implementation](https://github.com/huggingface/transformers),
and [DFGN](https://github.com/woshiyyya/DFGN-pytorch).
We thank them for open-sourcing their projects.


## Citation

If you use this code useful, please star our repo or consider citing:
```
@article{fang2019hierarchical,
  title={Hierarchical graph network for multi-hop question answering},
  author={Fang, Yuwei and Sun, Siqi and Gan, Zhe and Pillai, Rohit and Wang, Shuohang and Liu, Jingjing},
  journal={arXiv preprint arXiv:1911.03631},
  year={2019}
}
```

## License

MIT
