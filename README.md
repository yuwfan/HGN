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

2. Model training
```
python train.py --config_file configs/train.roberta.json
```

3. Inference
```
python predict.py
```

If you want to preprocess data again, please refer to Preprocess section.

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

## Inference on Fullwiki Setting

## Citation
If you use this code useful, please star our repo or consider citing:
```
@inproceedings{fang2020HGN,
  title={Hierarchical Graph Network for Multi-hop Question Answering,
  author={Yuwei Fang, Siqi Sun, Zhe Gan, Rohit Pillai, Shuohang Wang, Jingjing Liu},
  booktitle={EMNLP},
  year={2020}
}
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

## License

MIT
