# C-Former

This repository is the implementation for Paper "Transformers are Good Clusterers for LifeLong User Behavior Sequence Modeling".




## Requirements

* Ensure you have Python and PyTorch (version 1.8 or higher) installed. Our setup utilized Python 3.8 and PyTorch 1.13.0.
* Should you wish to leverage GPU processing, please install CUDA.



## Dataset

We use three public real-world datasets (Taobao, Alipay and Tmall) in our experiments. You can download the datasets from the links below.

- **Taobao**: The raw dataset can be downloaded from https://tianchi.aliyun.com/dataset/dataDetail?dataId=649. If you want to know how to preprocess the data, please refer to `./data/taobao_900/preprocess.py`
- **Alipay**: The raw dataset can be downloaded from https://tianchi.aliyun.com/dataset/dataDetail?dataId=53. If you want to know how to preprocess the data, please refer to `./data/alipay_900/preprocess.py`
- **Tmall**: The raw dataset can be downloaded from https://tianchi.aliyun.com/dataset/dataDetail?dataId=42. If you want to know how to preprocess the data, please refer to `./data/tmall_900/preprocess.py`




## Example

If you have downloaded the source codes, you can train C-Former model. 

```
$ cd main
$ python build_tmall_900_to_parquet.py
$ python run_expid.py
```

You can change the model parameters in `./main/config/General_config/model_config.yaml`


