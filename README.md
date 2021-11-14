# 2021-NEU-NLP课程

## Requirements

python==3.8

pytorch==1.4.0

## Model

LSTM模型定义在utils.py文件中，引入：

```python
from utils import MyLSTM
```

## Checkpoint

models文件夹中有两个bin文件记录checkpoint，"LSTMlm_model_epoch5.bin"为torch.nn.LSTM的训练结果，"MyLSTM_model_epoch5.bin"为MyLSTM的训练结果。

## Training

The command to train is listed below:

```
python LSTMLM.py
```

## Test

The command to test is listed below:

```
python test.py  # model_type  0 or 1  # 0: torch.nn.LSTM(); 1: MyLSTM()
```

## Resualt

| Model         | loss     | ppl     |
| ------------- | -------- | ------- |
| torch.nn.LSTM | 5.727529 | 307.209 |
| MyLSTM        | 5.783124 | 324.772 |

## Others
其他的一些与课程相关的尝试
