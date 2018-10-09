数据：产业链数据
    train：5638条
    dev：626条
    test：1510条

BiLSTM-ATT:
    batch-size:32
    num-epoch:30
    hidden_size:800
    不使用预训练词向量
    结果：loss 0.197743, acc 0.960265

    batch-size:32
    num-epoch:40
    hidden_size:800
    不使用预训练词向量： loss 0.177909, acc 0.964901
    使用预训练词向量： loss 0.175751, acc 0.958278


BiLSTM:
    batch-size:32
    num-epoch:30
    hidden_size:800
    不使用预训练词向量
    结果：loss 0.451592, acc 0.948344

    batch-size:32
    num-epoch:40
    hidden_size:800
    不使用预训练词向量
    结果：loss 1.00402, acc 0.933113

结论：
    1、相同训练参数情况下，加上Attention层之后分类结果明显变好；
    2、仅LSTM时训练用40个epoch有点过拟合

