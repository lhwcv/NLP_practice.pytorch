## L4


使用BiLSTM + CRF 进行快递单信息抽取

### prerequisites

#### Software

```

cd  NLP_practice/
pip install --no-cache-dir -r requirements.txt  -i https://mirrors.aliyun.com/pypi/simple/

```

### Train

```
cd kg_practice/L4/
sh run.sh

```
results will save in ./exp/


### results in Waybill test
```
              precision    recall  f1-score   support

          A1       1.00      1.00      1.00       198
          A2       0.99      0.99      0.99       196
          A3       0.97      0.96      0.97       197
          A4       0.73      0.72      0.73       199
           P       0.90      0.81      0.85       200
           T       0.81      0.79      0.80       200

   micro avg       0.90      0.88      0.89      1190
   macro avg       0.90      0.88      0.89      1190
weighted avg       0.90      0.88      0.89      1190

===> f1_score:  0.8894557823129251

```

### TODO

增加demo 来进行结构化信息抽取
