# 川柳を評価するAI

## 概要

川柳を「面白い」か「面白くない」かを判定するAI

## データ

* 面白くない川柳
  - ここにあるもの全部（35000作品）
  - http://www.okajoki.com/db/search.php?page=327

* 面白い川柳
  - 第一生命のサラリーマン川柳コンクール
    - http://event.dai-ichi-life.co.jp/company/senryu/index.html
  - 面白”川柳”のまとめ（絶対、笑える！）
    - https://matome.naver.jp/odai/2132045818468665101

* テスト(評価) 用川柳
  - シルバー川柳=>面白いはず
    - http://www.yurokyo.or.jp/news/silversenryu.html
  - +面白くない川柳の一部をテストデータにする

## やり方

* Character-level CNN前提。RNNにはしない。

~~~1. 川柳を単語分解　１０次元に分解。川柳が１０単語に満たない分は、空白で埋める。~~~
1. 
2. 単語を数値化する
3. 数値化された単語１０次元＋「面白い」「面白くない」の２次元ラベルをつけ、教師データとする
4. 面白い川柳、面白くない川柳を3の状態のデータにしてCNNに学習させる
5. 学習された機械にテスト用川柳を評価させる


### モデル

* ４層くらいのDNN 隠れ層はReLU, 出力層はステップ関数
* CNN カーネルサイズは１？

### 単語の数値化

* Word Embedding
* word2vec
* GloVe
* WordNet
* Doc2vec
* fastText
* Bag-of-words
* Term Frequency

http://gagbot.net/machine-learning/terminology2

https://datumstudio.jp/blog/%E3%80%90%E7%89%B9%E5%88%A5%E9%80%A3%E8%BC%89%E3%80%91-%E3%81%95%E3%81%81%E3%80%81%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86%E3%82%92%E5%A7%8B%E3%82%81%E3%82%88%E3%81%86%EF%BC%81-2

### 情報の順位付け


### mecab

### mechanizeがないのでmechanicalSoup

pip install MechanicalSoup

やっぱりrubyのmechanizeでやる


https://keras.io/ja/datasets/
 

## pythonメモ

### 文字の数値化
```
int.from_bytes(u'あ'.encode('utf-8'),'big')
```

### numpyで配列の追加
```
import numpy as np
x = np.array( [] )
for i in range( 10 ):
    x = np.append( x, i )
```

### 文字列をnumpy配列にする
```
np.array(list('あいうえお'))
```

### numpyで次元の変換
```
np.zeros(6).reshape(2,3)
```

### 文字列を数値化してリストにする
x = []
for c in list('あいうえお'):
  x.append(np.int.from_bytes(c.encode('utf-8'),'big'))

### 配列の長さを揃える
y=np.array([])
y.resize(10,refcheck=False)
y
array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])

### csv

```
import csv

with open('some.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)  # ヘッダーを読み飛ばしたい時

    for row in reader:
        print row          # 1行づつ取得できる
```

```
import pandas as pd

df = pd.read_csv('some.csv')

print df       # show all column
print df['A']  # show 'A' column
```
