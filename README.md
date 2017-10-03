Natural Language Processing
===============




## tensorflow

pip install tensorflow


## keras

pip install keras

```
sudo apt-get install -y python-pip python-dev
sudo apt-get install -y python-tk graphviz
# install tensorflow
pip install --upgrade tensorflow
# install keras
pip install --upgrade keras
# install hdf5
pip install --upgrade h5py
# install for visualize
pip install matplotlib
pip install pydot graphviz
```

## 用語集

### パーセプトロン

* 人工ニューロンやニューラルネットワークの一種である。
* 視覚と脳の機能をモデル化したものであり、パターン認識を行う。シンプルなネットワークでありながら学習能力を持つ。
  - 単純パーセプトロン (Simple perceptron)
    入力層と出力層のみの2層からなる  線形非分離な問題を解けない
  - 多層パーセプトロン
    * デビッド・ラメルハートとジェームズ・マクレランドはパーセプトロンを多層にし、バックプロパゲーション（誤差逆伝播学習法）で学習させることで、線型分離不可能な問題が解けるように、単純パーセプトロンの限界を克服した。


### MLP
* Multi Layer Perceptron 多層パーセプトロン

### 全結合層(Full conneced layer)
* すべてのノードが次の層のすべてのノードとつながっている層のこと

### ロジスティック回帰
* 活性化関数として、ステップ関数の代わりにシグモイド関数を使ったモデル
* ステップ関数を使うのは「単純パーセプトロン」?
* 確率的な分類になる 単純な０／１ではない

### 確率変数
* http://www.geisya.or.jp/~mwm48961/statistics/variable1.htm
* ラベルでなく変数（数値化）?

### 多クラスロジスティック回帰(multi-class logistic regression)
* 活性化関数はsoftmax

### エポック数
* http://st-hakky.hatenablog.com/entry/2017/01/17/165137
* エポック数とは、「一つの訓練データセットを何回繰り返して学習させるか」の数のことです。
* Early Stopping

### 活性化関数(Activation Function)
* 入力値がしきい値を超えると急激に変化する関数
* http://qiita.com/namitop/items/d3d5091c7d0ab669195f
* 活性化関数は、入力信号の総和がどのように活性化するかを決定する役割を持ちます。これは、次の層に渡す値を整えるような役割をします。
  - シグモイド(sigmoid) 任意の実数を0~1の間に写像する
  - ReLU(rectified linear unit)
  - tanh
  - softmax 多クラス分類で使う
  - 階段関数 (ステップ関数)
     正の数では１を返し、負の数では０を返すような関数

### 尤度関数 ゆうど〜 (likelihood function)

* 重みw,バイアスbを最尤推定するための関数
* 最尤とは
  - 統計学において、与えられたデータからそれが従う確率分布の母数を点推定する方法である。
* https://detail.chiebukuro.yahoo.co.jp/qa/question_detail/q1439818099
  - 起きた現象を、「最も引き起こす確率が高い」場合を考えるのが最尤法です。
  - 最尤推定量は、汎用性があり、少ない観測値からも一定の推定量を導ける点で有用ですが、必ずしも高い精度で値を推定しているとは言えません。
* https://www.slideshare.net/iranainanimosuteteshimaou/ss-55173144
  - 観測されたデータDから分布H（のパラメータ）を推定すること

### 損失関数(loss function) 誤差関数(error function)とも

* 尤度関数の対数をとって符号を反転したもの
  - -log L(w,b)
  - 尤度関数は積になり（Π＝Σの掛け算版）、微分が難しくなるため簡単にする（ということらしい）
* https://keras.io/ja/getting-started/sequential-model-guide/
  - モデルが最小化しようとする目的関数です．引数として，定義されている損失関数の識別子を文字列として与える（categorical_crossentropyやmseなど），もしくは目的関数を関数として与えることができます．
* http://qiita.com/mine820/items/f8a8c03ef1a7b390e372
  - ようは2つの値の差が小さくなるような関数のこと
* 最適な状態からどのくらいの誤差があるのかを表す関数
  - 交差エントロピー誤差関数
  - E(w,b)

~~* =>バックプロパゲーションにおいて、誤差を最小化するための関数?~~

### 最適化問題(optimization problem)

* 関数が最大・最小となる状態を求める問題のこと
* 一般に「関数を最適化する」とは、関数を最小化するパラメータを求めること（最大の符号を反転させれば最小になるので）
* 尤度関数を最大化するには尤度関数を各パラメータで偏微分する
  - 微分してゼロになる値=勾配ゼロの値=最小点(最大点)

### 最適化アルゴリズム

* 引数として，定義されている最適化手法の識別子を文字列として与える（rmspropやadagradなど），もしくは Optimizerクラスのインスタンスを与えることができます
  * 勾配降下法(gradient descent)
  * sgd(確率的勾配降下法)
  * ミニバッチ勾配降下法
  * rmsprop

~~* =>バックプロパゲーションにおいて、誤差を最適化するための関数?~~

### 学習率

* ハイパーパラメータの一種
* 最初は大きく、徐々に小さくしていくのがよい
* モメンタム:学習率そのものは固定だが、調整項で擬似的にこれを表現
* Nesterovモメンタム
* Adagrad:学習率そのものを自動調整
* Adadelta
* RMSprop
* Adam


### ハイパーパラメータ


### 勾配消失問題(vanishing gradient problem)

* モデルの学習において各パラメータの勾配を求めるがこの勾配がゼロになってしまう問題のこと
* 層が増えるほど顕著になる
* 層が深くなくても起こる　次元数が多い場合とか(シグモイドに渡されるWx+bの値がおおきくなるため)
* この問題を回避するには「微分しても値が小さくならない活性化関数」を考える必要がある

### オーバーフィッティング、過学習、過剰適合(overfitting)

### ドロップアウト

* オーバーフィッティングを防ぐために、学習の際にニューロンをランダムにドロップアウト(除外)させるもの
* ドロップアウト確率pは、一般的には0.5が選ばれる


## 川柳を評価するAI

### 概要

川柳を「面白い」か「面白くない」かを判定するAI

### データ

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

### やり方

* CNN前提。RNNにはしない。

1. 川柳を単語分解　１０次元に分解。川柳が１０単語に満たない分は、空白で埋める。
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
 

### pythonメモ

文字の数値化
```
int.from_bytes(u'あ'.encode('utf-8'),'big')
```

numpyで配列の追加
```
import numpy as np
x = np.array( [] )
for i in range( 10 ):
    x = np.append( x, i )
```

文字列をnumpy配列にする
```
np.array(list('あいうえお'))
```

numpyで次元の変換
```
np.zeros(6).reshape(2,3)
```

文字列を数値化してリストにする
x = []
for c in list('あいうえお'):
  x.append(np.int.from_bytes(c.encode('utf-8'),'big'))

配列の長さを揃える
y=np.array([])
y.resize(10,refcheck=False)
y
array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])

csv

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