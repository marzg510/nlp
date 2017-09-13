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
  * 単純パーセプトロン (Simple perceptron)
    入力層と出力層のみの2層からなる  線形非分離な問題を解けない
  * 多層パーセプトロン
* デビッド・ラメルハートとジェームズ・マクレランドはパーセプトロンを多層にし、バックプロパゲーション（誤差逆伝播学習法）で学習させることで、線型分離不可能な問題が解けるように、単純パーセプトロンの限界を克服した。


### MLP
* Multi Layer Perceptron 多層パーセプトロン

### 全結合層(Full conneced layer)
* すべてのノードが次の層のすべてのノードとつながっている層のこと

### ロジスティック回帰
* 活性化関数として、ステップ関数の代わりにシグモイド関数を使ったモデル
* 確率的な分類になる 単純な０／１ではない

### 確率変数
* http://www.geisya.or.jp/~mwm48961/statistics/variable1.htm
* ラベルでなく変数（数値化）?

### 多クラスロジスティック回帰(multi-class logistic regression)

### エポック数
* http://st-hakky.hatenablog.com/entry/2017/01/17/165137
* エポック数とは、「一つの訓練データセットを何回繰り返して学習させるか」の数のことです。

### 活性化関数(Activation Function)
* 入力値がしきい値を超えると急激に変化する関数

* http://qiita.com/namitop/items/d3d5091c7d0ab669195f
* 活性化関数は、入力信号の総和がどのように活性化するかを決定する役割を持ちます。これは、次の層に渡す値を整えるような役割をします。
  * シグモイド(sigmoid) 任意の実数を0~1の間に写像する
  * ReLU
  * softmax 多クラス分類で使う
  * 階段関数 (ステップ関数)
     正の数では１を返し、負の数では０を返すような関数

### 損失関数(loss function) 誤差関数(error function)とも
* https://keras.io/ja/getting-started/sequential-model-guide/
* モデルが最小化しようとする目的関数です．引数として，定義されている損失関数の識別子を文字列として与える（categorical_crossentropyやmseなど），もしくは目的関数を関数として与えることができます．
* http://qiita.com/mine820/items/f8a8c03ef1a7b390e372
* ようは2つの値の差が小さくなるような関数のこと
* 最適な状態からどのくらいの誤差があるのかを表す関数
 - 交差エントロピー誤差関数E(w,b)

#####* =>バックプロパゲーションにおいて、誤差を最小化するための関数?

### 尤度関数 ゆうど〜 (likelihood function)

* 最尤推定するための関数
* 最尤とは
 - 統計学において、与えられたデータからそれが従う確率分布の母数を点推定する方法である。
* https://detail.chiebukuro.yahoo.co.jp/qa/question_detail/q1439818099
 - 起きた現象を、「最も引き起こす確率が高い」場合を考えるのが最尤法です。
 - 最尤推定量は、汎用性があり、少ない観測値からも一定の推定量を導ける点で有用ですが、必ずしも高い精度で値を推定しているとは言えません。
* https://www.slideshare.net/iranainanimosuteteshimaou/ss-55173144
 - 観測されたデータDから分布H（のパラメータ）を推定すること

### 最適化問題(optimization problem)

* 関数が最大・最小となる状態を求める問題のこと
* 一般に「関数を最適化する」とは、関数を最小化するパラメータを求めること（最大の符号を反転させれば最小になるので）
* 尤度関数を最大化するには尤度関数を各パラメータで偏微分する

### 最適化アルゴリズム

* 引数として，定義されている最適化手法の識別子を文字列として与える（rmspropやadagradなど），もしくは Optimizerクラスのインスタンスを与えることができます
* =>バックプロパゲーションにおいて、誤差を最適化するための関数?
  * 勾配降下法(gradient descent)
  * sgd(確率的勾配降下法)
  * ミニバッチ勾配降下法
  * rmsprop

### 学習率

### ハイパーパラメータ

