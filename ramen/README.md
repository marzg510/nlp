# ramen database


## Character-level CNN

- https://qiita.com/bokeneko/items/c0f0ce60a998304400c8

### データ加工

1. レビューデータからレビューと点数だけ取り出す
```
awk -F'\t' 'NR>1 {print $5"\t"$6}' reviews.txt | head
```
2. レビューテキストからノイズ除去 <br />とか
<br />
\0
```
awk -F'\t' 'NR>1 {print $5"\t"$6}' reviews.txt | sed 's#<br /># #g'
```

3. 点数をネガポジ情報に変換

最初は
 - 30点未満ならネガティブ評価 = 0
 - 30点以上70点未満はどちらでもない(ミドル) = 1
 - 70点以上ならポジティブ評価 = 2
 
としたが、９割がたが2になるので、以下のように変更中

変更後
 - 50点未満ならネガティブ評価 = 0
 - 80点以上ならポジティブ評価 = 2
 - 50点以上80点未満はどちらでもない(ミドル) = 1

分布を確認
```
awk -F'\t' 'NR>1 {print $5}' reviews.txt | sort -n | uniq -c
awk -F'\t' 'NR>1 {eval=($5<50 ? 0 : ($5>=80 ? 2 : 1) );print eval}' reviews.txt | sort | uniq -c
   1270 0
  52043 1
  54992 2

```


4. テキストの最大値を確認
  入力層の要素数を決めたい
```
awk -F'\t' 'NR>1 {gsub("<br />"," ",$6);print length($6)}' reviews.txt | sort -gr | head
```
=>8500にする



5. 準備データ
- reviews-ccnn-teacher-data.txt
    - レビュー点数,NEGA/MIDL/POSI判定、テキスト
    - 10万件
    - 初期の分類条件
- reviews-ccnn-teacher-data01.txt
    - 判定数値、テキスト
    - 10件
    - 初期の分類条件
- reviews-ccnn-teacher-data02.txt
    - 判定数値、テキスト
    - 10万件
    - 初期の分類条件
- reviews-ccnn-teacher-data03.txt
    - 判定数値、テキスト
    - 10万件
    - 2つめの分類条件
- reviews-ccnn-train-data04.txt
    - 判定数値、テキスト
    - 10万件
    - 2つめの分類条件
    - nullデータの削除版

```
#awk -F'\t' 'NR>1 {eval=($5<30 ? 0 : ($5>=70 ? 2 : 1) );print $5"\t"eval"\t"$6}' reviews.txt | sed 's#<br /># #g' | head > reviews-ccnn-teacher-data01.txt
awk -F'\t' 'NR>1 {eval=($5<30 ? 0 : ($5>=70 ? 2 : 1) );print eval"\t"$6}' reviews.txt | sed 's#<br /># #g' | head > reviews-ccnn-teacher-data01.txt
awk -F'\t' 'NR>1 {eval=($5<30 ? 0 : ($5>=70 ? 2 : 1) );print eval"\t"$6}' reviews.txt | sed 's#<br /># #g' > reviews-ccnn-teacher-data02.txt
awk -F'\t' 'NR>1 {eval=($5<50 ? 0 : ($5>=80 ? 2 : 1) );print eval"\t"$6}' reviews.txt | sed 's#<br /># #g' > reviews-ccnn-teacher-data03.txt
awk -F'\t' 'NR>1 {eval=($5<50 ? 0 : ($5>=80 ? 2 : 1) );print eval"\t"$6}' reviews.txt | sed 's#<br /># #g' | sed 's#\x0##g' > reviews-ccnn-train-data04.txt
```

## 第一段階
- Character-level CNNを自分なりに解釈しまずは実装
- 文字をUNIODEに変換、コード値を単純にUNICODEの最大値で割ったものベクトルとしてを入力層に投入
- 文章自体は１次元なので１次元の畳み込み層を作成
- 結果
　　半々の予測になる　間違いも多い

## 第２段階
- 1次元より２次元のほうが特徴を捉えやすいのではないかと仮設
- 入力のベクトル表現を密ベクトル化して２次元化
- 畳み込み層を２次元に変更
- 蜜ベクトル化のアルゴリズム詳細は不明




## command memo
### 解凍
```
unzip ramendb2.zip
```

### GPUのステータス確認
```
nvidia-smi
```
```
Wed Mar  7 03:35:15 2018
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 384.111                Driver Version: 384.111                   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla M60           On   | 00000000:00:1E.0 Off |                    0 |
| N/A   66C    P0    92W / 150W |   7356MiB /  7613MiB |     97%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      2455      C   ...naconda3/envs/tensorflow_p36/bin/python  7345MiB |
+-----------------------------------------------------------------------------+
```


