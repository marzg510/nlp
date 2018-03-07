# ramen database


## Character-based CNN

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

```
#awk -F'\t' 'NR>1 {eval=($5<30 ? 0 : ($5>=70 ? 2 : 1) );print $5"\t"eval"\t"$6}' reviews.txt | sed 's#<br /># #g' | head > reviews-ccnn-teacher-data01.txt
awk -F'\t' 'NR>1 {eval=($5<30 ? 0 : ($5>=70 ? 2 : 1) );print eval"\t"$6}' reviews.txt | sed 's#<br /># #g' | head > reviews-ccnn-teacher-data01.txt
awk -F'\t' 'NR>1 {eval=($5<30 ? 0 : ($5>=70 ? 2 : 1) );print eval"\t"$6}' reviews.txt | sed 's#<br /># #g' > reviews-ccnn-teacher-data02.txt
awk -F'\t' 'NR>1 {eval=($5<50 ? 0 : ($5>=80 ? 2 : 1) );print eval"\t"$6}' reviews.txt | sed 's#<br /># #g' > reviews-ccnn-teacher-data03.txt
awk -F'\t' 'NR>1 {eval=($5<50 ? 0 : ($5>=80 ? 2 : 1) );print eval"\t"$6}' reviews.txt | sed 's#<br /># #g' | sed 's#\x0##g' > reviews-ccnn-train-data04.txt
```
```

4. テキストの最大値を確認
  入力層の要素数を決めたい
```
awk -F'\t' 'NR>1 {gsub("<br />"," ",$6);print length($6)}' reviews.txt | sort -gr | head
```
=>8500にする



5. 準備データ
- reviews-ccnn-teacher-data.txt
 -- レビュー点数,NEGA/MIDL/POSI判定、テキスト
 -- 10万件
 -- 初期の分類条件
- reviews-ccnn-teacher-data01.txt
 -- 判定数値、テキスト
 -- 10件
 -- 初期の分類条件
- reviews-ccnn-teacher-data02.txt
 -- 判定数値、テキスト
 -- 10万件
 -- 初期の分類条件
- reviews-ccnn-teacher-data03.txt
 -- 判定数値、テキスト
 -- 10万件
 -- 2つめの分類条件
- reviews-ccnn-train-data04.txt
 -- 判定数値、テキスト
 -- 10万件
 -- 2つめの分類条件
 -- nullデータの削除版
