# Mini-Jev 開発・学習計画

## 1. 目的

TypeSafe AI の Jev が目指していると考えられる、

> **自然言語を理解し、自由な文章を生成するのではなく、意思決定（Decision）を確率として直接出力するモデル**

の仕組みを、Hugging Face + PyTorch を使ってローカルPC上で段階的に再現・理解する。

ただし、Jevの内部アーキテクチャはすべて公開されているわけではないため、以下では **「Jevそのものの再現」ではなく「Jev-inspired / Mini-Jev」** を開発する。

---

# 2. 全体ロードマップ

```text
Phase 0
Jevの仕組みを理解
    ↓
Phase 1
BERTによる通常のテキスト分類
    ↓
Phase 2
Questionを入力に追加
    ↓
Phase 3
Dynamic Choice Decision Model
    ↓
Phase 4
Probability Calibration
    ↓
Phase 5
Agentへの組み込み
    ↓
Phase 6
RLCD-likeな学習方法を検討
```

---

# 3. Phase 0：Jevの仕組みを理解する

## 目的

Jevと一般的なLLMの違いを理解する。

### 一般的なLLM

```text
Input
  ↓
Tokenizer
  ↓
Transformer
  ↓
Hidden Representation
  ↓
LM Head
  ↓
次のTokenの確率
  ↓
Token生成
  ↓
次のToken
  ↓
...
```

Autoregressive Generationによって文章を生成する。

---

## Jevの仮説

Jevの内部構造は公開情報だけでは断定できないが、仮説として、

```text
Input
  ↓
Transformer / Language Model
  ↓
Hidden Representation
  ↓
Decision Head
  ↓
ChoiceごとのScore
  ↓
Softmax
  ↓
Probability
```

という構造を考える。

つまり、

> 「次にどのTokenを生成するか」

ではなく、

> 「与えられた選択肢のどれを選ぶべきか」

を直接計算する。

### 重要

これは **Jevの内部実装が実際にこの構造であると確認されたものではない**。

Mini-Jevでは、この仮説を実際に実装して検証する。

---

# 4. Phase 1：BERTによる通常の分類器

## 目的

まず、最も基本的な

```text
Transformer
    ↓
Hidden Representation
    ↓
Classification Head
    ↓
Class Probability
```

を実装する。

ここではまだ「Jev」を作らない。

**普通のBERTテキスト分類問題を解く。**

---

## 4.1 アーキテクチャ

```text
「商品が壊れていたので返品したい」
              ↓
          Tokenizer
              ↓
             BERT
              ↓
       Hidden Representation
              ↓
        Classification Head
              ↓
            Softmax
              ↓
┌──────────────────────┐
│ 配送       0.02       │
│ 返品       0.91       │
│ 支払い     0.01       │
│ 商品不良   0.06       │
└──────────────────────┘
```

---

## 4.2 BERTの役割

BERTを「Embeddingモデル」と捉えることもできるが、より正確には、

> **入力テキストを意味的なHidden Representationへ変換するEncoder**

として利用する。

例えば、

```text
Text
 ↓
BERT
 ↓
[CLS] hidden vector
 ↓
768-dimensional vector
```

というようにベクトル表現を得る。

そのベクトルをClassification Headに渡す。

---

## 4.3 Classification Head

例えば、

```python
logits = Linear(hidden_state)
probabilities = Softmax(logits)
```

とする。

4クラスなら、

```text
hidden vector
      ↓
Linear
      ↓
4-dimensional logits
      ↓
Softmax
      ↓
4つの確率
```

となる。

---

## 4.4 学習

まずは通常のCross Entropy Lossを使う。

```text
Input
 ↓
BERT
 ↓
Classification Head
 ↓
Prediction
 ↓
Cross Entropy Loss
 ↓
Backpropagation
 ↓
Parameter Update
```

### Phase 1で確認すること

* BERTのhidden representationとは何か
* Classification Headとは何か
* Logitとは何か
* Softmaxとは何か
* Cross Entropy Lossとは何か
* Transformerの出力から分類がどのように行われるか

---

# 5. Phase 2：Questionを入力する

Phase 1では、

```text
文章 → 固定されたクラス
```

だった。

Phase 2では、

```text
State
Question
    ↓
BERT
    ↓
Decision
```

という形式にする。

---

## 例

```json
{
  "state": "商品が壊れていたので返品したい",
  "question": "問い合わせカテゴリは？"
}
```

↓

```text
返品
```

---

## 目的

「単なるIntent Classification」から、

> **質問に対するDecision**

へ近づける。

---

# 6. Phase 3：Dynamic Choice Decision Model

ここがMini-Jevの中心。

Phase 1では、

```text
クラス = 固定
```

だった。

Phase 3では、

```text
State
Question
Choices
```

を入力として与える。

---

## 入力例

```json
{
  "state": "商品が壊れていたので返品したい",
  "question": "問い合わせカテゴリは？",
  "choices": [
    "配送",
    "返品",
    "支払い",
    "商品不良"
  ]
}
```

---

## 出力

```json
{
  "answer": "返品",
  "probabilities": {
    "配送": 0.02,
    "返品": 0.91,
    "支払い": 0.01,
    "商品不良": 0.06
  },
  "confidence": 0.91
}
```

---

# 7. Dynamic Choiceのモデル構造

候補ごとにScoreを計算する。

```text
State + Question
       ↓
Context Encoder
       ↓
Context Representation
```

一方、

```text
Choice 1
Choice 2
Choice 3
...
       ↓
Choice Encoder
       ↓
Choice Representations
```

そして、

```text
Context Representation
        +
Choice Representation
        ↓
Decision Head
        ↓
Score
```

を各Choiceに対して実行する。

---

## 数式イメージ

```text
h_context = Encoder(State, Question)

h_choice_i = Encoder(Choice_i)

score_i = f(
    h_context,
    h_choice_i
)
```

最後に、

```text
P(choice_i)
=
Softmax(score_i)
```

とする。

---

# 8. Cross Encoder方式

最初のDynamic Choice実装では、より単純な方法として、

```text
[CLS]
State
[SEP]
Question
[SEP]
Choice
[SEP]
```

をChoiceごとにBERTへ入力する方法も使える。

例えば、

```text
State + Question + 「返品」
State + Question + 「配送」
State + Question + 「支払い」
State + Question + 「商品不良」
```

をそれぞれBERTに通す。

```text
             BERT
              ↓
          Score(返品)

             BERT
              ↓
          Score(配送)

             BERT
              ↓
          Score(支払い)

             BERT
              ↓
        Score(商品不良)
```

最後に4つのScoreをSoftmaxする。

### メリット

実装が比較的簡単。

### デメリット

Choice数に比例してEncoder計算量が増える。

---

# 9. Phase 4：Calibration

Phase 3までで、

```text
返品: 0.91
配送: 0.05
支払い: 0.02
商品不良: 0.02
```

のような確率が出るようになる。

しかし、

> **0.91という確率を本当に信頼してよいのか？**

という問題がある。

---

## 9.1 AccuracyとConfidenceは別

例えば、

```text
予測        Confidence
返品        0.95
配送        0.93
支払い      0.91
```

でも、実際には間違いが多い可能性がある。

つまり、

> Accuracyが高い ≠ ProbabilityがCalibrationされている

---

# 10. ECE

Calibration評価としてExpected Calibration Error（ECE）を測定する。

例えば、

```text
Confidence 0.8～0.9
```

の予測を集めたとき、

```text
平均Confidence = 0.85
実際のAccuracy = 0.84
```

ならCalibrationは良い。

一方、

```text
平均Confidence = 0.85
実際のAccuracy = 0.60
```

なら過信している。

---

# 11. Temperature Scaling

まずはRLを使わず、Temperature Scalingを試す。

```text
logits
   ↓
logits / T
   ↓
Softmax
   ↓
Calibrated Probability
```

Validation Datasetを使って最適なTemperature `T` を求める。

---

## 評価指標

Phase 4では、

* Accuracy
* ECE
* NLL
* Brier Score
* Reliability Diagram

などを見る。

特に、

```text
Accuracy
+
Calibration
```

をセットで評価する。

---

# 12. Phase 5：Agentへの組み込み

ここから実用的な方向へ発展させる。

現在のRAG / Agent開発との接続を考える。

---

## 例：RAGを実行するか判断

```text
User Query
    ↓
Mini-Jev
    ↓
「RAG検索が必要か？」
    ↓
YES / NO
```

例えば、

```text
YES: 0.97
NO : 0.03
```

ならRetrieverを実行。

---

## Agent Architecture

```text
                 User
                  ↓
              Mini-Jev
                  ↓
       ┌──────────┴──────────┐
       ↓                     ↓
   RAG必要               RAG不要
       ↓                     ↓
   Retriever                LLM
       ↓
   Reranker
       ↓
  十分な情報？
       ↓
      YES
       ↓
      LLM
```

---

# 13. ConfidenceをAgentの制御に使う

例えば、

```text
Confidence >= 0.95
        ↓
     Tool実行
```

```text
0.70 <= Confidence < 0.95
        ↓
    大きなLLMに確認
```

```text
Confidence < 0.70
        ↓
   Human / Escalation
```

のようなRoutingが考えられる。

ただし、閾値は実験によって決める。

---

# 14. Phase 6：RLCD-like Approach

最後に、Jevで説明されているRLCDの考え方を参考にする。

重要なのは、

> **単に正解率を上げるだけではなく、Probabilityが実際の成功確率を反映するようにする**

こと。

---

## 通常の分類

```text
正解を当てる
    ↓
Reward
```

---

## Calibrationを重視

```text
Prediction
+
Probability
+
Actual Outcome
        ↓
Calibrationを評価
        ↓
Model Update
```

---

# 15. ただしRLは最後にする

最初からRLを実装しない。

推奨順序：

```text
① Supervised Classification
        ↓
② Dynamic Choice
        ↓
③ Probability Calibration
        ↓
④ ECE / Brier / NLL評価
        ↓
⑤ Agent Routing
        ↓
⑥ RLCD-likeな学習方法
```

特にProbability Calibrationが目的なら、最初からRLを使う必要はない。

まずは、

* Cross Entropy
* NLL
* Brier Score
* Temperature Scaling

などの**確率予測として自然な方法**で性能を確認する。

---

# 16. 技術スタック

```text
Python
PyTorch
Hugging Face Transformers
Hugging Face Datasets
scikit-learn
evaluate
```

---

# 17. 最初のモデル

Phase 1では、日本語BERT系モデルを利用する。

候補：

```text
cl-tohoku/bert-base-japanese-v3
```

など。

ただし、実装時点でHugging Face上のモデル・Tokenizer・推奨利用方法を確認する。

---

# 18. プロジェクト構成

```text
mini_jev/
│
├── data/
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
│
├── model/
│   ├── encoder.py
│   ├── classification_head.py
│   └── decision_head.py
│
├── train.py
├── evaluate.py
├── calibrate.py
├── inference.py
│
└── README.md
```

---

# 19. データ

最初は小規模なデータセットでよい。

例えば、

```text
配送
返品
支払い
商品不良
```

の4カテゴリ。

例：

```json
{
  "text": "商品が壊れていたので返品したい",
  "label": "返品"
}
```

Phase 3では、

```json
{
  "state": "商品が壊れていたので返品したい",
  "question": "問い合わせカテゴリは？",
  "choices": [
    "配送",
    "返品",
    "支払い",
    "商品不良"
  ],
  "answer": "返品"
}
```

へ変更する。

---

# 20. 評価の考え方

単純なAccuracyだけでなく、

```text
Accuracy
Precision
Recall
F1
ECE
NLL
Brier Score
Latency
Memory Usage
```

を測定する。

特にMini-Jevでは、

```text
正解率
        +
確率の信頼性
        +
推論速度
```

の3つを重要視する。

---

# 21. 最終的に検証したい仮説

今回のプロジェクトでは、最終的に以下を検証する。

### 仮説1

```text
Transformerのhidden representation
        ↓
Decision Head
        ↓
Choice Probability
```

という構造で、自然言語による意思決定が可能か。

### 仮説2

Autoregressive Generationを使わなくても、Decision用途では十分な性能を出せるか。

### 仮説3

Dynamic Choiceにすると、

```text
固定クラス分類
```

より汎用的なDecision Modelにできるか。

### 仮説4

Calibrationを改善することで、

```text
Confidence
```

をAgentのRoutingに利用できるか。

### 仮説5

小型のDecision ModelをLLMの前段に置くことで、

```text
全てをLLMに判断させる
```

よりも、

```text
Small Decision Model
        ↓
必要なときだけ
        ↓
Large LLM
```

という構成が有効になるか。

---

# 22. 最終形のイメージ

```text
                     User
                       ↓
                ┌─────────────┐
                │  Mini-Jev   │
                │             │
                │ Transformer │
                │      ↓      │
                │ DecisionHead│
                └──────┬──────┘
                       ↓
              Choice Probabilities
                       ↓
          ┌────────────┼────────────┐
          ↓            ↓            ↓
       Tool実行      RAG検索       LLM
          ↓            ↓            ↓
       結果取得      Reranker     回答生成
                       ↓
                      LLM
                       ↓
                    Response
```

---

# 23. 開発順序

最初は以下だけを完成させる。

```text
STEP 1
Hugging FaceからBERTをロード
        ↓
STEP 2
BERTのhidden representationを確認
        ↓
STEP 3
Classification Headを追加
        ↓
STEP 4
4クラス分類を学習
        ↓
STEP 5
Accuracyを測定
        ↓
STEP 6
Probabilityを確認
        ↓
STEP 7
Dynamic Choiceへ変更
        ↓
STEP 8
Calibration
        ↓
STEP 9
Agent Routing
        ↓
STEP 10
RLCD-likeな手法を実験
```

---

# 24. このプロジェクトで理解するべき本質

最終的に理解したいのは、

> **LLM = 必ず文章を生成するモデル**

ではなく、

```text
Language Understanding
        ↓
Representation
        ↓
Task-specific Head
```

という考え方。

そして、

```text
Generation
```

を目的とするのではなく、

```text
Decision
```

を目的にモデルを設計すると、

```text
Latency
Cost
Output Control
Probability
Calibration
Agent Routing
```

などを別の観点から最適化できる可能性がある。

Mini-Jevでは、この考え方を**BERT分類 → Dynamic Decision → Calibration → Agent**という順番で実際にコードを書きながら検証する。
