# BOAZ 9주차 과제
## BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
### Abstract
- BART는 denoising autoencoder 기반 사전학습(sequence-to-sequence) 기법.

- 기존의 사전학습 모델들은 이해나 생성 등 특정 태스크에만 강점을 보임. → 범용성 부족.

<학습 과정>
1. 텍스트에 임의의 노이즈를 줌.
2. 원본 텍스트를 복원하도록 학습함.


- 구조는 Transformer 기반으로, Encoder는 BERT처럼 bidirectional, Decoder는 GPT처럼 autoregressive한 구조를 가짐.

**BERT+GPT의 일반화된 형태.**

<성능>
- GLUE, SQuAD 등 이해 과제에서 RoBERTa와 유사.
- Summarization, QA, Dialogue generation 등 생성 과제에서 큰 개선 (ROUGE +6).
- 번역(WMT RO-EN)에서 back-translation 대비 BLEU +1.
---
### Introduction
Self-supervised pretraining은 NLP 전반에서 큰 성과를 거둠. 하지만 기존 모델은 한쪽에 치우치는 문제가 생김.
1. BERT → 이해 과제에 강점, 생성에는 약점.
2. GPT → 생성 가능, 하지만 양방향 문맥 활용 불가.
![alt text](image/1.png)
```
BERT : Random token들이 mask로 대체됨. 문서는 Bidirectionally하게 인코딩 됨. 누락된 token들이 독립적으로 예측됨. → BERT는 generation에 쉽게 사용되지 못함.

GPT : Token들이 auto regressively하게 예측되므로 generation에 사용됨. 그러나 단어들은 왼쪽 context에만 의존. → Bidirectional interaction 학습하지 못함.
```
- BART는 두 방식을 결합 → **이해 + 생성** 모두 대응 가능. 다양한 노이즈를 입력에 적용 → 더 광범위한 end task에 적용 가능.
![alt text](image/2.png)


- 번역 fine-tuning 방식으로 BART위에 Transformer layer를 쌓아 외국어를 노이즈로 변환한 영어로 매핑하게 하여 사전학습이 된 타깃 언어 모델로 활용 가능함을 보임.
- abaltion study를 통해 기존의 사전학습 기법들을 BART의 framework 내에서 재현해 냄. → BART가 다양한 task에서 가장 좋은 성능을 보임.
---
### Model
#### Architecture
표준 Transformer seq2seq 아키텍처를 사용함. GeLUs를 활성화 함수로 사용함.
- Base: encoder 6층 + decoder 6층.
- Large: encoder 12층 + decoder 12층. → BERT와 유사.

<BERT와의 차이>
1. 디코더의 각 층은 인코더의 마지막 은닉층에 대해 cross attention을 수행함.
2. BART랑 다르게 BERT는 단어 예측 전에 추가적으로 feed foward neural network를 사용함.(10% 더 많은 parameter 사용)

#### Pre-training
autoencoding 모델들은 특정한 노이즈 방식에 맞춰 학습함. 하지만, BART는 훨씬 범용적인 방식을 사용함. 문서를 훼손한 뒤 원래 문서로 복원하는 과정을 통해 학습. Negative log-likelihood 최소화.하는 방식으로 최적화.

![alt text](image/3.png)
1. Token Masking\
BERT와 동일하게 입력에서 임의의 토큰을 골라 [MASK]로 바꿈.

2. Token Deletion\
랜덤 토큰을 아예 삭제함. Token Masking과 달리 모델이 어떤 위치가 빠졌는지까지 스스로 추론해야 함.

3. Text Infilling\
여러 개의 연속된 span을 골라 하나의 [MASK]로 대체함. 이 과정은 모델이 얼마나 많은 토큰이 빠졌는지까지 추론하도록 만듦.

4. Sentence Permutation
문서를 문장 단위로 나눈 뒤, 문서를 무작위로 섞음.

5. Document Rotation
문서에서 임의의 토큰 하나를 골라 그 토큰이 문서의 시작이 되도록 문서를 회전시킴. 이를 통해 모델이 문서의 올바른 시작 위치를 파악하도록 학습함.
---
### Fine-tuning
- Classification\
 encoder+decoder 입력 동일 → 최종 decoder hidden state → linear classifier.

- Token classification\
각 토큰의 decoder hidden state 사용.

- Generation\
 seq2seq 방식 그대로 fine-tune (요약, QA 등).

- Machine Translation\
새로운 encoder를 학습시켜 BART decoder와 결합.
---
### Experiments
#### small-scale (base, 1M steps)
- 순수 LM, Permuted LM, Masked LM, MASS, UniLM 등과 비교.
- 결과: Text infilling + sentence shuffling 조합이 가장 안정적.

#### Large-scale (RoBERTa 수준)
- 데이터: 160GB 뉴스, 책, 웹 텍스트.
- Batch size 8000, step 500k.
- Objective: text infilling + sentence permutation.
- Discriminative tasks: GLUE, SQuAD → RoBERTa와 비슷.
![alt text](image/4.png)

- Generation tasks:
1. Summarization: CNN/DM, XSum → ROUGE +6.
2. Dialogue (ConvAI2): PPL 11.8.
3. Abstractive QA (ELI5): ROUGE-L +1.2.
4. Translation (WMT16 RO-EN): BLEU +1.1.
![alt text](image/5.png)
---
### Analysis
- 단순 문장 섞기/회전만으로는 효과 미약 → masking/infilling이 핵심.

- Bidirectional encoder는 문맥 이해가 중요한 과제(SQuAD 등)에 필요.

- 생성 자유도가 큰 QA(ELI5)에서는 순수 LM이 강세.

- 요약 결과: 추상화 수준 높고 문법적 유창성 향상.
---
### Conclusion

- BART = BERT + GPT 통합형 프레임워크.

- 이해와 생성 task 모두에서 강력한 성능.

- RoBERTa 수준 리소스로 학습 시, 이해 성능 유지 + 생성 성능 대폭 향상.

- 앞으로는 특정 task 맞춤형 노이즈 설계가 새로운 연구 방향이 될 것임.


## An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling

### Abstract

* 순환 신경망(RNN)은 전통적으로 순차적 데이터(sequence)를 다루는 데 가장 널리 쓰임.
* 하지만 RNN은 학습/추론 속도가 느리고 병렬화가 어렵다는 단점이 있음.
* 본 논문에서는 RNN, CNN, self-attention 기반 모델들을 동일 조건에서 비교 평가함.
* **결론**: CNN 계열 모델이 RNN보다 더 빠르고 안정적이며, 여러 sequence modeling task에서 동등하거나 더 나은 성능을 냄.

---

### Introduction

* 자연어 처리, 음성 인식 등 sequence task는 전통적으로 LSTM, GRU 같은 RNN 기반 모델이 주도.
* 문제: RNN은 긴 의존성(long-term dependency)을 다루기 어렵고, 병렬화 불리.
* 대안으로 CNN(Convolutional sequence model)과 self-attention이 제시됨.
* 연구 목표: **RNN, CNN, Transformer 계열을 다양한 task에 대해 공정하게 비교**.

---

### Temporal Convolutional Networks (TCN)
#### Sequence Modeling
- Sequence Modeling에서는 길이 $T$의 입력 시퀀스 $(x_0, ..., x_T)$를 받아 동일한 길이의 출력 시퀀스로 매핑함.
- 이때 각 시점의 출력은 오직 과거 입력에만 의존해야 하며, 미래 정보는 사용할 수 없음.
- 따라서 모델은 주어진 입력과 실제 출력의 차이를 최소화하는 손실 함수를 통해 학습됨.

#### Causal Convolutions
- TCN은 인과적 합성곱(causal convolution)을 기반으로 함.
- 입력과 같은 길이의 출력을 생성하며, 미래의 정보가 과거로 흘러가지 않도록 보장해야 함.
- 구현은 1D fully-convolutional network (FCN) 구조를 사용하며, zero padding을 추가해 출력 길이가 입력과 동일하게 유지됨.
- 특정 시간 $t$의 출력이 현재 및 과거 시점의 입력과만 합성곱되도록 제한됨.

#### Dilated Convolutions
![alt text](image/6.png)
- RNN은 시간 순서대로 한 단계씩 정보가 전달되므로 과거 정보를 장기적으로 유지하기 어려움.
- LSTM/GRU는 단기 기억 소실 문제를 어느 정도 개선했지만, 긴 시퀀스를 다룰 때는 여전히 한계가 존재함.
- 논문에서는 이 단기 기억 소실 문제를 해결하기 위해 **확장 합성곱(Dilated Convolutions)**을 제안함.
- Dilated convolution은 필터 적용 시 입력을 일정 간격(dilation factor)으로 건너뛰며 연결하는 합성곱 방식임. $k$: 필터 크기, $d$: dilation factor임. dilation을 1, 2, 4로 증가시키면 receptive field가 지수적으로 커져 긴 과거 정보까지 효율적으로 반영할 수 있음. 따라서 TCN은 작은 깊이와 파라미터로도 RNN보다 더 긴 의존성을 처리할 수 있음.

#### Residual Connections
![alt text](image/7.png)
- TCN이 충분히 깊어지면, 기울기 소실(Vanishing Gradient) 문제가 발생할 수 있음.
- 즉, 네트워크가 깊어질수록 초기 층이 제대로 학습되지 않음.- TCN 모델에 Residual Connection을 적용함.
- ResNet과 유사한 잔차 연결(Residual Connections)을 추가해 기울기 소실 문제를 방지함.
- 기존 출력값에 입력값을 직접 더하는 구조(Identity Mapping)를 사용해 안정적인 학습이 가능함.

---

### Experiments

* **평가 task**:

  1. Language modeling (WikiText-103, PTB 등)
  2. Character-level modeling (text8, enwik8)
  3. Sequential MNIST
  4. Permuted MNIST
  5. Polyphonic music modeling

* **설정**: 모든 모델을 동일한 최적화 조건에서 학습해 성능 비교.

<주요 결과>

* **Language modeling**:

  * LSTM은 여전히 강력하지만, CNN 기반 TCN도 비슷한 성능을 보임.
* **Character-level modeling**: TCN이 RNN보다 더 우수한 성능.
* **Sequential MNIST / Permuted MNIST**: CNN 계열이 RNN 대비 빠르고 안정적.
* **Polyphonic music modeling**: CNN이 RNN과 유사하거나 더 나은 성능.

---

### Conclusion

* CNN 기반 sequence 모델(TCN 등)은 RNN의 대안으로 매우 유망함.
* RNN은 긴 의존성 학습에 한계가 있고 병렬화도 불리하지만, CNN은 병렬화에 강점이 있음.
* Self-attention은 효율성이 떨어지지만 특정 과제에서 잠재적 가능성을 보임.
* **종합적으로 CNN은 RNN을 대체할 수 있는 강력한 후보**임을 실험적으로 입증함.

