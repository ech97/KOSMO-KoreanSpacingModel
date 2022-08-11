# KOSMO: KOrean Spacing MOdel

[제작시기]

21.09.01 -

---

[Reference]

논문링크: [Fast and Accurate Entity Recognition with Iterated Dilated](https://arxiv.org/abs/1702.02098)

	- 2017.02 (Arxiv)
	- Emma Strubell, Patrick Verga, David Belanger, Andrew AcCallum

blog: https://blog.ukjae.io/posts/korean-spacing-model/

---

## 용어정리

### 1. MaxPool1D

> 시계열 데이터에서 사용하는 Pooling 방식

![image-20220803153638056](https://raw.githubusercontent.com/ech97/save-image-repo/master/image/image-20220803153638056.png)

### 2. 2D-Dilated(Atrous) Convolution

> - 작품에서 사용하는 1D Dilated Convolution을 설명하기 전에 이해를 돕기위해 2D Dilated Convolution 소개
>
> - **기존 3x3의 Kernel 사이사이에 0을 주입하여 5x5 Kernel로 변형하여 더 넓은 Receptive Field를 가짐**

![img](https://raw.githubusercontent.com/ech97/save-image-repo/master/image/img.gif)

- 기존의 Kernel / Atrous Convolution의 Kernel

  - $\left[\begin{matrix}1 & 3 & 5 \\ 2 & 4 & 6 \\ 3 & 5 & 7 \end {matrix} \right]$ / $\left[\begin{matrix}1 & 0 & 3 & 0 & 5 \\ 0 & 0 & 0 & 0 & 0 \\ 2 & 0 & 4 & 0 & 6 \\ 0 & 0 & 0 & 0 & 0 \\ 3 & 0 & 5 & 0 & 7 \end {matrix} \right]$

  - **다음과 같이 3x3 → 5x5 Kernel로 변경됨** = 더 넓은 Receptive field



---

## 개요

### 1. 개발동기

- 2021년 임베디드 소프트웨어 경진대회에, 청각장애인(전농인)의 배달 전화 문의를 원활하게 하기 위한 **실시간 음성인식 및 합성 기능을 채팅형태로 변환**하는 **어플리케이션**을 개발을 목표로 함

- 이를 위해 아래의 몇가지 기술이 필요

  > 실시간성이 중요하기 때문에, 모델들의 추론 속도 향상이 최우선 목표

  - 다감정 음성합성기

    > **삼성전자 서비스센터**에 CS 직무로 근무해본 바, 상담 진행시에 감정이 섞인 발화는 문제 해결 속도에 **분명한** 영향을 미치기때문에,
    >
    > 전농인 또한 **화났거나, 슬픈 감정**이 있는 TTS를 사용하여, 비장애인과 동일한 문제해결속도를 보장하고싶었음

    - Text2mel: Tacotron2에 GST model을 결합하여 Alignment정보와 Style Token을 학습시킨 뒤, Fastspeech에 전이학습
    - Mel2wav: 빠른 속도의 Parallel Wave GAN을 통해 Melspectrogram을 Wave로 변환

  - 감성분석 모델

    - SKT Brain에서 사전학습 및 제작한 Korean BERT 모델 위에 추가적인 Layer를 쌓아,
    - 한국어 문장 감성을 통해 감성에 맞는 음성합성을 진행

  - 음성인식기

    - 정확도가 높은 (CER < 5%)의 한국어 음성인식 모델(Vec2Wav2.0)은 실시간성이 없음
    - 실시간 성이 보장되는 모델(RNN-T)는 음성인식률이 좋지 못함(WER > 30%)
    - 따라서 실시간성과, 정확성이 보장된 KALDI Toolkit을 활용

  - **띄어쓰기 모델**

    - **Kaldi의 음성인식 결과가 형태소 단위로 나오기때문에 띄어쓰기를 교정하여 가독성을 높여야 함**

----

### 2. Related research

#### 2-1. 문제상황

- 대부분의 한국어 띄어쓰기 라이브러리는 띄어쓰기 **삭제**가 아닌, 띄어쓰기 **삽입**에만 집중하고 있었음
  - [soyspacing](https://github.com/lovit/soyspacing)의 경우 띄어쓰기를 시행할지 말지 Binary Classification으로 해결
- RNN모델을 사용하여 추론시간이 긴 경우가 많았음
  - [KoSpacing](https://github.com/haven-jeon/KoSpacing)의 경우 CNN-FFN-**GRU**-FFN의 구조
  - [Chatspace]()의 경우 CNN-FFN-**BiLSTM**-FFN의 구조

#### 2-2. 해결방안

> [Fast and Accurate Entity Recognition with Iterated Dilated](https://arxiv.org/abs/1702.02098) 논문을 참고하여 LSTM을 Atrous(Dilated) Convolution으로 대체

- Sequantial Model (GRU, LSTM)의 경우에는 병렬처리에 매우 비효율적이며, Training/Inference Time이 좋지 못함
- CNN 방식으로 병렬처리의 단점을 보완할 수 있지만, 좁은 Receptive field([용어설명은 링크의 용어설명 파트 참고](https://ech97.tistory.com/entry/mobilenetv2))를 가짐
- 이때의 단점을 극복하기 위해, Dilated Convolution 방식을 고안
  - 이를 통해 Bi-LSTM-CRF구조에 비해 14~20배의 속도 향상 가능

##### 1D Dilated Convolution

> NLP에선 1D Convolution의 Kernel size를 통해 한번에 보는 단어의 개수를 조절 (n-gram과 유사)

![image-20220803152334119](https://raw.githubusercontent.com/ech97/save-image-repo/master/image/image-20220803152334119.png)

![image-20220803152619803](https://raw.githubusercontent.com/ech97/save-image-repo/master/image/image-20220803152619803.png)

- 기존의 CNN보다 Dilated CNN이 같은 Layer수로 더 많은 Token 관찰 가능

- 하지만 단점도 존재
  - Overfitting이 쉽게 일어나므로, Layer를 깊게 쌓았을때 학습이 더 진행이 되지 않는 경우 발생
    - 이를 해결하기 위해 3개의 Dilated Layer를 사용하여 Block으로 만든 뒤, 해당 Block을 계속 재사용
    - 또한 **Dropout** with expectation-inear regularization 방식 사용

---

## 설명

### 1. 목표

- 딥러닝을 이용하여 **빠른속도**의 띄어쓰기의 **추가** 및 **삭제** 기능 구현
  - 빠른속도: 1 문장 내의 띄어쓰기 추론 시간 < 1ms
  - 띄어쓰기: 없는 띄어쓰기 **추가**, 과한 띄어쓰기 **삭제**

### 2. 모델

#### 2-1. 학습 방법

- Multi Classifier: 3가지 상태로 구분
  - 0: 변경 없음
  - 1: 현재 문자 뒤 공백 문자 추가
  - 2: 현재 공백 문자 삭제
- 이전 띄어쓰기 라이브러리의 Squential Model을 Atrous Convolution 으로 교체

- **띄어쓰기가 잘 된 '원래 문장'**과 임의로 띄어쓰기 추가 및 삭제를 진행한 '변형 문장'으로 Self Supervised Learning
- 또한 BOS, EOS태그를 붙여서 문장 구별
  - ```<s>```, ```</s>```

| 구분        | 내용                                                         |
| ----------- | ------------------------------------------------------------ |
| 원래 문장   | "안녕하세요 저는 멋짐보단 멈춘 사자처럼 입니다"              |
| 학습할 문장 | \<s>안 녕하세요저는멋 짐보단멈춘 사자처 럼입니다</s\>        |
| 학습 레이블 | [0, 0, 2, 0, 0, 1, 0, 1, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0] |

#### 2-2 학습 데이터

- 실제 사용시에는 프로젝트가 사용 될 도메인에 맞는 말뭉치를 이용하여 학습
  - AIHub의 배달, 음식점, 일상대화 말뭉치 / 2000 문장
- 하지만, 테스트 용으로는 [나무위키텍스트](https://github.com/lovit/namuwikitext)를 활용하여 다양한 조건에 대해 Accuracy를 측정할 예정

---

## 구현

