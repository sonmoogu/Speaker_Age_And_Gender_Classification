# Speaker_Age_And_Gender_Classification

## wavPreprocess

* Saving_Mel.m : 로컬 경로내 wav파일 -> Mel_Spectogram 이미지로 모두 변환
  * AutoScaled 된 Mel_Spec 이미지 샘플 **[노인남녀]_test(명령어음성)**
  ![mel_old_test_1](https://user-images.githubusercontent.com/73811196/130889131-b04b202e-701e-4e4a-8a1e-4e0dcf8a082f.png)
  
-----

## VGG16 Pre-trained model transfer learning

 ![vggmodel](https://user-images.githubusercontent.com/74817754/130889369-157cee32-738e-4674-92de-90f68ce58865.jpg) 
 
 Ref) https://neurohive.io/en/popular-networks/vgg16/
 
 다음과 같은 vgg모델을 사용한다.
 
마지막 출력층의 경우 maxpooling, activation(relu) , dropout, activation(softmax)를 거쳐 생성한다.

데이터 셋의 경우 앞서와 같이 wav 파일을 mel-spectrogram으로 변환하는 wavPreprocess를 사용한다.

          Dataset : Train, Validation : AIHUB 자유대화 [노인, 일반남녀, 어린이]

          Test : AIHUB 명령어 [노인, 일반남녀, 어린이]
          
Ref) https://aihub.or.kr/aidata/33305



-----


## 동영상 강의 묶음, 재생목록 (Video Lectures)

Video 강좌는 제가 개인적으로 생각하는 순차적 학습 단계 입니다. 물론, 난이도와도 연관이 있습니다. 

**파이썬 (Python), 데이터분석 (Pandas, Numpy), 시각화 (Matplotlib, Seaborn, Bokeh, Folium)**

* [생애 첫 코딩 - 파이썬 (김정욱)](https://learnaday.kr/open-course/geNpyx)
  * 코딩 학원을 운영하고 있는 김정욱 대표의 파이썬 입문 강좌 (3시간). 라이트 과정은 무료로 제공하고 있습니다.
* [파이썬 강좌 코딩 기초 강의 Python | 김왼손의 왼손코딩](https://www.youtube.com/watch?v=c2mpe9Xcp0I&list=PLGPF8gvWLYyrkF85itdBHaOLSVbtdzBww&index=1)
* [딥러닝을 위한 파이썬 - 신경식님](https://learnaday.kr/open-course/ZiYShf)
* [NumPy(넘파이) 기본 - T아카데미](https://www.youtube.com/watch?v=zNrDbG4tNGo&list=PL9mhQYIlKEhf04ToiDFvNzKL0OP4W27TW)
* [Pandas 기본기 다지기 - T아카데미](https://www.youtube.com/watch?v=M_lKmt-wSvY&list=PL9mhQYIlKEhfG_gWF-DclKs6vXS6SkmQN)
* [Pandas로 하는 시계열 데이터분석 - T아카데미](https://www.youtube.com/watch?v=oNLaw2Q8Irw&list=PL9mhQYIlKEhd60Qq4r2yC7xYKIhs97FfC)
* [입문자를 위한 파이썬 기초 따라잡기 - 재즐보프](https://www.youtube.com/watch?v=BvJhYPQSDLI&list=PLnIaYcDMsScyhT18mwY71rV_aHdP-OhLd)
* [파이썬 데이터 시각화 튜토리얼 - 재즐보프](https://www.youtube.com/watch?v=TIjsrH_THhs&list=PLnIaYcDMsScyrZZXH6LTXMrOLXJ-7hznD)

