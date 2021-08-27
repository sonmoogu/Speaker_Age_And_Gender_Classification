# Speaker_Age_And_Gender_Classification

## wavPreprocess

* Saving_Mel.m : 로컬 경로내 wav파일 -> Mel_Spectogram 이미지로 모두 변환
  * AutoScaled 된 Mel_Spec 이미지 샘플 **[노인남녀]_test(명령어음성)**
  ![mel_old_test_1](https://user-images.githubusercontent.com/73811196/130889131-b04b202e-701e-4e4a-8a1e-4e0dcf8a082f.png)
  
  Test데이터의 Mel-spectrogram 변환 과정에서는 Train, Validation(16kHz)와 같이 사용하기 위하여 
  
  48kHz --> 16kHz의 Downsampling 과정을 추가하였다.
-----
  
## CNN2D_Manual 

* CNN2D_Age_Classification_rmsprop_idg_6000.ipynb
  * STEP 1 ~ STEP 9 walkthrough 형식으로 컴파일시 모델 자동 저장
     
     [사용 데이터: AI HUB](https://aihub.or.kr/aihub-data/natural-language/about)
     
     
       Dataset(wav) : Train, Validation : AIHUB 자유대화 (노인, 일반남녀, 어린이) : 16kHz

                      Test : AIHUB 명령어 (노인, 일반남녀, 어린이) : 48kHz(사용 시 downsampling 필요) 

## MobileNet_V3 

* A.	Age_Classifier_Using_Fine_tuned_TFHuB_Tranfer_Learning.ipynb
  * STEP 0 ~ STEP 7 walkthrough 형식으로 컴파일시 모델 자동 저장
  * [모델 vector source](https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5)
  
-----

## VGG16 model transfer learning

 ![vggmodel](https://user-images.githubusercontent.com/74817754/130889369-157cee32-738e-4674-92de-90f68ce58865.jpg) 
 
 Ref) https://neurohive.io/en/popular-networks/vgg16/
 
 다음과 같은 vgg모델을 사용한다.
 
마지막 출력층의 경우 maxpooling, activation(relu) , dropout, activation(softmax)를 거쳐 생성한다.

데이터 셋의 경우 앞서와 같이 wav 파일을 mel-spectrogram으로 변환하는 wavPreprocess를 사용한다.

          [사용 데이터: AI HUB] (https://aihub.or.kr/aihub-data/natural-language/about)
          Dataset(wav) : Train, Validation : AIHUB 자유대화 [노인, 일반남녀, 어린이] : 16kHz

                         Test : AIHUB 명령어 [노인, 일반남녀, 어린이] : 48kHz(사용 시 downsampling 필요)
          
Ref) https://aihub.or.kr/aidata/33305

## GMM based age, gender recognition

![image 1](https://user-images.githubusercontent.com/73654014/131099698-6f5a9fb1-d667-4bbc-af06-e1554c8a5b3c.png)

노인 남녀, 일반 남녀, 아동 -> 5가지 클레스 분류
활용한 소스 코드 
=> Ref) https://github.com/SuperKogito/Voice-based-gender-recognition

파일구성 :

-ModelsTrainer.py : 분류를 하기 위해서 GMM모델을 생성하는 코드로써. ModelsTrainer 메인 함수 선언시 TrainingData폴더내에 분류할 파일명을 만들고 그안에 .wav파일 데이터를 저장한다.
예) dir => '''/GMM_based_age_gender_recognition/TrainingData/Childrens/(...).wav wav파일들의 디렉토리는 이렇게 된다.

-GenderIdentifier_other.py : GMM모델을 읽어오고 Testdata 폴더 에서 음성 파일을 하나하나의 특성을 추출하고 읽어온 GMM 모델과의 점수를 계산하여 출력한다. 이에 클래스 중 가장 큰 점수를 갖고 있는 값이 예상값이 되는것이다.
입력으로 받는 dir 목록 : 
1. TestingData : (실제로 Test할 데이터이고 TestingData폴더 안에 분류할 라벨의 폴더를 만들어서 그안에 데이터를 넣으면 된다.) 예) '''/GMM_based_age_gender_recognition/TestingData/Childrens/(...).wav -> 이안에 Children데이터만 넣으며. 기대값이 Children이 된다.
2. GMM(.gmm파일) : ModelsTrainer.py에서 생성한 GMM파일로써 분류할 클레스 만큼 GMM파일이 있으며 그것의 저장소를 다 값으로 전달하여 Testdata에서의 스코어를 계산할때 사용한다.

-Silenceremove_try.py : ModelsTrainer.py에서 음성 특성인 (Praat 특성값)을 추출하기전 전처리 과정으로써 묵음제거를 합니다.

-featurextractor_other.py : Praat의 파이썬 내부 라이브러리 Parselmouth를 이용하여 추출한 음성 특성값으로(mean F0 Hz, Shimmer,jitter 등)테스트할 때나 트레인할때 사용한다.(Praat 파라미터 추출)
-FeaturesExtractor_other.py : Praat로 추출한 음성 특성을 전달한다.

-FeaturesExtractor.py : 이것도 역시 트레인할때나 테스트 할때 음성 특성을 추출하는것인데 MFCC,delta MFCC, delta-delta MFCC 이렇게를 합친 값을 생성하는 코드이다(MFCC 파라미터 추출)

-featueextrctor_for_old : 노인, 일반 만을 구분하기 위해 그 둘사이에서 차이나는 음성 특성을 선택하여 뽑는 코드이다.

**!!!!중요한 것이 MFCC로 Train -> MFCC로 생성한 GMM모델로 Test  ,  Praat로 Train -> Praat로 생성한 GMM 모델로 Test))**
