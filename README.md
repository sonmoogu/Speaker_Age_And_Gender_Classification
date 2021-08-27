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
