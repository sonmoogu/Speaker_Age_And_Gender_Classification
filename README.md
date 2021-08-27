# Speaker_Age_And_Gender_Classification

## wavPreprocess

* Saving_Mel.m : 로컬 경로내 wav파일 -> Mel_Spectogram 이미지로 모두 변환
  * AutoScaled 된 Mel_Spec 이미지 샘플 **[노인남녀]_test(명령어음성)**
  ![mel_old_test_1](https://user-images.githubusercontent.com/73811196/130889131-b04b202e-701e-4e4a-8a1e-4e0dcf8a082f.png)
  
  Test데이터의 Mel-spectrogram 변환 과정에서는 Train, Validation(16kHz)와 같이 사용하기 위하여 
  
  48kHz --> 16kHz의 Downsampling 과정을 추가하였다.
  
-----

## VGG16 model transfer learning

 ![vggmodel](https://user-images.githubusercontent.com/74817754/130889369-157cee32-738e-4674-92de-90f68ce58865.jpg) 
 
 Ref) https://neurohive.io/en/popular-networks/vgg16/
 
 다음과 같은 vgg모델을 사용한다.
 
마지막 출력층의 경우 maxpooling, activation(relu) , dropout, activation(softmax)를 거쳐 생성한다.

데이터 셋의 경우 앞서와 같이 wav 파일을 mel-spectrogram으로 변환하는 wavPreprocess를 사용한다.

          Dataset(wav) : Train, Validation : AIHUB 자유대화 [노인, 일반남녀, 어린이] : 16kHz

                         Test : AIHUB 명령어 [노인, 일반남녀, 어린이] : 48kHz(사용 시 downsampling 필요)
          
Ref) https://aihub.or.kr/aidata/33305
