## VGG16 Pre-trained model transfer learning

 ![vggmodel](https://user-images.githubusercontent.com/74817754/130889369-157cee32-738e-4674-92de-90f68ce58865.jpg) 
 
 Rf) https://neurohive.io/en/popular-networks/vgg16/
 
 다음과 같은 vgg모델을 사용한다.
 
마지막 출력층의 경우 maxpooling, activation(relu) , dropout, activation(softmax)를 거쳐 생성한다.

데이터 셋의 경우 앞서와 같이 wav 파일을 mel-spectrogram으로 변환하여 사용한다. 



