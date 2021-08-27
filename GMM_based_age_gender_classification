
import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
from python_speech_features import mfcc
from python_speech_features import delta
import parselmouth
import pandas as pd
from featureextractor_other import featureextractor

import os
class FeaturesExtractor:
    def __init__(self):
        pass
       
    def extract_features(self, audio_path):
        """
        Extract voice features including the Mel Frequency Cepstral Coefficient (MFCC)
        from an audio using the python_speech_features module, performs Cepstral Mean
        Normalization (CMS) and combine it with MFCC deltas and the MFCC double
        deltas.
     
        Args: 	    
            audio_path (str) : path to wave file without silent moments. 
        Returns: 	    
            (array) : Extracted features matrix. 	
        """
        '''
        data=pd.DataFrame(["Fo(pitch)"])
        Sound=parselmouth.Sound(audio_path)
        pitch=Sound.to_pitch()
        formant = Sound.to_formant_burg(time_step = 2)
        df = pd.DataFrame({"times":formant.ts()})
        df['F0(pitch)'] = df['times'].map(lambda x: pitch.get_value_at_time(time = x))

        print(df['F0(pitch)'])
        if int in df['F0(pitch)']:
            pitch_value=df['F0(pitch)'].max()
        else:
            pitch_value=0
        '''
        rate, audio  = read(audio_path)
        mfcc_feature = mfcc(# The audio signal from which to compute features.
                            audio,
                            # The samplerate of the signal we are working with.
                            rate,
                            # The length of the analysis window in seconds. 
                            # Default is 0.025s (25 milliseconds)
                            winlen       = 0.05,
                            # The step between successive windows in seconds. 
                            # Default is 0.01s (10 milliseconds)
                            winstep      = 0.01,
                            # The number of cepstrum to return. 
                            # Default 13.
                            numcep       = 5,
                            # The number of filters in the filterbank.
                            # Default is 26.
                            nfilt        = 30,
                            # The FFT size. Default is 512.
                            nfft         = 2400,
                            # If true, the zeroth cepstral coefficient is replaced 
                            # with the log of the total frame energy.
                            appendEnergy = True)
        newpath=os.path.join("C:\\Users\\user\\SpeakerRecognition_tutorial_",audio_path)
        features=featureextractor.getfeautre(newpath)
        print(features)
        new_features=features.reshape(-1,1)
        print(new_features)
        
        '''print(features)'''
        mfcc_feature  = preprocessing.scale(mfcc_feature)
        
        deltas        = delta(mfcc_feature, 2)
        double_deltas = delta(deltas, 2)
        combined      = np.hstack((mfcc_feature, deltas, double_deltas))
        '''combined_=np.concatenate((features,combined),axis=1)'''
        '''combined_=np.vstack(features,combined)'''
        return new_features
