import os
import pickle
import warnings
import numpy as np
from sklearn.mixture import GaussianMixture
from FeaturesExtractor_other import FeaturesExtractor
from FeaturesExtractor import FeaturesExtractor_
from silenceremove_try import silence_removal
import wave
warnings.filterwarnings("ignore")


class ModelsTrainer:

    def __init__(self, females_files_path, males_files_path,olds_files_path,nonolds_files_path,childrens_files_path,olds_male_path,olds_female_path):
        self.females_training_path = females_files_path
        self.males_training_path   = males_files_path
        self.old_training_path=olds_files_path
        self.noolds_training_path=nonolds_files_path
        self.childrens_training_path=childrens_files_path
        self.olds_female_training_path=olds_female_path
        self.olds_male_training_path=olds_male_path
        self.features_extractor    = FeaturesExtractor()
        self.features_extractor_=FeaturesExtractor_()


    def oldageprocess(self):
        oldsfemale,oldsmale,nonoldsfemale,nonoldsmale,childrens=self.get_file_paths__(self.olds_female_training_path,self.olds_male_training_path,self.females_training_path,self.males_training_path,self.childrens_training_path)
        mean_olds_female=0
        mean_olds_male=0
        mean_nonolds_female=0
        mean_nonolds_male=0
        mean_childrens=0
        
        
        oldsfemale_voice_features,old_female_num,old_female_HNR,old_female_stdev,old_female_shimmer,old  = self.collect_features(oldsfemale)
        oldsfemales_gmm = GaussianMixture(n_components = 80, max_iter = 200, covariance_type='diag', n_init = 3)
        oldsfemales_gmm.fit(oldsfemale_voice_features)
        self.save_gmm(oldsfemales_gmm, "oldfemales_praat_")
        
        
       
        
        print("old(female) 의 HNR평균값 : "+str(float(old_female_HNR)/float(old_female_num))+"\n")
        print("old(female) 의 Stdev평균값 : "+str(float(old_female_stdev)/float(old_female_num))+"\n")
        print("old(female) 의 Shimmer평균값 : "+str(float(old_female_shimmer)/float(old_female_num))+"\n")

        
        oldsmale_voice_features,mean_olds_male,old_male_f0_max,old_male_num,old_male_HNR,old_male_stdev,old_male_shimmer=self.collect_features_(oldsmale)
        oldsmales_gmm   = GaussianMixture(n_components = 80, max_iter = 200, covariance_type='diag', n_init = 3)
        oldsmales_gmm.fit(oldsmale_voice_features)
        self.save_gmm(oldsmales_gmm,   "oldmales_praat_")
        
        
        print("old(male) 의 Fo평균값 : "+str(float(mean_olds_male)/float(old_male_num))+"\n")
        print("old(male) 의 Fo최대값 : "+str(old_male_f0_max)+"\n")
        print("old(male) 의 HNR평균값 : "+str(float(old_male_HNR)/float(old_male_num))+"\n")
        print("old(male) 의 Stdev평균값 : "+str(float(old_male_stdev)/float(old_male_num))+"\n")
        print("old(male) 의 Shimmer평균값 : "+str(float(old_male_shimmer)/float(old_male_num))+"\n")
        
        
        


        
        nonoldsfemale_voice_features=self.collect_features_(nonoldsfemale)
        nonoldfemales_gmm   = GaussianMixture(n_components = 80, max_iter = 200, covariance_type='diag', n_init = 3)
        nonoldfemales_gmm.fit(nonoldsfemale_voice_features)
        self.save_gmm(nonoldfemales_gmm,   "nonoldfemales_mfcc_re")
        
        
        
        nonoldsmale_voice_features=self.collect_features_(nonoldsmale)
        nonoldmales_gmm   = GaussianMixture(n_components = 80, max_iter = 200, covariance_type='diag', n_init = 3)
        nonoldmales_gmm.fit(nonoldsmale_voice_features)
        self.save_gmm(nonoldmales_gmm,   "nonoldmales_mfcc_re")

        children_voice_feautres=self.collect_features_(childrens)
        children_gmm = GaussianMixture(n_components = 80, max_iter = 200, covariance_type='diag', n_init = 3)
        children_gmm.fit(children_voice_feautres)
        self.save_gmm(children_gmm,"childrens_mfcc_")
        
    
        '''

        print("nonold(female) 의 Fo평균값 : "+str(float(mean_nonolds_female)/float(nonold_female_num))+"\n")
        print("nonold(female) 의 Fo최대값 : "+str(nonold_female_f0_max)+"\n")
        print("nonold(female) 의 HNR평균값 : "+str(float(nonold_female_HNR)/float(nonold_female_num))+"\n")
        print("nonold(female) 의 Stdev평균값 : "+str(float(nonold_female_stdev)/float(nonold_female_num))+"\n")
        print("nonold(female) 의 Shimmer평균값 : "+str(float(nonold_female_shimmer)/float(nonold_female_num))+"\n")

        print("nonold(male) 의 Fo평균값 : "+str(float(mean_nonolds_male)/float(nonold_male_num))+"\n")
        print("nonold(male) 의 Fo최대값 : "+str(nonold_male_f0_max)+"\n")
        print("nonold(male) 의 HNR평균값 : "+str(float(nonold_male_HNR)/float(nonold_male_num))+"\n")
        print("nonold(male) 의 Stdev평균값 : "+str(float(nonold_male_stdev)/float(nonold_male_num))+"\n")
        print("nonold(male) 의 Shimmer평균값 : "+str(float(nonold_male_shimmer)/float(nonold_male_num))+"\n")
        '''
        '''
        print("children의 Fo평균값 : "+str(float(mean_childrens)/float(childrens_num))+"\n")
        print("children의 Fo최대값 : "+str(childrens_f0_max)+"\n")
        print("children의 HNR평균값 : "+str(float(childrens_HNR)/float(childrens_num))+"\n")
        # generate gaussian mixture models
        
        oldsfemales_gmm = GaussianMixture(n_components = 64, max_iter = 200, covariance_type='diag', n_init = 3)
        oldsmales_gmm   = GaussianMixture(n_components = 64, max_iter = 200, covariance_type='diag', n_init = 3)
        nonoldsfemale_gmm=GaussianMixture(n_components = 64, max_iter = 200, covariance_type='diag', n_init = 3)
        nonoldsmale_gmm=GaussianMixture(n_components = 64, max_iter = 200, covariance_type='diag', n_init = 3)
        childrens_gmm=GaussianMixture(n_components = 64, max_iter = 200, covariance_type='diag', n_init = 3)

        oldsfemales_gmm.fit(oldsfemale_voice_features)
        oldsmales_gmm.fit(oldsmale_voice_features)
        nonoldsfemale_gmm.fit(nonoldsfemale_voice_feautures)
        nonoldsmale_gmm.fit(nonoldsmale_voice_feautures)
        childrens_gmm.fit(childrens_voice_feautres)
        # fit features to models
        # save models
        self.save_gmm(oldsfemales_gmm, "oldfemales")
        self.save_gmm(oldsmales_gmm,   "oldmales")
        self.save_gmm(nonoldsfemale_gmm,"nonoldfemale")
        self.save_gmm(nonoldsmale_gmm,"nonoldsmale")
        self.save_gmm(childrens_gmm,"childrens")

        print("old(female) 의 Fo평균값 : "+str(float(mean_olds_female)/float(olds_female_num))+"\n")
        print("old(female) 의 Fo최대값 : "+str(olds_female_f0_max)+"\n")
        print("old(male) 의 Fo평균값 : "+str(float(mean_olds_male)/float(olds_male_num))+"\n")
        print("old(male) 의 Fo최대값 : "+str(olds_male_f0_max)+"\n")
        print("nonold(female)의 F0평균값 : "+str(float(mean_nonolds_female)/float(nonolds_female_num))+"\n")
        print("nonold(female)의 F0최대값 : "+str(nonolds_female_f0_max)+"\n")
        print("nonold(male)의 F0평균값 : "+str(float(mean_nonolds_male)/float(nonolds_male_num))+"\n")
        print("nonold(male)의 F0최대값 : "+str(nonolds_male_f0_max)+"\n")
        print("children의 Fo평균값 : "+str(float(mean_childrens)/float(childrens_num))+"\n")
        print("children의 Fo최대값 : "+str(childrens_f0_max)+"\n")
        '''

    def get_duration(self,audio_path):
        audio = wave.open(audio_path)
        frames = audio.getnframes()
        rate = audio.getframerate()
        duration = frames / float(rate)
        return duration
    def get_file_paths__(self,olds_female_training_path,olds_male_training_path,nonolds_female_training_path,nonolds_male_training_path,childrens_training_path):
        olds_female=[ os.path.join(olds_female_training_path, f) for f in os.listdir(olds_female_training_path) ]
        olds_male=[ os.path.join(olds_male_training_path, f) for f in os.listdir(olds_male_training_path) ]
        nonolds_female=[ os.path.join(nonolds_female_training_path, f) for f in os.listdir(nonolds_female_training_path) ]
        nonolds_male=[ os.path.join(nonolds_male_training_path, f) for f in os.listdir(nonolds_male_training_path) ]
        childrens=[ os.path.join(childrens_training_path, f) for f in os.listdir(childrens_training_path) ]

        return olds_female,olds_male,nonolds_female,nonolds_male,childrens

    def new_get_file_paths(self, females_training_path, males_training_path,childrens_training_path):

        # get file paths
        females = [ os.path.join(females_training_path, f) for f in os.listdir(females_training_path) ]
        males   = [ os.path.join(males_training_path, f) for f in os.listdir(males_training_path) ]
        childrens = [ os.path.join(childrens_training_path, f) for f in os.listdir(childrens_training_path) ]
        return females, males,childrens

    def get_file_paths(self, females_training_path, males_training_path,childrens_training_path):
        # get file paths
        females = [ os.path.join(females_training_path, f) for f in os.listdir(females_training_path) ]
        males   = [ os.path.join(males_training_path, f) for f in os.listdir(males_training_path) ]
        childrens = [ os.path.join(childrens_training_path, f) for f in os.listdir(childrens_training_path) ]
        return females, males,childrens

    def collect_features_(self, files):
        """
    	Collect voice features from various speakers of the same gender.

    	Args:
    	    files (list) : List of voice file paths.

    	Returns:
    	    (array) : Extracted features matrix.
    	"""
        num_=0
        features = np.asarray(())
        # extract features for each speaker
        for file in files:
            print("%5s %10s" % ("PROCESSNG ", file))
            # extract MFCC & delta MFCC features from audio
            vector    = self.features_extractor_.extract_features(file)
            # stack the features
            if features.size == 0:  features = vector
            else:                   features = np.vstack((features, vector))
            num_+=1
            print(num_)
        return features
    def collect_features(self, files):
        """
    	Collect voice features from various speakers of the same gender.

    	Args:
    	    files (list) : List of voice file paths.

    	Returns:
    	    (array) : Extracted features matrix.
    	"""
        features = np.asarray(())
        # extract features for each speaker
        mean_f0=0
        mean_HNR=0
        num_=0
        max=0
        mean_stdev=0
        mean_shimmer=0
        mean_jitter=0
        for file in files:
            silence_removal(file,file.split('.')[0] + "_without_silence.wav")

            # extract MFCC & delta MFCC features from audio
            length=self.get_duration(file.split('.')[0] + "_without_silence.wav")
            print(length)
            if length < 1:
                
                continue
            print("%5s %10s" % ("PROCESSING ", file))
            # extract MFCC & delta MFCC features from audio
            vector    = self.features_extractor.extract_features(file)
            
            
            '''
            Fo_value=vector[0][0].reshape(-1,1)
            HNR_value=(float(vector_list[0][1])).reshpae(-1,1)
            jitter_value=(float(vector_list[0][2])).reshpe(-1,1)
            shimmer_value=(float(vector_list[0][3])).reshape(-1,1)
            combined      = np.hstack((Fo_value, HNR_value, jitter_value,shimmer_value))
            
            print(combined)
            
            '''
            print(vector)
            
            
            mean_HNR+=float(vector[0])
            mean_stdev+=float(vector[1])
            mean_shimmer+=float(vector[2])
            mean_jitter+=float(vector[3])
            
            '''
            if float(vector[0])>max:
                max=float(vector[0])
            '''
            
            # stack the features
            num_+=1
            print(num_)
            
            if features.size == 0:  features = vector
            else:                   features = np.vstack((features, vector))
            
            os.remove(file.split('.')[0] + "_without_silence.wav")
        return features,num_,mean_HNR,mean_stdev,mean_shimmer,mean_jitter

    def save_gmm(self, gmm, name):
        """ Save Gaussian mixture model using pickle.

            Args:
                gmm        : Gaussian mixture model.
                name (str) : File name.
        """
        filename = name + ".gmm"
        with open(filename, 'wb') as gmm_file:
            pickle.dump(gmm, gmm_file)
        print ("%5s %10s" % ("SAVING", filename,))


if __name__== "__main__":
    models_trainer = ModelsTrainer("C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\svmCode\\TrainingData\\nonolds_female_", "C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\svmCode\\TrainingData\\nonolds_male_","C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\svmCode\\TrainingData\\olds","C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\svmCode\\TrainingData\\nonolds","C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\svmCode\\TrainingData\\childrens_",
    "C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\svmCode\\TrainingData\\olds_male","C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\svmCode\\TrainingData\\olds_female_")
    models_trainer.oldprocess()
