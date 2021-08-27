import os
import pickle
import warnings
import numpy as np
from python_speech_features.base import mfcc
from FeaturesExtractor_other import FeaturesExtractor
from FeaturesExtractor import FeaturesExtractor_
import os
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from featureextractor_for_old import featureextractor__
warnings.filterwarnings("ignore")


class GenderIdentifier:

    def __init__(self, nonolds_females_path,nonolds_male_path,childrens_files_path,old_female_path,old_male_path, females_model_path, males_model_path,olds_model_path,nonolds_model_path,childrens_model_path,olds_female_model_path,olds_male_model_path,nonolds_female_model_path,nonolds_male_model_path,nonoldfemale_another_path,nonoldmale_another_path,children_another_path,olds_female_anotherpath,olds_male_another_path):
       
        
        self.childrens_training_path=childrens_files_path
        self.nonolds_female_training_path=nonolds_females_path
        self.nonolds_male_training_path=nonolds_male_path
        self.olds_female_training_path=old_female_path
        self.olds_male_training_path=old_male_path
        self.olds_female_another_path=olds_female_anotherpath
        self.olds_male_another_path=olds_male_another_path
        self.error                 = 0
        self.total_sample          = 0
        self.features_extractor    = FeaturesExtractor()
        self.features_extractor_    = FeaturesExtractor_()
        self.features_extractor__   = featureextractor__
        # load models
        self.anotherchildren_gmm=pickle.load(open(children_another_path,'rb'))
        self.anotherfemales_gmm=pickle.load(open(nonoldfemale_another_path, 'rb'))
        self.anothermales_gmm=pickle.load(open(nonoldmale_another_path, 'rb'))
        self.females_gmm = pickle.load(open(females_model_path, 'rb'))
        self.males_gmm   = pickle.load(open(males_model_path, 'rb'))
        self.olds_gmm   = pickle.load(open(olds_model_path, 'rb'))
        self.nonolds_gmm   = pickle.load(open(nonolds_model_path, 'rb'))
        self.childrens_gmm   = pickle.load(open(childrens_model_path, 'rb'))
        self.old_females_gmm=pickle.load(open(olds_female_model_path,'rb'))
        self.old_males_gmm=pickle.load(open(olds_male_model_path,'rb'))
        self.nonold_females_gmm=pickle.load(open(nonolds_female_model_path,'rb'))
        self.nonold_males_gmm=pickle.load(open(nonolds_male_model_path,'rb'))
        self.another_old_females_gmm=pickle.load(open(olds_female_anotherpath,'rb'))
        self.another_old_males_gmm=pickle.load(open(olds_male_another_path,'rb'))
    def oldageprocess(self):
        files=self.get_file_paths__(self.olds_female_training_path,self.olds_male_training_path,self.nonolds_female_training_path,self.nonolds_male_training_path,self.childrens_training_path)
        fault_list=[0,0,0,0,0]
        old_female_fault=[0,0,0,0]
        old_male_fault=[0,0,0,0]
        nonold_female_fault=[0,0,0,0]
        nonold_male_fault=[0,0,0,0]
        childrens_fault=[0,0,0,0]
        total_score=[]
        names=["olds_female","olds_male","nonolds_female","nonold_male","children"]
        
        num__=0
        for file in files:
            
            self.total_sample += 1
            print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(file)))
            

            vector_praat = self.features_extractor.extract_features(file)
            vector_mfcc=self.features_extractor_.extract_features(file)
            vector_for_old=self.features_extractor__.getfeautre(file)
            

            winner_praat,of_praat_score,om_praat_score,nf_praat_score,nm_praat_score=self.identify_oldage_praat(vector_praat)
            winner_mfcc,of_mfcc_score,om_mfcc_score,nf_mfcc_score,nm_mfcc_score,children_mfcc_score = self.identify_oldage_mfcc(vector_mfcc)
        



            expected_result=file.split("\\")[7][:-1]
            old_score_mfcc=of_mfcc_score+om_mfcc_score
            old_score_praat=of_praat_score+om_praat_score
            nonold_score_mfcc=nf_mfcc_score+nm_mfcc_score
            nonold_score_praat=nf_praat_score+nm_praat_score
            female_score_mfcc=of_mfcc_score+nf_mfcc_score
            female_score_praat=of_praat_score+nf_praat_score
            male_score_mfcc=om_mfcc_score+nm_mfcc_score
            male_score_praat=om_praat_score+nm_praat_score
            
            age_result="none"
            gender_result="none"
            real_winner="none"
            i=0
            
            old_score=[]
            mfcc_score=[]
            '''
            max=total_score[0]
            for i in range(0,5):
                if total_score[i]>max:
                    max=total_score[i]
                    num_=i
                else:
                    max=max
            '''
            print(total_score)

         
            '''
            if (nm_mfcc_score>om_mfcc_score) :
                age_result="olds_"
            else:
                age_result="nonolds_"
        
                
                if (old_score_mfcc+old_score_praat)>(nonold_score_mfcc+nonold_score_praat):
                    age_result="olds_"
                else:
                    age_result="nonolds_"
            '''
            

            if  (female_score_praat>male_score_praat):
                gender_result="female"
            else:
                gender_result="male"
            '''
            if gender_result=="female":
                if (old_score_mfcc+old_score_praat>nonold_score_mfcc+nonold_score_praat) :
                    age_result="olds_"
                else:
                    age_result="nonolds_"
            else:
            '''
            age_result,old_age_score,nonold_age_score,children_age_score=self.identify_only_old(vector_for_old) 

            old_score.append(old_age_score)
            old_score.append(nonold_age_score)
            old_score.append(children_age_score)
            mfcc_score.append(of_mfcc_score)
            mfcc_score.append(nf_mfcc_score)
            mfcc_score.append(om_mfcc_score)
            mfcc_score.append(nf_mfcc_score)
            mfcc_score.append(children_mfcc_score)

        
            i=0
            old_num=0
            mfcc_num=0
            max_old=old_score[0]
            max_mfcc=mfcc_score[0]
            for i in range(0,3):
                if old_score[i]>max_old:
                    max_old=old_score[i]
                    old_num=i
                else:
                    max_old=max_old
            new_mfcc_score=self.bubble_sort(mfcc_score)
            print(new_mfcc_score)
            a=mfcc_score[4]
            if (new_mfcc_score.index(a)==4 or new_mfcc_score.index(a)==3) and old_num==2:
                
                real_winner="children"
            else:
                if gender_result == "female":
                    old_score=old_age_score+of_mfcc_score
                    nonold_score=nonold_age_score+nf_mfcc_score
                    
                else:
                    old_score=old_age_score+om_mfcc_score
                    nonold_score=nonold_age_score+nm_mfcc_score
        
                if old_score > nonold_score :
                    age_result="olds_"
                else:
                    age_result="nonolds_"
                real_winner=age_result+gender_result      
           
            i=0
            '''
            max_mfcc=total_score_mfcc[0]
            max_praat=total_score_praat[0]
            for i in range(0,5):
                if total_score_mfcc[i]>max_mfcc:
                    max_mfcc=total_score_mfcc[i]
                    num_mfcc=i
                else:
                    max_mfcc=max_mfcc
                if total_score_praat[i]>max_praat:
                    max_praat=total_score_praat[i]
                    num_praat=i
                else:
                    max_praat=max_praat
            
            if num_praat==4 and num_mfcc==4:
                real_winner="children"
            else:
                real_winner=real_winner_guess
            '''
            
            '''
            if real_winner=="children":
                real_winner="children"
            else:
                if real_winner_guess==real_winner_score:
                    real_winner=real_winner_guess
                else:
                    if old_score_mfcc>nonold_score_mfcc:
                        age_result="olds_"
                        real_winner=age_result+gender_result
                    else:
                        age_result="nonolds_"
                        real_winner=age_result+gender_result
            
            '''
            print("%10s %6s %1s" % ("+ EXPECTATION",":", expected_result))
            print("%10s %3s %1s" %  ("+ IDENTIFICATION", ":", real_winner))
            

            if real_winner != expected_result: 
                self.error += 1
                if expected_result=="olds_female":
                    fault_list[0]+=1
                    if real_winner=="olds_male":#예상은old 인식은nonold
                        old_female_fault[0]+=1
                    elif real_winner=="nonolds_female":#예상은 old 인식은 children
                        old_female_fault[1]+=1
                    elif real_winner=="nonolds_male":
                        old_female_fault[2]+=1
                    else:
                        old_female_fault[3]+=1
                elif expected_result=="olds_male":
                    fault_list[1]+=1
                    if real_winner=="olds_female":#예상은old 인식은nonold
                        old_male_fault[0]+=1
                    elif real_winner=="nonolds_female":#예상은 old 인식은 children
                        old_male_fault[1]+=1
                    elif real_winner=="nonolds_male":
                        old_male_fault[2]+=1
                    else:
                        old_male_fault[3]+=1
                elif expected_result=="nonolds_female":
                    fault_list[2]+=1
                    if real_winner=="olds_female":#예상은old 인식은nonold
                        nonold_female_fault[0]+=1
                    elif real_winner=="olds_male":#예상은 old 인식은 children
                        nonold_female_fault[1]+=1
                    elif real_winner=="nonolds_male":
                        nonold_female_fault[2]+=1
                    else:
                        nonold_female_fault[3]+=1
                elif expected_result=="nonolds_male":
                    fault_list[3]+=1
                    if real_winner=="olds_female":#예상은old 인식은nonold
                        nonold_male_fault[0]+=1
                    elif real_winner=="olds_male":#예상은 old 인식은 children
                        nonold_male_fault[1]+=1
                    elif real_winner=="nonolds_female":
                        nonold_male_fault[2]+=1
                    else:
                        nonold_male_fault[3]+=1
                else:
                    fault_list[4]+=1
                    if real_winner=="olds_female":#예상은old 인식은nonold
                        childrens_fault[0]+=1
                    elif real_winner=="olds_male":#예상은 old 인식은 children
                        childrens_fault[1]+=1
                    elif real_winner=="nonolds_female":
                        childrens_fault[2]+=1
                    else:
                        childrens_fault[3]+=1
            print("old_female_fault:"+str(fault_list[0])+"\n")
            print("old(female) fault => " + "old(male) : "+str(old_female_fault[0])+" nonold(female ): "+str(old_female_fault[1])+" nonold(male ): "+str(old_female_fault[2])+" children : "+str(old_female_fault[3])+"\n")
            print("old_male_fault:"+str(fault_list[1])+"\n")
            print("old(male) fault => " + "old(female) : "+str(old_male_fault[0])+" nonold(female ): "+str(old_male_fault[1])+" nonold(male ): "+str(old_male_fault[2])+" children : "+str(old_male_fault[3])+"\n")
            print("nonold_female_fault :"+str(fault_list[2])+"\n")
            print("nonold(female) fault => " + "old(female) : "+str(nonold_female_fault[0])+" old(male ): "+str(nonold_female_fault[1])+" nonold(male): "+str(nonold_female_fault[2])+" children : "+str(nonold_female_fault[3])+"\n")
            print("nonold_male_fault :"+str(fault_list[3])+"\n")
            print("nonold(male) fault => " + "old(female) : "+str(nonold_male_fault[0])+" old(male ): "+str(nonold_male_fault[1])+" nonold(female ): "+str(nonold_male_fault[2])+" children : "+str(nonold_male_fault[3])+"\n")
            print("childrens_fault : "+str(fault_list[4])+"\n")
            print("childrens fault => " + "old(female) : "+str(childrens_fault[0])+" old(male ): "+str(childrens_fault[1])+" nonold(female): "+str(childrens_fault[2])+" nonold(male) : "+str(childrens_fault[3])+"\n")
            num__+=1
            print(num__)

            

               

        accuracy     = ( float(self.total_sample - self.error) / float(self.total_sample) ) * 100
        accuracy_msg = "*** Accuracy = " + str(round(accuracy, 3)) + "% ***"
       
        print(accuracy_msg)
        '''
        print("old(female)_fault : "+str(fault_list[0])+"\n")
        print("old(female) fault => " + "old(male) : "+str(old_female_fault[0])+" nonold(female ): "+str(old_female_fault[1])+" nonold(male ): "+str(old_female_fault[2])+" children : "+str(old_female_fault[3])+"\n")
        print("old(male)_fault : "+str(fault_list[1])+"\n")
        print("old(male) fault => " + "old(female) : "+str(old_male_fault[0])+" nonold(female ): "+str(old_male_fault[1])+" nonold(male ): "+str(old_male_fault[2])+" children : "+str(old_male_fault[3])+"\n")
        print("nonold(female)_fault : "+str(fault_list[2])+"\n")
        print("nonold(female) fault => " + "old(female) : "+str(nonold_female_fault[0])+" old(male ): "+str(nonold_female_fault[1])+" nonold(male): "+str(nonold_female_fault[2])+" children : "+str(nonold_female_fault[3])+"\n")
        print("nonold(male)_fault : "+str(fault_list[3])+"\n")
        print("nonold(male) fault => " + "old(female) : "+str(nonold_male_fault[0])+" old(male ): "+str(nonold_male_fault[1])+" nonold(female ): "+str(nonold_male_fault[2])+" children : "+str(nonold_male_fault[3])+"\n")
        print("childrens_fault : "+str(fault_list[4])+"\n")
        print("childrens fault => " + "old(female) : "+str(childrens_fault[0])+" old(male ): "+str(childrens_fault[1])+" nonold(female): "+str(childrens_fault[2])+" nonold(male) : "+str(childrens_fault[3])+"\n")
        '''
        old_female_num=len(os.listdir(self.olds_female_training_path))
        old_male_num=len(os.listdir(self.olds_male_training_path))
        nonold_female_num=len(os.listdir(self.nonolds_female_training_path))
        nonold_male_num=len(os.listdir(self.nonolds_male_training_path))
        children_num=len(os.listdir(self.childrens_training_path))
        print("acurracy score => "+ "old_female : "+str(float(old_female_num-fault_list[0])/float(old_female_num))+"  old_male : "+str(float(old_male_num-fault_list[1])/float(old_male_num))+"  non_old_female : "+str(float(nonold_female_num-fault_list[2])/float(nonold_female_num))+"  non_old_male : "+str(float(nonold_male_num-fault_list[3])/float(nonold_male_num))+"  children : "+str(float(children_num-fault_list[4])/float(children_num)))


        

    def oldprocess(self):
        files=self.get_file_paths_(self.nonolds_training_path,self.olds_training_path,self.childrens_training_path)
        fault_list=[0,0,0]
        old_fault=[0,0]
        nonold_fault=[0,0]
        children_fault=[0,0]
        for file in files:
            self.total_sample += 1
            print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(file)))
            vector = self.features_extractor.extract_features(file)
            winner = self.identify_old(vector)
            expected_age = file.split("\\")[7][:-1]

            print("%10s %6s %1s" % ("+ EXPECTATION",":", expected_age))
            print("%10s %3s %1s" %  ("+ IDENTIFICATION", ":", winner))
            

            if winner != expected_age: 
                self.error += 1
                if expected_age=="old":
                    fault_list[0]+=1
                    if winner=="nonold":#예상은old 인식은nonold
                        old_fault[0]+=1
                    else:#예상은 old 인식은 children
                        old_fault[1]+=1
                elif expected_age=="nonold":
                    fault_list[1]+=1
                    if winner=="old":#예상은nonold 인식은 old
                        nonold_fault[0]+=1
                    else:#예상은 nonold 인식은 children
                        nonold_fault[1]+=1
                else:
                    fault_list[2]+=1
                    if winner=="old":#예상은children 인식은 old
                        children_fault[0]+=1
                    else:#예상은 children 인식은 nonold
                        children_fault[1]+=1



        accuracy     = ( float(self.total_sample - self.error) / float(self.total_sample) ) * 100
        accuracy_msg = "*** Accuracy = " + str(round(accuracy, 3)) + "% ***"
       
        print(accuracy_msg)
        print("old_fault : "+str(fault_list[0])+"\n")
        print("nonold_fault : "+str(fault_list[1])+"\n")
        print("old : "+str(fault_list[0])+"\n")
        print("old fault => " + "nonold : "+str(old_fault[0])+" children : "+str(old_fault[1])+"\n")
        print("nonold : "+str(fault_list[1])+"\n")
        print("nonold fault => " + "old : "+str(nonold_fault[0])+" children : "+str(nonold_fault[1])+"\n")
        print("children : "+str(fault_list[2])+"\n")
        print("children fault => " + "old : "+str(children_fault[0])+" nonold : "+str(children_fault[1])+"\n")
    
    def onlyoldprocess(self):
        files=self.new_get_file_paths(self.olds_female_training_path,self.olds_male_training_path,self.nonolds_female_training_path,self.nonolds_male_training_path)
        fault_list=[0,0]
        female_list=[0,0]
        male_list=[0,0]
        for file in files:
            self.total_sample += 1
            print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(file)))

            vector = self.features_extractor_.extract_features(file)
            winner = self.identify_only_old(vector)
            if "nonold" in file.split("\\")[7][:-1]:
                expected_gender = "nonold"
            else:
                expected_gender = "old"
            
            print("%10s %6s %1s" % ("+ EXPECTATION",":", expected_gender))
            print("%10s %3s %1s" %  ("+ IDENTIFICATION", ":", winner))

            if winner != expected_gender: 
                self.error += 1
                if expected_gender=="nonold":
                    fault_list[0]+=1
                    if  "female" in file.split("\\")[7][:-1]:
                        female_list[0]+=1
                    else:
                        male_list[0]+=1
                else:
                    fault_list[1]+=1
                    if "female" in file.split("\\")[7][:-1]:
                        female_list[1]+=1
                    else:
                        male_list[1]+=1
                
            


            print("----------------------------------------------------")

        accuracy     = ( float(self.total_sample - self.error) / float(self.total_sample) ) * 100
        accuracy_msg = "*** Accuracy = " + str(round(accuracy, 3)) + "% ***"
        print(accuracy_msg)
        print("nonold : "+str(fault_list[0])+"\n")
        print("nonold_female :"+str(female_list[0])+"nonold_male"+str(male_list[0]))
        print("old : "+str(fault_list[1])+"\n")
        print("old_female : " + str(female_list[1])+ "old_male : "+ str(male_list[1]))


    def genderprocess(self):
        files=self.new_get_file_paths(self.olds_female_training_path,self.olds_male_training_path,self.nonolds_female_training_path,self.nonolds_male_training_path,self.childrens_training_path)
        fault_list=[0,0,0]
        nonold_list=[0,0,0]
        old_list=[0,0,0]
        children_list=[0,0,0,0]
        num_=0
        for file in files:
            self.total_sample += 1
            print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(file)))

            vector_praat = self.features_extractor.extract_features(file)
            vector_mfcc=self.features_extractor_.extract_features(file)
            print(vector_praat[0])

            winner_,f_praat_score,m_praat_score,children_praat_score=self.identify_gender_mfcc(vector_mfcc)
            winner,f_mfcc_score,m_mfcc_score,children_mfcc_score = self.identify_gender_praat(vector_praat)

            if "female" in file.split("\\")[7][:-1]:
                expected_gender = "female"
            elif "male" in file.split("\\")[7][:-1]:
                expected_gender="male"
            else:
                expected_gender="children"

            if(winner_==winner):
                real_winner=winner
            else:
                if (f_praat_score+f_mfcc_score)>(m_praat_score+m_mfcc_score):
                    if (f_praat_score+f_mfcc_score)>(children_praat_score+children_mfcc_score):
                        real_winner="female"
                    else:
                        real_winner="children"
                else:
                    if (m_praat_score+m_mfcc_score)>(children_praat_score+children_mfcc_score):
                        real_winner="male"
                    else:
                        real_winner="children"


            print("%10s %6s %1s" % ("+ EXPECTATION",":", expected_gender))
            print("%10s %3s %1s" %  ("+ IDENTIFICATION", ":", real_winner))
            

            if real_winner != expected_gender: 
                self.error += 1
                if expected_gender=="female":
                    fault_list[0]+=1

                    if real_winner=="children":
                        
                        if "nonold" in file.split("\\")[7][:-1]:
                            children_list[0]+=1
                            nonold_list[0]+=1
                        else:
                            children_list[1]+=1
                            old_list[0]+=1
                    
                    else:
                        if "nonold" in file.split("\\")[7][:-1]:
                            nonold_list[0]+=1
                        else:
                            old_list[0]+=1
                    
                elif expected_gender=="male":
                    fault_list[1]+=1
                    if real_winner=="children":
                        if "nonold" in file.split("\\")[7][:-1]:
                            children_list[2]+=1
                            nonold_list[1]+=1
                        else:
                            children_list[3]+=1
                            old_list[1]+=1
                    else:
                        if "nonold" in file.split("\\")[7][:-1]:
                            nonold_list[1]+=1
                        else:
                            old_list[1]+=1
                else:
                    fault_list[2]+=1
                

            print("old_female_fault:"+str(old_list[0])+"\n")
            print("old_male_fault:"+str(old_list[1])+"\n")
            print("nonold_female_fault :"+str(nonold_list[0])+"\n")
            print("nonold_male_fault :"+str(nonold_list[1])+"\n")
            print("children_fault =>"+"nonold_female : "+str(children_list[0])+"old_female : "+str(children_list[1])+"nonold_male : "+str(children_list[2])+"old_male : "+str(children_list[3]))
            

            num_+=1
            print(num_)

            print("----------------------------------------------------")

        accuracy     = ( float(self.total_sample - self.error) / float(self.total_sample) ) * 100
        accuracy_msg = "*** Accuracy = " + str(round(accuracy, 3)) + "% ***"
        print(accuracy_msg)
        print("female : "+str(fault_list[0])+"\n")
        print("nonold에러 : " + str(nonold_list[0])+"old에러 : "+str(old_list[0]))
        print("male : "+str(fault_list[1])+"\n")
        print("nonold에러 : " + str(nonold_list[1])+"old에러 : "+str(old_list[1]))
        print("acurracy score => "+ "old_female : "+str(float(old_list[0])/float(len(os.listdir(self.olds_female_training_path))))+"old_male : "+str(float(old_list[1])/float(len(os.listdir(self.olds_male_training_path))))+"non_old_female : "+str(float(nonold_list[0])/float(len(os.listdir(self.nonolds_female_training_path))))+"non_old_male : "+str(float(nonold_list[1])/float(len(os.listdir(self.nonolds_male_training_path))))+"children : "+str(float(fault_list[2])/float(len(os.listdir(self.childrens_training_path)))))




    def process(self):
        files = self.get_file_paths(self.females_training_path, self.males_training_path)
        fault_list=[0,0]
        # read the test directory and get the list of test audio files
        for file in files:
            self.total_sample += 1
            print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(file)))

            vector = self.features_extractor.extract_features(file)
            winner = self.identify_gender(vector)
            expected_gender = file.split("\\")[7][:-1]

            print("%10s %6s %1s" % ("+ EXPECTATION",":", expected_gender))
            print("%10s %3s %1s" %  ("+ IDENTIFICATION", ":", winner))

            if winner != expected_gender: 
                self.error += 1
                if expected_gender=="female":
                    fault_list[0]+=1
                else:
                    fault_list[1]+=1


        
            print("----------------------------------------------------")

        accuracy     = ( float(self.total_sample - self.error) / float(self.total_sample) ) * 100
        accuracy_msg = "*** Accuracy = " + str(round(accuracy, 3)) + "% ***"
        print(accuracy_msg)
        print("female : "+str(fault_list[0])+"\n")
        print("male : "+str(fault_list[1])+"\n")

    def new_get_file_paths(self,old_females_training_path,old_males_training_path,nonold_females_training_path,nonold_males_training_path,childrens_training_path):
        old_females = [ os.path.join(old_females_training_path, f) for f in os.listdir(old_females_training_path) ]
        old_males = [ os.path.join(old_males_training_path, f) for f in os.listdir(old_males_training_path) ]
        nonold_females = [ os.path.join(nonold_females_training_path, f) for f in os.listdir(nonold_females_training_path) ]
        nonold_males = [ os.path.join(nonold_males_training_path, f) for f in os.listdir(nonold_males_training_path) ]
        childrens= [ os.path.join(childrens_training_path, f) for f in os.listdir(childrens_training_path) ]
        files=childrens+nonold_females+nonold_males+old_females+old_males

        return files
    def bubble_sort(self,arr):
        for i in range(len(arr)-1,0,-1):
            for j in range(i):
                if arr[j]>arr[j+1]:
                    arr[j],arr[j+1]=arr[j+1],arr[j]
        return arr
    def get_file_paths__(self,old_females_training_path,old_males_training_path,nonold_females_training_path,nonold_males_training_path,childrens_training_path):
        old_females = [ os.path.join(old_females_training_path, f) for f in os.listdir(old_females_training_path) ]
        old_males = [ os.path.join(old_males_training_path, f) for f in os.listdir(old_males_training_path) ]
        nonold_females = [ os.path.join(nonold_females_training_path, f) for f in os.listdir(nonold_females_training_path) ]
        nonold_males = [ os.path.join(nonold_males_training_path, f) for f in os.listdir(nonold_males_training_path) ]
        childrens= [ os.path.join(childrens_training_path, f) for f in os.listdir(childrens_training_path) ]
        files=childrens+nonold_males+old_females+old_males+nonold_females

        return files


    def get_file_paths(self, females_training_path, males_training_path):
        # get file paths
        females = [ os.path.join(females_training_path, f) for f in os.listdir(females_training_path) ]
        males   = [ os.path.join(males_training_path, f) for f in os.listdir(males_training_path) ]
        files   = females + males
        return files
    

    def get_file_paths_(self,nonold_training_path,old_training_path,childrens_training_path):
        olds=[ os.path.join(old_training_path, f) for f in os.listdir(old_training_path) ]
        nonolds=[ os.path.join(nonold_training_path, f) for f in os.listdir(nonold_training_path) ]
        childrens=[ os.path.join(childrens_training_path, f) for f in os.listdir(childrens_training_path) ]
        files=olds+nonolds+childrens
        return files

    def identify_oldage_mfcc(self,vector):
        scores=[]
        names=["olds_female","olds_male","nonolds_female","nonold_male","children"]
        num_=0

        is_oldfemale_scores         = np.array(self.another_old_females_gmm.score(vector))
        is_oldfemale_log_likelihood = is_oldfemale_scores.sum()
        scores.append(is_oldfemale_log_likelihood)

        is_oldmale_scores         = np.array(self.another_old_males_gmm.score(vector))
        is_oldmale_log_likelihood = is_oldmale_scores.sum()
        scores.append(is_oldmale_log_likelihood)
        
        # male hypothesis scoring
        is_nonold_female_scores         = np.array(self.anotherfemales_gmm.score(vector))
        is_nonoldfemale_log_likelihood = is_nonold_female_scores.sum()
        scores.append(is_nonoldfemale_log_likelihood)

        is_nonold_male_scores = np.array(self.anothermales_gmm.score(vector))
        is_nonoldmale_log_likelihood=is_nonold_male_scores.sum()
        scores.append(is_nonoldmale_log_likelihood)

        is_children_scores         = np.array(self.anotherchildren_gmm.score(vector))
        is_children_log_likelihood = is_children_scores.sum() 
        scores.append(is_children_log_likelihood)
        max=scores[0]

        print("%10s %5s %1s" % ("+ old(female) SCORE",":", str(round(is_oldfemale_log_likelihood, 3))))
        print("%10s %7s %1s" % ("+ old(male) SCORE", ":", str(round(is_oldmale_log_likelihood,3))))
        print("%10s %7s %1s" % ("+ nonold(female) SCORE", ":", str(round(is_nonoldfemale_log_likelihood,3))))
        print("%10s %7s %1s" % ("+ nonold(male) SCORE", ":", str(round(is_nonoldmale_log_likelihood,3))))
        print("%10s %7s %1s" % ("+ children SCORE", ":", str(round(is_children_log_likelihood,3))))

        i=0
        for i in range(0,5):
            if scores[i]>max:
                max=scores[i]
                num_=i
            else:
                max=max
        winner=names[num_]
        
        
        return winner,is_oldfemale_log_likelihood,is_oldmale_log_likelihood,is_nonoldfemale_log_likelihood,is_nonoldmale_log_likelihood,is_children_log_likelihood

    def identify_oldage_praat(self,vector):
        scores=[]
        names=["olds_female","olds_male","nonolds_female","nonold_male","children"]
        num_=0
        
        is_oldfemale_scores         = np.array(self.old_females_gmm.score(vector))
        is_oldfemale_log_likelihood = is_oldfemale_scores.sum()
        scores.append(is_oldfemale_log_likelihood)

        is_oldmale_scores         = np.array(self.old_males_gmm.score(vector))
        is_oldmale_log_likelihood = is_oldmale_scores.sum()
        scores.append(is_oldmale_log_likelihood)
        
        # male hypothesis scoring
        is_nonold_female_scores         = np.array(self.females_gmm.score(vector))
        is_nonoldfemale_log_likelihood = is_nonold_female_scores.sum()
        scores.append(is_nonoldfemale_log_likelihood)

        is_nonold_male_scores = np.array(self.males_gmm.score(vector))
        is_nonoldmale_log_likelihood=is_nonold_male_scores.sum()
        scores.append(is_nonoldmale_log_likelihood)

        
        max=scores[0]
        

        print("%10s %5s %1s" % ("+ old(female) SCORE",":", str(round(is_oldfemale_log_likelihood, 3))))
        print("%10s %7s %1s" % ("+ old(male) SCORE", ":", str(round(is_oldmale_log_likelihood,3))))
        print("%10s %7s %1s" % ("+ nonold(female) SCORE", ":", str(round(is_nonoldfemale_log_likelihood,3))))
        print("%10s %7s %1s" % ("+ nonold(male) SCORE", ":", str(round(is_nonoldmale_log_likelihood,3))))
       
        i=0
        for i in range(0,4):
            if scores[i]>max:
                max=scores[i]
                num_=i
            else:
                max=max
        winner=names[num_]
        
        
        return winner,is_oldfemale_log_likelihood,is_oldmale_log_likelihood,is_nonoldfemale_log_likelihood,is_nonoldmale_log_likelihood

    def identify_only_old(self,vector):
        is_old_scores         = np.array(self.olds_gmm.score(vector))
        print(is_old_scores)
        
        '''is_old_scores         = np.array(cosine_similarity(self.olds_gm_m,vector))'''
        is_old_log_likelihood = is_old_scores.sum()
        
        # male hypothesis scoring
        is_nonold_scores         = np.array(self.nonolds_gmm.score(vector))
        is_nonold_log_likelihood = is_nonold_scores.sum()

        is_children_scores = np.array(self.childrens_gmm.score(vector))
        is_children_log_likelihood = is_children_scores.sum()

        
       
        print("%10s %5s %1s" % ("+ old SCORE",":", str(round(is_old_log_likelihood, 3))))
        print("%10s %7s %1s" % ("+ nonold SCORE", ":", str(round(is_nonold_log_likelihood,3))))
        
        print("%10s %7s %1s" % ("+ children SCORE", ":", str(round(is_children_log_likelihood,3))))

        if is_old_log_likelihood > is_nonold_log_likelihood:
            if is_old_log_likelihood > is_children_log_likelihood:
                winner="olds_"
            else:
                winner="children"
        else:
            if is_nonold_log_likelihood > is_children_log_likelihood:
                winner="nonolds_"
            else:
                winner="children"
        return winner,is_old_log_likelihood,is_nonold_log_likelihood,is_children_log_likelihood
    def identify_old(self,vector):
        is_old_scores         = np.array(self.olds_gmm.score(vector))
        
        is_old_log_likelihood = is_old_scores.sum()
        
        # male hypothesis scoring
        is_nonold_scores         = np.array(self.nonolds_gmm.score(vector))
        is_nonold_log_likelihood = is_nonold_scores.sum()


        is_children_scores         = np.array(self.childrens_gmm.score(vector))
        is_children_log_likelihood = is_children_scores.sum() 

        print("%10s %5s %1s" % ("+ old SCORE",":", str(round(is_old_log_likelihood, 3))))
        print("%10s %7s %1s" % ("+ nonold SCORE", ":", str(round(is_nonold_log_likelihood,3))))
        print("%10s %7s %1s" % ("+ children SCORE", ":", str(round(is_children_log_likelihood,3))))


        if is_old_log_likelihood > is_nonold_log_likelihood:
            if is_old_log_likelihood>is_children_log_likelihood:
                winner="old"
            else:
                winner="children"
        else:
            if is_nonold_log_likelihood>is_children_log_likelihood:
                winner="nonold"
            else:
                winner="children"
        return winner
    def identify_gender_mfcc(self,vector):
        is_female_scores         = np.array(self.anotherfemales_gmm.score(vector))
        is_female_log_likelihood = is_female_scores.sum()
        # male hypothesis scoring
        is_male_scores         = np.array(self.anothermales_gmm.score(vector))
        is_male_log_likelihood = is_male_scores.sum()

        is_children_scores         = np.array(self.anotherchildren_gmm.score(vector))
        is_children_log_likelihood = is_children_scores.sum() 

        print("%10s %5s %1s" % ("+ mfcc-FEMALE SCORE",":", str(round(is_female_log_likelihood, 3))))
        print("%10s %7s %1s" % ("+ mfcc-MALE SCORE", ":", str(round(is_male_log_likelihood,3))))
        print("%10s %7s %1s" % ("+ mfcc-children SCORE", ":", str(round(is_children_log_likelihood,3))))

        if is_male_log_likelihood > is_female_log_likelihood: 
            if is_male_log_likelihood>is_children_log_likelihood:
                winner="male"
            else:
                winner="children"
        else:
            if is_female_log_likelihood>is_children_log_likelihood:
                winner="female"
            else:
                winner="children"

        


        return winner,is_female_log_likelihood,is_male_log_likelihood,is_children_log_likelihood


    def identify_gender_praat(self, vector):
        # female hypothesis scoring
        is_female_scores         = np.array(self.females_gmm.score(vector))
        print(is_female_scores)
        is_female_log_likelihood = is_female_scores.sum()
        # male hypothesis scoring
        is_male_scores         = np.array(self.males_gmm.score(vector))
        is_male_log_likelihood = is_male_scores.sum()

        is_children_scores         = np.array(self.childrens_gmm.score(vector))
        is_children_log_likelihood = is_children_scores.sum() 

        print("%10s %5s %1s" % ("+ praat-FEMALE SCORE",":", str(round(is_female_log_likelihood, 3))))
        print("%10s %7s %1s" % ("+ praat-MALE SCORE", ":", str(round(is_male_log_likelihood,3))))
        print("%10s %7s %1s" % ("+ praat-children SCORE", ":", str(round(is_children_log_likelihood,3))))
        '''
        if (abs(is_male_log_likelihood-is_female_log_likelihood)>0.1)or (vector[0] >180 or vector[0]<150):
        '''
        if is_male_log_likelihood > is_female_log_likelihood: 
            if is_male_log_likelihood>is_children_log_likelihood:
                winner="male"
            else:
                winner="children"
        else:
            if is_female_log_likelihood>is_children_log_likelihood:
                winner="female"
            else:
                winner="children"
        

        return winner,is_female_log_likelihood,is_male_log_likelihood,is_children_log_likelihood

def identy():
    dir="C:\\Users\\user\\SpeakerRecognition_tutorial_\\feat_logfbank_nfilt40\\test\\voice"
    dir_=os.path.join(dir,"test.wav")
    dir__=os.path.join(dir,"enroll.wav")
    gender_identifier = GenderIdentifier(dir_, dir__, "females.gmm", "males.gmm")
    gender_identifier.process()

if __name__== "__main__":
    gender_identifier = GenderIdentifier("C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\Code\\TestingData_\\nonolds_female^","C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\Code\\TestingData_\\nonolds_males","C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\Code\\TestingData_\\children^", "C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\Code\\TestingData_\\olds_female^","C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\Code\\TestingData_\\olds_male^","C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\Code\\nonoldfemales_praat.gmm", "C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\Code\\nonoldmales_praat.gmm","C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\Code\\olds_praat_.gmm","C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\Code\\nonolds_praat_.gmm","C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\Code\\childrens_praat.gmm"
    ,"C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\Code\\oldfemales_praat.gmm","C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\Code\\oldmales_praat.gmm","C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\Code\\nonoldfemale.gmm","C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\Code\\nonoldsmale.gmm","C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\Code\\nonoldfemales_mfcc_re.gmm","C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\Code\\nonoldmales_mfcc_re.gmm","C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\Code\\childrens_mfcc_.gmm","C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\Code\\oldfemales_mfcc_.gmm","C:\\Users\\user\\SpeakerRecognition_tutorial_\\Voice-based-gender-recognition-master\\Code\\oldmales_mfcc_.gmm")
    gender_identifier.oldageprocess()
