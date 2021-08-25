clc 
clear  
close all 
 
path_name = 'E:\명령어 음성(일반남녀) 데이터\Training\3.키오스크_원천_1_명령어(일반)_training\일반\training\키오스크\1\script1_j_0044';
lists = dir(path_name);
n = length(lists);
for k = 3:n
    file_name = strcat(path_name,'\',lists(k).name);
    [audioIn,fs]=audioread(file_name);
    Fs_new = 16000;
melSpectrogram(audioIn,Fs_new);
colorbar('off')
axis off;

 

%Saving output to new folder 
pathname = 'E:\명령어 음성(일반남녀) 데이터\Training\[med]commands_MFCC';%output folder 
out=strcat('mel_med_test_command_mfcc_',num2str(k-2),'.png'); %output data name 
matdata = fullfile(pathname,out);  


saveas(gcf,matdata);
end