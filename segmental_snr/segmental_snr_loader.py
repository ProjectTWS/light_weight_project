from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from glob import glob
import numpy as np
import librosa
import natsort
import csv
import torch
from cls_feat_extract import feature_extractor
from segmental_snr_loss import segmental_snr

def samsung_inference_dataloader_ASC(feature_options, partition):
        return DataLoader(
            samsung_inference_dataset(feature_options, partition),
            batch_size=feature_options.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True
        )


class samsung_inference_dataset(Dataset):
    def __init__(self, feature_options, partition):
        self.frame_length = feature_options.frame_length

        self.file_dir='{}/mix/{}/'.format(feature_options.data_path, partition)
        
       
        self.dir_list = glob('{}/mix/{}/**/*_events'.format(feature_options.data_path, partition))
        self.dir_list = natsort.natsorted(self.dir_list, reverse=False)
     
        self.partition = partition
        f = open("./inference/result.csv", "r")
        reader = csv.reader(f)
        lines = []
   
        for line in reader:
            lines.append(line)
        lines.pop(0)
        self.lines = natsort.natsorted(lines, reverse=False)
        f.close()
        self.feat_cls = feature_extractor()
        
        self.target_sound_type=['Vehicle_horn_and_car_horn_and_honking',
                                'Police_car_siren',
                                'speech',
                                'Fire_alarm',]
        # Scene and target sound dict
        # Acoustic scene -> target sound
        self.scene_target = {
                            'Bus': ['Vehicle_horn_and_car_horn_and_honking', 'Police_car_siren', 'speech',],
                            'Metro': ['Fire_alarm', 'speech',],
                            'Metro_station': ['Fire_alarm', 'speech',],
                            'Park': ['speech',],
                            'Street_traffic': ['Vehicle_horn_and_car_horn_and_honking', 'Police_car_siren',],
                            }

        # self.scene_onehot = {'Bus': [1, 0, 0, 0, 0],
        #                     'Metro': [0, 1, 0, 0, 0],
        #                     'Metro_station': [0, 0, 1, 0, 0],
        #                     'Park': [0, 0, 0, 1, 0],
        #                     'Street_traffic': [0, 0, 0, 0, 1]}
    def get_txt_data(self, txt_name, target_sound):
        txt_data=[]
        with open(txt_name, 'r') as f:
            
            while True:
                line=f.readline().rstrip()

                if not line:
                    break

                line=line.split()

                if line[-1] in target_sound:
                    txt_data.append(line)
                    

   
        return txt_data

    def get_batch_data(self, dir_name, index):
        """
        Args:
            dir_name:
        Returns: y: mixture
                 s: target sounds
                 c: condition vector (one-hot or embedding)
        """
        # check this scene        
        ans_scene = dir_name.split('/')[-2]        
        scene = self.lines[index][1]
        fn = dir_name.split('/')[-1].replace('_events','.wav')
        
        
      
    

        # get the target sound
        target_sound = self.scene_target[ans_scene]

        segmental_txt=dir_name.replace('_events','.txt')
        txt_data=self.get_txt_data(segmental_txt, target_sound)
   
        source_list = glob(dir_name + '/*.wav')
        
        # print(source_list)
        
        x_list = []
        s_list = []
        #exist_target = False
        for _, source in enumerate(source_list):
            x, _ = librosa.load(source, mono=False, sr=None)
            x_list.append(x)
            if source.split('/')[-1][12:-4] in target_sound:
                s_list.append(x)
                #exist_target = True

        # sound mixture
        y = np.array(x_list).sum(axis=0) # [n_src, sr*10] -> [sr*10]
        # target sound
        if len(s_list) == 0:
            s = np.zeros_like(y)
        else:
            s = np.array(s_list).sum(axis=0)

        # condition vector
        # c = np.array(self.scene_onehot[scene])
        # Check if the target sound exist in directory
        # if exist_target:
        #     with open(dir_name.replace('_events', '.txt')) as f:
        #         lines = f.readlines()
        #         for line in lines:
        #             if target_sound == line.split('\t')[-1]:
        #                 print('h')
        # else:
            # cut random duration
            # predefined frame length
        # print(y.shape)
        # print(s.shape)
        # print(c.shape)
        # exit()

      

        
        wav, _ = self.norm(y)      # [Batch, sr*10]
        logmel = self.feat_cls.wav2logmel(wav) # [nmels, nframe]
        return y, s, torch.tensor(logmel), [fn, ans_scene], txt_data


    def __getitem__(self, index):
        dir_name = self.dir_list[index]
        return self.get_batch_data(dir_name, index)


    def __len__(self):
        return len(self.dir_list)

    def norm(self, wav, target_dbfs = -25):
        rms = (wav**2).mean()**0.5
        scalarclean = 10**(target_dbfs / 20) / rms
        return wav * scalarclean, scalarclean



if __name__ == '__main__':
    from attrdict import AttrDict
    feature_option = {'batch_size': 1, 'frame_length':10, 'data_path':'/root/harddisk1/Dataset/new_dataset_16k'}
    feature_option = AttrDict(feature_option)
    ### only using in batch size 1
    loader=samsung_inference_dataloader_ASC(feature_option, 'tt')
    loss_func=segmental_snr

    for i in loader:

        y,s, logmel, etc, txt=i # y: mixture, s: target
        snri=loss_func(y, s,y, txt)
        print(snri)


        audfopau=input()
