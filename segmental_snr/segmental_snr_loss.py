import torch
import asteroid

if __name__ == '__main__':
    from attrdict import AttrDict
    feature_option = {'batch_size': 1, 'frame_length':4}
    feature_option = AttrDict(feature_option)

    dl = samsung_dataloader(feature_option, 'tr')
    for i, data in enumerate(dl):
        print(data)