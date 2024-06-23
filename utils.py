import torch
import numpy as np

#custom pytorch dataset meant for persuasion data
class PersuasionDataset(torch.utils.data.Dataset):
    def __init__(self, raw_text, encodings, labels, feature_masks):
        self.encodings = encodings
        self.labels = labels
        self.raw_text = raw_text
        self.feature_masks = feature_masks

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['raw_text'] = self.raw_text[idx]
        item['labels'] = self.labels[idx]
        item['feature_masks'] = self.feature_masks[idx]
        return item

    def __len__(self):
        return len(self.labels)

# get feature mask (for input texts) using offset_mapping returned by hf tokenizers
def get_feature_mask_from_offsets(offset_mapping):
    # offset mapping obtained from hf tokenzier expected to be L x 2
    feature_map = []
    cur_word_num = 0
    end_reached = False  # in case of padding tokens
    for i, offset in enumerate(offset_mapping):
        if end_reached: #if end reached then only have padding tokens
            feature_map.append(cur_word_num + 1)
            continue

        if offset[0] == 0 and offset[1] == 0 and cur_word_num == 0:  # if begin token
            feature_map.append(cur_word_num)
            cur_word_num += 1
            continue
        elif offset[0] == 0 and offset[1] == 0 and cur_word_num > 0:  # if end token
            cur_word_num += 1
            feature_map.append(cur_word_num)
            end_reached = True
            continue

        else:  # other tokens
            if offset[0] == offset_mapping[i - 1][1]:  # if current token a continuation of the previous word
                feature_map.append(cur_word_num)
            else:  # if current token the start of a NEW word
                cur_word_num += 1
                feature_map.append(cur_word_num)

    return feature_map