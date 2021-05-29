# -*- coding:utf-8 -*-
from transformers import AlbertForTokenClassification
from torch.utils.data.dataloader import DataLoader
import torch
from transformers import AutoTokenizer
import numpy as np
import pickle
from genre.trie import Trie
from genre.hf_model import GENRE
import logging
import os
import json

device = "cuda"
eval_batch_size = 16

# 初始化logger
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(console)

logger.info("加载引语模型")
tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
model = AlbertForTokenClassification.from_pretrained(
    "models/checkpoint-1200/", num_labels=8).to(device).eval()


logger.info("程序初始化，加载实体表")
# 加载模型参数
with open("models/kilt_titles_trie_dict.pkl", "rb") as f:
    trie = Trie.load_from_dict(pickle.load(f))
logger.info("加载实体链接模型")
EDmodel = GENRE.from_pretrained(
    "models/hf_entity_disambiguation_aidayago").to(device).eval()
memo = {}

# 构造数据集


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, dic):
        self.dic = dic

    def __getitem__(self, idx):
        item = {"input_ids": self.dic['input_ids'][idx],
                "attention_mask": self.dic['attention_mask'][idx]}
        return item

    def __len__(self):
        return len(self.dic['input_ids'])


def predict(dataset):
    loader = DataLoader(dataset, batch_size=eval_batch_size)
    model.eval()
    output = np.array([])
    for step, batch in enumerate(loader):
        with torch.no_grad():
            outputs = model(**batch).logits.cpu().numpy()
            output = np.append(output, outputs)
    return np.reshape(output, (-1, 511, 8))


# 按换行符分段


def segment(txt):
    pos = 0
    segs = []
    offsets = []
    while pos < len(txt):
        if txt[pos] == '\n':
            while pos < len(txt) and txt[pos] == '\n':
                pos += 1
        else:
            left = pos
            while pos < len(txt) and txt[pos] != '\n':
                pos += 1
            segs.append(txt[left:pos])
            offsets.append((left, pos))
    return segs, offsets

# 引语提取


def displayFormatResult(input_id, attention, prediction, offset_map, overall_offset):
    result = []
    predict = np.argmax(prediction, axis=2)
    for i in range(len(input_id)):
        tuple_list = []
        tuple_type = []
        j = 0
        while j < len(predict[i]):
            if attention[i][j] == 0:
                break
            if predict[i][j] == 3:
                left = j
                while j < len(predict[i]) and predict[i][j] == 3:
                    if attention[i][j] == 0:
                        break
                    j += 1
                tuple_list.append((left, j))
                tuple_type.append(0)  # 说话人

            elif j < len(predict[i]) and predict[i][j] == 0:
                j += 1

            elif predict[i][j] == 1 or predict[i][j] == 2:
                left = j
                while j < len(predict[i]) and (predict[i][j] == 1 or predict[i][j] == 2):
                    if attention[i][j] == 0:
                        break
                    j += 1
                tuple_list.append((left, j))
                tuple_type.append(1)  # 匿名

            elif predict[i][j] == 4 or predict[i][j] == 5:
                left = j
                while j < len(predict[i]) and (predict[i][j] == 4 or predict[i][j] == 5):
                    if attention[i][j] == 0:
                        break
                    j += 1
                tuple_list.append((left, j))
                tuple_type.append(2)  # 向左

            elif predict[i][j] == 6 or predict[i][j] == 7:
                left = j
                while j < len(predict[i]) and (predict[i][j] == 6 or predict[i][j] == 7):
                    if attention[i][j] == 0:
                        break
                    j += 1
                tuple_list.append((left, j))
                tuple_type.append(3)  # 向右

        for t in range(len(tuple_list)):
            if tuple_type[t] == 0:
                pass

            elif tuple_type[t] == 1:
                if False:
                    result.append({"mentionRaw": "Unknown",
                                   "quoteSpeakerCharOffsetsFirst": -1,
                                   "quoteSpeakerCharOffsetsSecond": -1,
                                   "quotation": str(tokenizer.decode(input_id[i][tuple_list[t][0]:tuple_list[t][1]])),
                                   "quoteCharOffsetsFirst": str(offset_map[i][tuple_list[t][0]][0]+overall_offset[i][0]),
                                   "quoteCharOffsetsSecond": str(offset_map[i][tuple_list[t][1]-1][1]+overall_offset[i][0]),
                                   "SegmentOffset": str(overall_offset[i][0]),
                                   "Type": "Anonymous"})

            elif tuple_type[t] == 2:
                back = t
                while back >= 0 and tuple_type[back] != 0:
                    back -= 1
                if back < 0:
                    if False:
                        result.append({"mentionRaw": "Unknown",
                                       "quoteSpeakerCharOffsetsFirst": -1,
                                       "quoteSpeakerCharOffsetsSecond": -1,
                                       "quotation": str(tokenizer.decode(input_id[i][tuple_list[t][0]:tuple_list[t][1]])),
                                       "quoteCharOffsetsFirst": str(offset_map[i][tuple_list[t][0]][0]+overall_offset[i][0]),
                                       "quoteCharOffsetsSecond": str(offset_map[i][tuple_list[t][1]-1][1]+overall_offset[i][0]),
                                       "SegmentOffset": str(overall_offset[i][0]),
                                       "Type": "TowardsLeftFailed"})
                else:
                    if tuple_list[t][1]-tuple_list[t][0] > 6:
                        mentionRaw=tokenizer.decode(input_id[i][tuple_list[back][0]:tuple_list[back][1]])
                        if len(mentionRaw) > 4:
                            result.append({"mentionRaw": mentionRaw,
                                        "quoteSpeakerCharOffsetsFirst": str(offset_map[i][tuple_list[back][0]][0]+overall_offset[i][0]),
                                        "quoteSpeakerCharOffsetsSecond": str(offset_map[i][tuple_list[back][1]-1][1]+overall_offset[i][0]),
                                        "quotation": tokenizer.decode(input_id[i][tuple_list[t][0]:tuple_list[t][1]]),
                                        "quoteCharOffsetsFirst": str(offset_map[i][tuple_list[t][0]][0]+overall_offset[i][0]),
                                        "quoteCharOffsetsSecond": str(offset_map[i][tuple_list[t][1]-1][1]+overall_offset[i][0]),
                                        "SegmentOffset": str(overall_offset[i][0]),
                                        "Type": "TowardsLeftSucceeded"})

            elif tuple_type[t] == 3:
                after = t
                while after < len(tuple_type) and tuple_type[after] != 0:
                    after += 1
                if after >= len(tuple_type):
                    if False:
                        result.append({"mentionRaw": "Unknown",
                                       "quoteSpeakerCharOffsetsFirst": -1,
                                       "quoteSpeakerCharOffsetsSecond": -1,
                                       "quotation": str(tokenizer.decode(input_id[i][tuple_list[t][0]:tuple_list[t][1]])),
                                       "quoteCharOffsetsFirst": str(offset_map[i][tuple_list[t][0]][0]+overall_offset[i][0]),
                                       "quoteCharOffsetsSecond": str(offset_map[i][tuple_list[t][1]-1][1]+overall_offset[i][0]),
                                       "SegmentOffset": str(overall_offset[i][0]),
                                       "Type": "TowardsRightFailed"})
                else:
                    if tuple_list[t][1]-tuple_list[t][0] > 6:
                        mentionRaw=tokenizer.decode(input_id[i][tuple_list[after][0]:tuple_list[after][1]])
                        if len(mentionRaw) > 4:
                            result.append({"mentionRaw": mentionRaw,
                                        "quoteSpeakerCharOffsetsFirst": str(offset_map[i][tuple_list[after][0]][0]+overall_offset[i][0]),
                                        "quoteSpeakerCharOffsetsSecond": str(offset_map[i][tuple_list[after][1]-1][1]+overall_offset[i][0]),
                                        "quotation": tokenizer.decode(input_id[i][tuple_list[t][0]:tuple_list[t][1]]),
                                        "quoteCharOffsetsFirst": str(offset_map[i][tuple_list[t][0]][0]+overall_offset[i][0]),
                                        "quoteCharOffsetsSecond": str(offset_map[i][tuple_list[t][1]-1][1]+overall_offset[i][0]),
                                        "SegmentOffset": str(overall_offset[i][0]),
                                        "Type": "TowardsRightSucceeded"})
    return result

# 实体链接


def getEntity(txt):
    if txt in memo:
        return memo[txt]
    else:
        if type(txt) != str:
            raise Exception("说话人不是字符串")
        else:
            sentences = ["[START_ENT] "+txt+" [END_ENT]"]
            result = EDmodel.sample(
                sentences, prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()))
            memo[txt] = (result[0][0]['text'], result[0][0]['logprob'].item())
            return result[0][0]['text'], result[0][0]['logprob'].item()

# 处理单个文件


def extractText(txt):
    torch.cuda.empty_cache() 
    segs, offsets = segment(txt)
    res = tokenizer(segs, padding='max_length', max_length=511,
                    truncation=True, return_offsets_mapping=True, return_tensors="pt").to(device)
    res_data = TestDataset(res)
    pred_res = predict(res_data)
    res = res.to('cpu')
    middle_result = displayFormatResult(
        res['input_ids'].numpy(), res["attention_mask"].numpy(), pred_res, res['offset_mapping'].numpy(), offsets)
    # logger.info("引语提取结束，开始实体链接")
    for i in range(len(middle_result)):
        if type(middle_result[i]['mentionRaw']) != str:
            logger.warning("说话人不是字符串："+str(middle_result[i]))
        else:
            linked = getEntity(middle_result[i]['mentionRaw'])
            middle_result[i]['mention'] = linked[0]
            middle_result[i]['mentionLinkLogProb'] = linked[1]
            middle_result[i]['links'] = "https://en.wikipedia.org/wiki/" + \
                linked[0].strip().replace(' ', "_")
    return middle_result

# 按文件夹处理文件


def folderProcess(folder_path, output_folder_path):
    files = sorted(os.listdir(folder_path))
    for i in files:
        try:
            if i.endswith('.json'):
                with open(os.path.join(folder_path, i), encoding='utf-8') as f:
                    data = json.loads(f.read())
                    data['content'] = data['content'].replace("''", "\"").replace("„", "\"").replace("“", "\"").replace("‟", "\"").replace("”", "\"").replace(
                        "〝", "\"").replace("〞", "\"").replace("〟", "\"").replace("‘", "'").replace("’", "'").replace("‛", "'").replace(",", ",").replace("—", "-")
                    quote_dict = extractText(data['content'])
                    data['quote'] = quote_dict
                    with open(os.path.join(output_folder_path, i), 'w', encoding="utf-8") as fw:
                        json.dump(data, fw)
                logger.info("输出文件 "+str(i))
            else:
                logger.warning("忽略文件 "+str(i))
        except Exception:
            logger.exception("在处理文件 "+i+" 时异常")


logger.info("加载完成，开始处理文件")

folderProcess("testin",
              "testout")
