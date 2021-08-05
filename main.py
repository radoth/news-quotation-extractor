# -*- coding:utf-8 -*-
from transformers import AlbertForTokenClassification
from torch.utils.data.dataloader import DataLoader
import torch
from transformers import AutoTokenizer
import numpy as np
import logging
import os
import json
import time
import spacy
import neuralcoref
from spacyEntityLinker import EntityLinker

device = "cuda"
eval_batch_size = 16

pronouns = ["i", "me", "you", "he", "him", "she", "her", "it", "we", "us", "they", "them", "my", "mine", "your", "yours",
            "his", "her", "hers", "its", "our", "ours", "their", "theirs"]

# 初始化logger
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(console)

logger.info("加载引语模型")
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
model = AlbertForTokenClassification.from_pretrained(
    "models/checkpoint-1400/", num_labels=8).to(device).eval()

logger.info("加载指代消解模型")
nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)

logger.info("加载实体链接模型")
nlp2 = spacy.load('en_core_web_sm')
entityLinker = EntityLinker()
nlp2.add_pipe(entityLinker, last=True, name="entityLinker")


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
                        mentionRaw = tokenizer.decode(
                            input_id[i][tuple_list[back][0]:tuple_list[back][1]])
                        if len(mentionRaw) > 1:
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
                        mentionRaw = tokenizer.decode(
                            input_id[i][tuple_list[after][0]:tuple_list[after][1]])
                        if len(mentionRaw) > 1:
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
entity_linking_sum = 0


def getEntity(txt):
    global entity_linking_sum
    count = time.time()
    doc = nlp2(txt)

    if len(doc._.linkedEntities) == 0:
        entity_linking_sum += (time.time()-count)
        return None
    else:
        entity_linking_sum += (time.time()-count)
        return {
            "label": str(doc._.linkedEntities[0].get_label()),
            "id": str(doc._.linkedEntities[0].get_id()),
            "span": str(doc._.linkedEntities[0].get_span()),
            "description": str(doc._.linkedEntities[0].get_description()),
            "super_entities": tuple(str(i) for i in doc._.linkedEntities[0].get_super_entities(limit=10))
        }


# 指代消解
def getCoreference(parsed_doc, speaker_begin, speaker_end):
    sent_span = parsed_doc.char_span(
        speaker_begin, speaker_end, alignment_mode="expand")
    if sent_span._.is_coref:
        target = sent_span._.coref_cluster.main
        return target.string, target.start_char, target.end_char, True
    else:
        return sent_span.string, sent_span.start_char, sent_span.end_char, False


# 处理单个文件


def extractText(txt):
    segs, offsets = segment(txt)
    res = tokenizer(segs, padding='max_length', max_length=511,
                    truncation=True, return_offsets_mapping=True, return_tensors="pt").to(device)
    token_time = time.time()
    res_data = TestDataset(res)
    pred_res = predict(res_data)
    res = res.to('cpu')
    extract_time = time.time()
    middle_result = displayFormatResult(
        res['input_ids'].numpy(), res["attention_mask"].numpy(), pred_res, res['offset_mapping'].numpy(), offsets)
    # logger.info("引语提取结束，开始实体链接")
    process_time = time.time()
    parsed_doc = nlp(txt)
    parsing_time = time.time()
    for i in range(len(middle_result)):
        if type(middle_result[i]['mentionRaw']) != str:
            logger.warning("说话人不是字符串："+str(middle_result[i]))

        elif int(middle_result[i]['quoteSpeakerCharOffsetsFirst']) >= 0 and int(middle_result[i]['quoteSpeakerCharOffsetsSecond']) >= 0:
            target_name, target_start, target_end, coref_status = getCoreference(parsed_doc, int(
                middle_result[i]['quoteSpeakerCharOffsetsFirst']), int(middle_result[i]['quoteSpeakerCharOffsetsSecond']))
            middle_result[i]['corefMention'] = target_name
            middle_result[i]['corefOffsetBegin'] = str(target_start)
            middle_result[i]['corefOffsetEnd'] = str(target_end)
            middle_result[i]['corefStatus'] = str(coref_status)

            linking_result = getEntity(target_name)
            if linking_result is not None:
                middle_result[i]['mention'] = linking_result['label']
                middle_result[i]['links'] = "https://en.wikipedia.org/wiki/" + \
                    linking_result['label'].strip().replace(' ', "_")
                middle_result[i]['mentionID'] = linking_result['id']
                middle_result[i]['mentionSpan'] = linking_result['span']
                middle_result[i]['mentionAbout'] = linking_result['description']
                middle_result[i]['mentionProperty'] = '&'.join(
                    linking_result['super_entities'])
                middle_result[i]['linkStatus'] = "True"
            else:
                middle_result[i]['mention'] = target_name
                middle_result[i]['linkStatus'] = "False"
        else:
            raise Exception("无效的引语区间")
    linking_time = time.time()
    return middle_result, token_time, extract_time, process_time, parsing_time, linking_time

# 按文件夹处理文件


def folderProcess(folder_path, output_folder_path):
    global entity_linking_sum
    files = sorted(os.listdir(folder_path))
    files_len = len(files)
    for no, i in enumerate(files):
        if no % 100 == 0:
            torch.cuda.empty_cache()
            logger.info("清除 CUDA Cache")
        try:
            if i.endswith('.json'):
                with open(os.path.join(folder_path, i), encoding='utf-8') as f:
                    start_time = time.time()
                    entity_linking_sum = 0.0
                    data = json.loads(f.read())
                    data['content'] = data['content'].replace("''", "\"").replace("„", "\"").replace("“", "\"").replace("‟", "\"").replace("”", "\"").replace(
                        "〝", "\"").replace("〞", "\"").replace("〟", "\"").replace("‘", "'").replace("’", "'").replace("‛", "'").replace(",", ",").replace("—", "-")
                    quote_dict, token_time, extract_time, process_time, parsing_time, linking_time = extractText(
                        data['content'])
                    data['quote'] = quote_dict
                    with open(os.path.join(output_folder_path, i), 'w', encoding="utf-8") as fw:
                        json.dump(data, fw)
                logger.info("进度：{:.2f}%\t输出文件 {}\t总耗时{:.0f}ms\t分词耗时{:.0f}ms\t提取耗时{:.0f}ms\t处理耗时{:.0f}ms\t指代消解耗时{:.0f}ms\t实体链接耗时{:.0f}ms\t实体链接实际耗时{:.0f}ms".format(no/files_len*100, i, (time.time()-start_time)
                            * 1000, (token_time-start_time)*1000, (extract_time-token_time)*1000, (process_time-extract_time)*1000, (parsing_time-process_time)*1000, (linking_time-parsing_time)*1000, entity_linking_sum*1000))
            else:
                logger.warning("忽略文件 "+str(i))
        except Exception:
            logger.exception("在处理文件 "+i+" 时异常")


logger.info("加载完成，开始处理文件")

folderProcess("input",
              "output")
