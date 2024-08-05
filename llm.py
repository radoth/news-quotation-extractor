from openai import OpenAI
import re,json
from tqdm import tqdm
client = OpenAI(base_url="http://localhost:18888/v1",api_key="123")

def extract_quoted_strings(text):
  # 用正则表达式提取引号内的句子
    pattern = re.compile(r'["“”](.*?)["“”]')
    matches = pattern.findall(text)
    quoted_strings = [match for match in matches]
    filter_quotes=[]
    for i in quoted_strings:
        if i.count(" ")>3:
            if "." in i or "," in i or "?" in i or "!" in i:
                filter_quotes.append(i)
    return filter_quotes

def extract_numbered_list(text):
  # 从序号列表中提取对应的说话人
    pattern = re.compile(r'\d+\.\s+([^\n]+)')
    matches = pattern.findall(text)
    return matches

def process_a_text(text):
    # 处理一个新闻
    text=text.strip()

    request=f"""Attribute the given quotations in the following text to the speaker. Each speaker must be a continuous substring within the text.

    Text:
    {text}

    Quotations:
    """

    quotations = extract_quoted_strings(request)

    for number, quote in enumerate(quotations):
        request += f"{number+1}. {quote}\n"

    request += """

    Answer Examples:
    1. Wang Dong
    2. Xi Jinping
    3. Joe Biden
    4. Donald Trump
    5. Unknown
    6. Not a quotation

    Answer:
    """

    # print(request)
    # exit(1)
    completion = client.chat.completions.create(
      model="openchat_3.5",
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": request}
      ]
    )

    result_list = extract_numbered_list(completion.choices[0].message.content.strip())
    # print(result_list)
    # print(quotations)
    result_dicts = []
    for a,b in zip(quotations,result_list):
        if "unknown" in b.lower() or "not a quotation" in b.lower():
            print(b)
            continue
        if b.lower() in text.lower():
            result_dicts.append({
                "mentionRaw": b,
                "quoteSpeakerCharOffsetsFirst": text.lower().find(b.lower()),
                "quoteSpeakerCharOffsetsSecond": text.lower().find(b.lower())+len(b),
                "quotation": a}
            )
        else:
            print(b)

    return result_dicts

def handle_json_file(input_path,output_path):
    # 按照指定格式处理文件
    input_data = open(input_path,encoding="utf-8")
    output_list = []
    for raw in tqdm(input_data):
        i = json.loads(raw)
        try:
          i["quote"]=process_a_text(i["content"])
        except Exception as e:
          print(e)
        output_list.append(i)
        with open(output_path, 'w',encoding="utf-8") as file:
            for entry in output_list:
                file.write(json.dumps(entry) + '\n')

# 在一个JSON文件中添加一个quote字段
handle_json_file("/home/thuair/processed_en_news.json","/home/thuair/output.json")
