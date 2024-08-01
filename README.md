# News Quotation Extractor
2024.07.24 UPDATE 增加大语言模型版本。
2022.01.15 UPDATE 新增多文本共现计算功能。



## 大语言模型完成引语抽取
说明：使用 LLM 完成引语提取和引语归因任务。

输入样例：processed_en_news.json

输出样例：out.json


运行一次：

```
python llm.py
```

程序将读入processed_en_news.json目录下的所有输入，处理后以out.json输出。

每个JSON可以包含多个新闻文本。


JSON 字段含义：

| 字段  |   含义  |
| ---- | ---- |
|   articleNum   |  文件名    |
|   title   |  新闻标题    |
|   time   |  新闻发表时间    |
|   source   |  新闻来源    |
|   content   |  新闻正文   |
|   quote   |  引语数组，内含多个引语条目    |
|   quoteSpeakerCharOffsetsFirst   | 说话人字符串在正文的偏移量     |
|   quoteSpeakerCharOffsetsSecond   | 说话人字符串在正文的偏移量      |
|   quotation   | 引语正文     |
|   quoteCharOffsetsFirst   | 引语字符串在正文的偏移量     |
|   quoteCharOffsetsSecond   | 引语字符串在正文的偏移量     |
|   mention   | 说话人的名字     |



## 小模型引语流水线

说明：完成引语提取、引语归因、实体链接和共现计算任务。引语提取和引语归因采用一个 fine-tunedBERT 模型，实体链接采用 neuralcoref 开源工具包完成。

要求：Python 3、Anaconda 环境
（使用其他虚拟环境需要修改setup.sh）

配置：
```
bash setup.sh
```

模型参数：将提供的模型checkpoint下载到 models 目录下。

输入/输出样例：见 input 和 output 目录。每个json文件是一篇新闻，每个输入对应一个输出。

运行一次：

```
python main.py
```
程序将读入input目录下的所有输入，处理后以相同的文件名输出到output目录中。


JSON 字段含义：

| 字段  |   含义  |
| ---- | ---- |
|   articleNum   |  文件名    |
|   title   |  新闻标题    |
|   time   |  新闻发表时间    |
|   source   |  新闻来源    |
|   content   |  新闻正文（**必须用\n分割**）    |
|   quote   |  引语数组，内含多个引语条目    |
|   quoteSpeakerCharOffsetsFirst   | 说话人字符串在正文的偏移量     |
|   quoteSpeakerCharOffsetsSecond   | 说话人字符串在正文的偏移量      |
|   quotation   | 引语正文     |
|   quoteCharOffsetsFirst   | 引语字符串在正文的偏移量     |
|   quoteCharOffsetsSecond   | 引语字符串在正文的偏移量     |
|   SegmentOffset   | 引语所在段在正文的偏移量     |
|   Type   | 提取引语的方式</br>例如：说话人在引语左侧</br>说话人在引语右侧     |
|   corefMention   | 说话人经过指代消解后的名字     |
|   corefOffsetBegin   | 说话人经过指代消解后的字符串在正文的偏移量     |
|   corefOffsetEnd   | 说话人经过指代消解后的字符串在正文的偏移量     |
|   corefStatus   | 指代消解是否成功     |
|   mention   | 说话人经过实体链接后的名字     |
|   links   | 说话人的维基百科链接     |
|   mentionID   | 说话人的维基数据ID     |
|   mentionSpan   | 被指代消解的字符串</br>例如：he、she、it     |
|   mentionAbout   | 说话人在维基百科的介绍</br>例如：当代著名作家     |
|   mentionProperty   | 说话人具有的属性</br>例如：人、组织、机构     |
|   linkStatus   | 实体链接是否成功     |

共现计算：

运行一次：

```
python co-occur.py
```



输入：

程序处理的数据是由str组成的list，例如：

```
input=["This is a sentence.","Country road, take me home.","Never gonna give you up."]
```

通过以下方式运行程序：

```
print(multi_text_co_occur(input,300),id2entity)
```

其中，input是上述文本列表，300是计算共现时的窗口大小，id2entity是实体ID和实体本身的对应关系。

<u>**详见co-occur.py中给出的纽约时报的例子。**</u>



输出：

1. 不同实体的共现次数统计，例如：

```
{(1316, 492313): 8, 
(22686, 170581): 8, 
(22686, 355522): 16, 
(22686, 492313): 10}
```

代表1316号实体和492313共现8次，22686号实体和170581号实体共现8次，以此类推，可以计算人物关系图。

2. 实体ID和实体本身的对应关系id2entity，例如：

```
{492313: {'label': 'Park Jin-young', 'link': 'https://en.wikipedia.org/wiki/Park_Jin-young', 'id': '492313', 'span': 'Jr.', 'description': 'South Korean singer and actor born on September 22, 1994', 'super_entities': ('human',)},

1316: {'label': 'Saddam Hussein', 'link': 'https://en.wikipedia.org/wiki/Saddam_Hussein', 'id': '1316', 'span': 'Saddam Hussein', 'description': 'Iraqi politician and President', 'super_entities': ('human',)}, 

170581: {'label': 'Nancy Pelosi', 'link': 'https://en.wikipedia.org/wiki/Nancy_Pelosi', 'id': '170581', 'span': 'Nancy Pelosi', 'description': 'Speaker of the United States House of Representatives', 'super_entities': ('human',)}, 

22686: {'label': 'Donald Trump', 'link': 'https://en.wikipedia.org/wiki/Donald_Trump', 'id': '22686', 'span': 'President Trump', 'description': '45th and current president of the United States', 'super_entities': ('human', 'billionaire')}, 

355522: {'label': 'Mitch McConnell', 'link': 'https://en.wikipedia.org/wiki/Mitch_McConnell', 'id': '355522', 'span': 'Mitch McConnell', 'description': 'United States Senator from Kentucky', 'super_entities': ('human',)}, 

359442: {'label': 'Bernie Sanders', 'link': 'https://en.wikipedia.org/wiki/Bernie_Sanders', 'id': '359442', 'span': 'Bernie Sanders', 'description': 'United States Senator from Vermont', 'super_entities': ('human',)}, 

33190271: {'label': 'Mark T. Esper', 'link': 'https://en.wikipedia.org/wiki/Mark_T._Esper', 'id': '33190271', 'span': 'Mark T. Esper', 'description': 'U.S. Secretary of Defense', 'super_entities': ('human',)}}
```

代表492313号实体对应的是Park Jin-young，及该实体的相关信息。
