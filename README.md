# QuoteBERT

要求：Python 3

模型：下载到 models 目录下。

输入/输出样例：见 input 和 output 目录。

运行：
```
bash setup.sh
python main.py
```

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

