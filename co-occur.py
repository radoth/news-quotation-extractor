import spacy
import neuralcoref
from spacyEntityLinker import EntityLinker
import logging
import time

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

# 加载模型
logger.info("加载指代消解模型")
nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)

logger.info("加载实体链接模型")
nlp2 = spacy.load('en_core_web_sm')
entityLinker = EntityLinker()
nlp2.add_pipe(entityLinker, last=True, name="entityLinker")

# 实体链接计时
entity_linking_sum = 0

# 实体ID的映射表
id2entity={}

def is_in_window(window_size,a,b):
    return abs(a[0]-b[0])<=window_size

def entity_linking(txt):
    """
    对某个字符串（如“Donald Trump”）进行实体链接，返回对应的实体ID。
    """
    global entity_linking_sum,id2entity
    count = time.time()
    doc = nlp2(txt)

    if len(doc._.linkedEntities) == 0:
        entity_linking_sum += (time.time()-count)
        return None
    else:
        entity_linking_sum += (time.time()-count)

        if "human" in tuple(str(i) for i in doc._.linkedEntities[0].get_super_entities(limit=10)):
            id2entity[int(doc._.linkedEntities[0].get_id())]={
                "label": str(doc._.linkedEntities[0].get_label()),
            "link":"https://en.wikipedia.org/wiki/" + \
                    str(doc._.linkedEntities[0].get_label()).strip().replace(' ', "_"),
            "id": str(doc._.linkedEntities[0].get_id()),
            "span": str(doc._.linkedEntities[0].get_span()),
            "description": str(doc._.linkedEntities[0].get_description()),
            "super_entities": tuple(str(i) for i in doc._.linkedEntities[0].get_super_entities(limit=10))
            }
            return int(doc._.linkedEntities[0].get_id())
        else:
            return None



# 计算文本中实体的共现
def coref(txt,window_size):
    """
    
    对某段纯文本计算共现，返回多个tuple，每个tuple是两个实体的一次共现。

    """
    count=time.time()
    doc=nlp(txt)

    results={}

    co_occur_tuples=[]

    for i in doc._.coref_clusters:
        main_entity=entity_linking(str(i.main))
        if main_entity is not None:
            original=results.get(main_entity,list())
            original.extend([(j.start_char,j.end_char) for j in i.mentions])
            results[main_entity]=original
    
    for i in results:
        for j in results:
            if i != j:
                for i_mention in results[i]:
                    for j_mention in results[j]:
                        if is_in_window(window_size,i_mention,j_mention):
                            co_occur_tuples.append((min(i,j),max(i,j)))


    return co_occur_tuples


# 计算频度直方图
def freq_dist_calc(tuples):
    freqdist={}
    for i in tuples:
        freqdist[i]=freqdist.get(i,0)+1
    return freqdist

# 计算多个文本
def multi_text_co_occur(txt_list,window_size):
    """
    
    输入一个str的list，在window_size大小的窗口中计算共现，并将结果汇总返回

    """
    global id2entity
    co_occur_list=[]

    for no,txt in enumerate(txt_list):
        start=time.time()
        coref_result=coref(txt,window_size=window_size)
        co_occur_list.extend(coref_result)
        logger.info("编号为 {} 的文件处理完毕，耗时 {:.0f}ms，新增共现tuple {} 个，共出现实体 {} 个，文件摘要：".format(no,(time.time()-start)*1000,len(coref_result),len(id2entity))+txt[:50])


    return freq_dist_calc(co_occur_list)


# 用来测试的文本
txt="""
Trump, Shifting Arguments, Urges Swift Dismissal of Impeachment Charges
The president injected new instability into his looming impeachment trial as Speaker Nancy Pelosi warned that nothing could “erase” the stain on his presidency. Speaker Nancy Pelosi is preparing to transmit the impeachment articles to the Senate this week. Speaker Nancy Pelosi is preparing to transmit the impeachment articles to the Senate this week.Credit...Erin Schaff/The New York Times WASHINGTON — President Trump on Sunday injected fresh instability into final preparations for the Senate’s impeachment trial, suggesting that senators should dismiss the House’s charges of high crimes and misdemeanors against him outright rather than dignifying them with a full tribunal. That unexpected statement, arriving amid a flurry of tweets, not only appeared to put the president at odds with Republican Senate leaders moving toward a full trial but also contradicted Mr. Trump’s own words from just hours earlier, when he argued for a trial that would include as witnesses Democratic House leaders who are prosecuting him. “Many believe that by the Senate giving credence to a trial based on the no evidence, no crime, read the transcripts, ‘no pressure’ Impeachment Hoax, rather than an outright dismissal, it gives the partisan Democrat Witch Hunt credibility that it otherwise does not have,” Mr. Trump wrote on Twitter on Sunday afternoon. “I agree!” It was the latest instance of the president publicly vacillating between his apparent desire to make the charges disappear and to precipitate an extended spectacle in the Senate that would turn the tables and put his critics on trial. Hours before, Speaker Nancy Pelosi had warned that anything short of a full trial with new witnesses and evidence would abet Mr. Trump’s attempts to cover up wrongdoing. As the subject of the charges on trial, Mr. Trump has no direct say over how the Senate proceeds and, by all accounts, Republicans do not have the votes needed to dismiss the case as Mr. Trump suggested, at least not before hearing opening arguments.
But the remarks could put lawmakers from his party in a difficult position in the weeks ahead if they have to justify to constituents loyal to the president why they are even giving a hearing to a case he derides as a “sham.” And the zigzagging underscored the challenge facing Mr. Trump’s defense team and the Senate majority leader, Mitch McConnell of Kentucky, as they try to navigate competing interests just days before the trial is to begin. Mr. McConnell has said he intends to run a trial similar to the 1999 impeachment proceeding against President Bill Clinton, meaning both House prosecutors and Mr. Trump’s defense team will be given a chance to present their arguments. Mr. Trump’s running impeachment commentary came on a day when much of Washington was busy preparing for the trial, only the third such proceeding in American history. Ms. Pelosi has said the House will send the two articles of impeachment charging Mr. Trump with abuse of power and obstruction of Congress sometime this week, with a trial commencing as soon as Wednesday. Appearing Sunday on ABC’s “This Week” before Mr. Trump’s tweet, Ms. Pelosi told senators considering forgoing a full trial that “dismissing is a cover-up” and asserted that whatever the outcome, the trial could never “erase” the stain of impeachment from Mr. Trump’s record. Even as the speaker expressed optimism that the House’s three-month impeachment inquiry had collected enough evidence to win a conviction on its two charges, she implicitly acknowledged the difficulty Democrats would have in doing so. And she warned that the Senate proceeding might be little more than a cover-up if Republican senators did not agree to summon new witnesses and documents that Mr. Trump blocked from the House. “We have confidence in our case that it’s impeachable, and this president is impeached for life, regardless of any gamesmanship on the part of Mitch McConnell,” the speaker said, referring to Mr. McConnell. The House impeached Mr. Trump last month on two counts related to what Democrats concluded was a campaign by the president to pressure Ukraine to investigate his domestic political rivals, including by withholding as leverage a White House meeting and nearly $400 million in vital military aid from the country. “There’s nothing,” Ms. Pelosi said, “the Senate can do that can ever erase that.”
The comments appeared to be an acknowledgment that given the highly polarized state of the nation and the Senate, there was little chance that the two-thirds of senators needed for conviction and removal would agree to do so. They also foreshadowed a likely election-year strategy by Democrats, who are prepared to argue that voters should serve as an appeals court on Mr. Trump’s fitness for office.
“They take an oath to have a fair trial, and we think that would be with witnesses and documentation,” Ms. Pelosi said. “Now the ball is in their court to either do that, or pay a price for not doing it.” Before he called for a dismissal of the case, Mr. Trump appeared to be piqued by Ms. Pelosi. He suggested on Twitter questions for Ms. Pelosi to answer on air and later responded to her statements about his legacy. “Why should I have the stigma of Impeachment attached to my name when I did NOTHING wrong?” he asked in one tweet, after her appearance, adding, “Very unfair to tens of millions of voters!”
In other posts, Mr. Trump repeated familiar attacks on the speaker and Representative Adam B. Schiff, the California Democrat and House Intelligence Committee chairman who oversaw the impeachment inquiry and is likely to lead the prosecution in the Senate. Both Ms. Pelosi and Mr. Schiff, he said, ought to be witnesses during the impeachment trial, a common argument among Republicans who believe the two lawmakers have mishandled their duties in their conduct of the impeachment inquiry.
Congressional Republicans close to Mr. Trump also pounced on the speaker’s prediction, arguing that it betrayed the Democrats’ true motivation throughout the impeachment process: to build a case, any case, to slime Mr. Trump’s re-election campaign before the November election.
“Remember this, next time they try to pretend they were ‘prayerfully’ or ‘reluctantly’ impeaching @realDonaldTrump. They weren’t,” Representative Mark Meadows, Republican of North Carolina, wrote on Twitter. “It was all about trying to politically damage the President.”
Ms. Pelosi unexpectedly withheld the articles in a bid to increase pressure on Republican senators to commit to calling witnesses, without whom Democrats privately concede they have no shot of even coming close to a conviction. Though she relented without winning any commitments from Republicans, the speaker argued on Sunday that she had succeeded in showing the public “the need for witnesses.”
Republican senators have already challenged the speed and exhaustiveness of the House’s inquiry, and questions about the strength of the evidence it gathered and witnesses it did not secure are likely to be front and center as the trial gets underway.
Under the rules Mr. McConnell intends to adopt, senators will eventually be able to vote on whether to call witnesses or other evidence, but there is no guarantee they will do so. Mr. McConnell has made clear that he wants a narrow and speedy trial, and that he does not foresee any chance of a conviction.
It would take only four Republican senators to break ranks and join Democrats to secure the majority needed to compel the new testimony — though even that may not be a guarantee.
“I don’t expect anything, but I don’t think it’s impossible,” said Senator Michael Bennet of Colorado, who is seeking the Democratic presidential nomination to take on Mr. Trump this fall, when asked about possible Republican defections.
“It’s important for public opinion to at least understand that what we are trying to do is hold the president accountable,” said Senator Michael Bennet, Democrat of Colorado.Credit...Anna Moneymaker/The New York Times
The most animated witness fight may be over John R. Bolton, the former White House national security adviser who said last week that he would be willing to testify in the trial if called by the Senate and even if the White House ordered him not to. Other impeachment witnesses and people close to him say that Mr. Bolton was deeply alarmed by Mr. Trump’s actions toward Ukraine and, as one of his top aides, has relevant, firsthand information to share with lawmakers.
Ms. Pelosi said on Sunday that she had not “excluded” the possibility that the House could subpoena Mr. Bolton to testify, but indicated that it was the Senate’s responsibility for now. Mr. Schiff agreed, saying that there was “little sense” in bringing Mr. Bolton into the House now.
"""

txt2="""
WASHINGTON — President Trump asserted without evidence on Thursday that a top Iranian commander killed in an American airstrike was plotting to blow up an American embassy.
The president’s unsubstantiated account comes as Democrats are demanding details about the intelligence underlying the Trump administration’s decision to kill Maj. Gen. Qassim Suleimani, the leader of the powerful Quds Force of Iran’s Islamic Revolutionary Guards Corps.
“We did it because they were looking to blow up our embassy,” Mr. Trump said in remarks to reporters during an unrelated event at the White House.
It was unclear whether Mr. Trump might have been disclosing new details about what the administration has called an “imminent” Iranian plot against American interests in the region, or whether he was referring to the pro-Iranian protesters who stormed the American Embassy in Baghdad last week, a subject he returned to later in his remarks.
Democrats and some Republicans are frustrated with the refusal of Trump officials to reveal more about the intelligence that prompted the targeted killing of a foreign military official in Mr. Suleimani on Jan. 2.
Earlier on Thursday, Vice President Mike Pence defended classified Capitol Hill briefings that national security officials delivered on Wednesday. Several member of Congress described the briefings as inadequate and even insulting in their lack of specifics.
“Some of that has to do with what’s called sources and methods,” Mr. Pence said on NBC’s “Today” show. “Some of the most compelling evidence that Qassem Soleimani was preparing an imminent attack against American forces and American personnel also represents some of the most sensitive intelligence that we have — it could compromise those sources and methods.”
Mr. Trump’s comments came during a brief exchange with reporters after a White House event on newly proposed environmental regulations.
"""
txt3="""
Senator Joseph R. Biden Jr. proposed a compromise.
It was the fall of 2002 and the Bush administration was pushing for sweeping authority to act against Saddam Hussein, claiming he had weapons of mass destruction. Some Democrats questioned the stated threat posed by Iraq and bristled at President George W. Bush’s broad request.
Mr. Biden, the Senate Foreign Relations Committee chairman, had been scrambling to draft a bipartisan resolution that would grant Mr. Bush the authority to use military force against Iraq, but was more restrictive than the war authorization that the president had sought.
As he often had in his long career, Mr. Biden sought bipartisan middle ground — this time, between those opposed to potential war and the White House desire for more open-ended power. Some antiwar members of his committee resisted his effort, worried that it would still pave the way to conflict. “We disagreed very strenuously,” said former Senator Barbara Boxer, Democrat of California.
Mr. Biden’s plan ultimately did not succeed, and he chose to focus on Mr. Bush’s reassurances of a diplomacy-first approach.
“At each pivotal moment,” Mr. Biden said of Mr. Bush, “he has chosen a course of moderation and deliberation, and I believe he will continue to do so. At least that is my fervent hope.”
On Oct. 11, he was one of 77 senators to authorize the use of military force in Iraq. Twenty-three colleagues, some of whom harbored grave doubts about the danger Iraq posed at the time, refused to back the president’s request.
"""
txt4="""
Joseph R. Biden Jr. on Tuesday delivered a forceful and detailed critique of the Trump administration’s escalation toward Iran as he sought at every turn to highlight his own commander-in-chief credentials and to evoke the stature of a president.
The former vice president, speaking from Pier 59 in New York against a backdrop of American flags, delivered a stern warning about President Trump’s stewardship of international affairs and painted a grim picture of the dangerous landscape in the Middle East.
Mr. Biden cast Mr. Trump as a hypocrite who preaches an inward-looking foreign policy but operates using unsteady, bellicose tactics, and faulted the president for bringing the nation “dangerously close” to war after an American drone strike that killed one of Iran’s top military commanders, Maj. Gen. Qassim Suleimani.
“Because he refuses to level with the American people about the dangers which he has placed American troops and our diplomatic corps, personnel and civilians, as well as our partners and allies, or demonstrated even a modicum of presidential gravitas, I will attempt to do that,” said Mr. Biden, who is one of 14 candidates seeking the Democratic presidential nomination. “That starts with an honest accounting of how we got where we are.”
Hours after he spoke, Iran launched more than a dozen ballistic missiles against United States military and coalition forces in Iraq, according to American military officials.
“What’s happening in Iraq and Iran today was predictable,” Mr. Biden said at a fund-raiser in the Philadelphia area on Tuesday night, citing the “chaos that’s ensuing” from Mr. Trump’s approach to the region. He added, “I just pray to God as he goes through what’s happening, as we speak, that he’s listening to his military commanders for the first time because so far that has not been the case.”
One of Mr. Biden’s rivals, Senator Elizabeth Warren of Massachusetts, opened a rally on Tuesday night at a packed house in Kings Theater in Brooklyn by telling the crowd about the Iranian strikes.
“My three brothers all served in the military,” she said. “At this moment, my heart and my prayers are with our military and with their families in Iraq and all around the world.”
“But this is a reminder why we need to de-escalate tension in the Middle East,” Ms. Warren continued. “The American people do not want a war with Iran.”
The crowd thundered in approval.
Senator Elizabeth Warren said at an event on Tuesday at Kings Theater in Brooklyn, “The American people do not want a war with Iran.”Credit...Calla Kessler/The New York Times
All of the leading Democratic candidates have excoriated the Trump administration over its posture toward Iran. Senator Bernie Sanders of Vermont has used the moment to highlight his longstanding antiwar credentials — and to point out, obliquely and explicitly, Mr. Biden’s vote to authorize the war in Iraq. And former Mayor Pete Buttigieg of South Bend, Ind., a military veteran, has emphasized his personal experience in a war zone.
In Mr. Biden’s telling, the chaos America now confronts — “heightened threats, chants of ‘death to America’ once more echoing across the Middle East,” with Iran and its allies “vowing revenge” — was stoked by Mr. Trump’s withdrawal from the Iran nuclear deal that was negotiated under the Obama administration.
He described a series of subsequent provocations and challenges for which the Trump administration, he argued, was unprepared, even as he also said that he had “no illusions about Iran,” led by a government that he said threatened American interests. But the decision to kill General Suleimani “may well do more to strengthen Iran’s position in the region than any of Suleimani’s plots would have ever accomplished,” he said, arguing that Mr. Trump owed the nation answers but had delivered only “tweets, threats and tantrums.”
General Suleimani was designated by the United States as a terrorist, and was responsible for the deaths of hundreds of American troops. Defense Secretary Mark T. Esper said Tuesday that attacks orchestrated by General Suleimani had been expected within “days,” adding that he “has the blood of hundreds of Americans, soldiers, on his hands and wounded thousands more,” and many Republicans, including some who are typically critical of Mr. Trump, said the action had made the nation safer. Democrats, too, including Mr. Biden, have said General Suleimani should face justice. But other officials have questioned whether an attack was in fact imminent.
Mr. Biden said that if there was an “imminent threat” that warranted this “extraordinary action,” Americans should receive “an explanation and the facts to back it up.”
“At precisely the moment when we should be rallying our allies to stand beside us and hold the line against threats, Donald Trump’s shortsighted, ‘America First’ dogmatism has come home to roost,” Mr. Biden said. He went on to add, “We are alone now. We’re alone and we’ll have to bear the cost of Donald Trump’s folly.”
On the campaign trail, Mr. Biden has emphasized his decades-long record in international affairs and extensive relationships abroad, highlighting a contrast with his rivals in the Democratic field, who have largely been focused on domestic matters throughout the race. In New York, the setting appeared designed to conjure the White House briefing room: Mr. Biden spoke against a blue backdrop before a room full of reporters. He walked in to the clicking of multiple cameras and offered his own prescription for the path forward, calling for “cleareyed, hard-nosed diplomacy grounded in a strategy that’s not about one-off decisions and one-upsmanship.”
“Mr. President,” he urged, “you have to explain your decision and your strategy to the American people. That’s your job as president, Mr. President. Not ‘Dear Leader.’ Not ‘Supreme Leader.’ Democracy runs on accountability, and nowhere is it more important than the power to make war and bring peace.”
Mr. Biden, who has faced renewed scrutiny of his foreign policy record and especially his Iraq war vote, also appeared to misspeak at times, referring to Iran when he apparently intended to say Iraq, and he appeared to say that the current perilous situation was “unavoidable,” when excerpts from the speech circulated before the appearance used the word “avoidable.”
Still, Mr. Biden’s advisers and allies hope that the gravity of the moment will further crystallize the importance of defeating Mr. Trump in the minds of voters, and polls continue to show that Democrats believe Mr. Biden has the best chance to do so, though other candidates have also polled strongly against the president in hypothetical head-to-head matchups.
The current crisis “just reinforces how high the stakes are in this election,” Mr. Biden said at a fund-raiser earlier Tuesday, during which he also envisioned a Senate dynamic in which one could see “Mitch McConnell changing some ideas or being more — how can I say — mildly cooperative,” even as Mr. Biden also discussed the imperative for Democrats to win back the Senate, revoking Mr. McConnell’s title as majority leader.
Also on Tuesday, on ABC’s “The View,” Ms. Warren said that by killing General Suleimani, Mr. Trump had “moved us close to the edge of war.”
“Suleimani was a bad guy,” she said, “but the question is, what’s the right response?”
“The job of the president of the United States is to keep America safer,” she added. “And having killed Suleimani does not make America safer.”
As evidence, Ms. Warren cited the withdrawal of American citizens from the region because of safety concerns and said Mr. Trump had tweeted “threats of war crimes” by threatening to strike cultural institutions.
“We need a president who actually has good judgment, and is willing to follow through,” she said.
Pressed on whether she believed General Suleimani to be a terrorist, Ms. Warren said: “Of course he is. He’s part of a group that our federal government has designated as a terrorist.”
As for how she would handle foreign policy in the Middle East, she said, “It is the responsibility of the commander in chief not to ask our military to solve problems that cannot be solved militarily.”
“The point is not whether or not Iran is a bad actor — they are,” she added. “But the question is: What are the right steps for the president? Use our diplomacy. Use the back channels. De-escalate and get Iran to the negotiating table.”
"""

print(multi_text_co_occur([txt,txt2,txt3,txt4,],300),id2entity)