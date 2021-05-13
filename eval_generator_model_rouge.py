#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
from typing import List, Any

import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity


def eval_flow(url: str,
              dataset_num: int = 200,
              model_type: str = None):
    """
    計算抽取、生成式摘要的rouge流程
    1. get dataset(input_content、summarize)
    2. call deepNlp api to get generator text
    3. save compare results
    4. calculate the rouge of summarize and generator text
    """
    # get dataset(input_content、summarize)
    datasets = load_dataset(dataset_num)
    summarize_texts = []
    generator_texts = []

    # call deepNlp api to get generator text
    for dataset in tqdm(datasets):
        # content = "烏法峰會將啟動印、巴加入程序上合組織擴員大門正式打開中新網北京7月6日電(記者張朔)上海合作組織成員國元首理事會第十五次會議本週在俄羅斯烏法舉行，中國國家主席習近平將出席。烏法峰會將通過關於啟動接收印度、巴基斯坦加入上合組織程序的決議，上合組織擴員的大門正式打開。中國外交部6日在北京舉行中外媒體吹風會，外交部副部長程國平介紹此次與會情況，並回答記者有關上合組織擴員等方面的提問。程國平說，上合組織成立14年來，不斷深化成員國間戰略協作和團結互信，全面推進各領域合作，已成為維護本地區安全穩定、促進共同發展的重要平台。去年杜尚別峰會以來，各成員國積極落實峰會成果，推動各領域合作進一步深化，取得一系列新成果，上合組織國際影響力不斷上升。程國平表示，隨著上合組織發展影響擴大，越來越多的域內國家紛紛提出加入。順應這些國家的利益需要並根據上合組織自身發展的需要，本次烏法峰會將通過關於啟動接收印度、巴基斯坦加入上合組織程序的政治決議，這就意味著上合組織擴員的大門正式打開，印、巴加入上合組織的法律程序啟動。程國平指出，我們也看到國與國之間難免存在著這樣、那樣的分歧與矛盾，這些都是歷史形成的，但並不影響國家間發展友好關係。印、巴加入上合組織不僅將對上合組織發展發揮重要作用，也將為促進印巴之間的友好關係不斷發展和改善發揮建設性作用。(完)"
        generator_texts.append(call_deepNlp_api(url=url,
                                                text1=dataset["article"],
                                                # content=content,
                                                model_type=model_type))
        summarize_texts.append(dataset["summarization"])
        # print(dataset)
        # print(generator_texts)
        # print(summarize_texts)
        # break

    print("assert length with summarize_dataset and generator_text")
    assert len(summarize_texts) == len(generator_texts)

    # save compare results
    print("save compare results")
    file_name = "cr.csv" if not model_type else f"{model_type}_cr.csv"
    save_compare_results(article_text=[d["article"] for d in datasets],
                         generator_text=generator_texts,
                         summarize_texts=summarize_texts,
                         file_name=file_name)
    print(f"{file_name} is eval success!!!")

    # calculate the rouge of summarize and generator text
    print("calculate the rouge of summarize and generator text")
    scores = calculate_rouge(hypothesis=generator_texts, reference=summarize_texts)
    print(scores)


def load_dataset(dataset_num: int):
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"total data: {len(data)}")
    # for i in range(3):
    #     print(f"data[{i}]:\nsummary: {data[i]['summarization']}\narticle: {data[i]['article']}\n")

    data = data[:dataset_num]
    # train_set, val_set = train_test_split(data,
    #                                       test_size=0.05,
    #                                       random_state=42)
    # print(f"train data len: {len(train_set)}")
    # print(f"val data len: {len(val_set)}")
    return data


def call_deepNlp_api(url: str,
                     text1: str,
                     text2: str = None,
                     model_type: str = None) -> Any:
    # task = {"summary": "Take out trash", "description": ""}
    if model_type:
        input_json = {
            "token": "",
            "model_type": model_type,
            "doc_list": [
                {
                    "id": "1",
                    "content": text1
                }
            ]
        }
        print(f"text1: {text1}")
    else:
        input_json = {
            "token": "eyJhbGciOiJIUzUxMiJ9.eyJhcHBfbmFtZSI6ImV5SmhiR2NpT2lKSVV6VXhNaUo5LmV5SmhjSEJmIiwiZW1haWwiOiJqYXNvbmxpbkBlbGFuZC5jb20udHcifQ.43EjrvpGrSfi2L3MkJmGxEUX6PsKx7tarRSeYU06HDDtOHCcV9bFtN-O2-qGVnbl084BgwFNrU6VGKDlOsS1xw",
            "app_name": "release test",
            "dev_mode": "True",
            "settings": {
                "split_sentence": "True",
                "concat_short_sent": "True",
                "min_seq_length": "2",
                "max_seq_length": "96",
                "batch_size": "512",
                "l2_normalize": "False",
                "debug_mode": "False"
            },
            "doc_list": [
                {
                    "id": "z1",
                    "content": text1
                },
                {
                    "id": "a2",
                    "content": text2
                }
            ]
        }

    resp = requests.post(url=url,
                         json=input_json,
                         headers={'Content-Type': 'application/json'})
    if resp.status_code == 200:
        result = resp.json()["result_list"]
        if "gn_summarizer" in url:
            return result[0]
        elif "summarizer" in url:
            return "".join([e["sentence"] for e in result[0]["entity_list"]])
        elif "embedding" in url:
            return [r["embedding"] for r in result]
    else:
        print(f"Error request status_code : {resp.status_code}\n resp is : {resp}")


def calculate_rouge(hypothesis, reference):
    """eval methods 支援list[str]格式"""
    rouge = Rouge()

    # print(f"hypothesis: {hypothesis}")
    # print(f"reference: {reference}")
    hypothesis = [" ".join(h) for h in hypothesis]
    reference = [" ".join(r) for r in reference]

    # hypothesis = ["the #### transcript is a written version of each day's cnn student news program use this transcript to help students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you saw on cnn student news"]
    # reference = ["this page includes the show transcript use the transcript to help students with reading comprehension andvocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teacher or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests students ' knowledge of even ts in the news"]

    scores = rouge.get_scores(hypothesis, reference, avg=True)
    return scores


def save_compare_results(article_text: List[str],
                         generator_text: List[str],
                         summarize_texts: List[str],
                         file_name: str):
    cars = {
        "article": article_text,
        "generator": generator_text,
        "summarize": summarize_texts
    }

    df = pd.DataFrame(cars, columns=["article", "generator", "summarize"])

    df.to_csv(file_name, index=False, header=True)


def _strip(s):
    return s.strip()


def cal_singleton_article(file_name: str):
    file_path = rf"cr_result/{file_name}.csv"

    df = pd.read_csv(file_path, encoding="utf-8")
    print(df.head())
    article, hypothesis, reference = df.article, df.generator, df.summarize

    rouge_result = {}
    for a, h, r in tqdm(zip(article, hypothesis, reference)):
        rouge = calculate_rouge([h], [r])
        # rouge_result[h[:5]] = rouge
        rouge_result[a[:20]] = [rouge[r]["r"] for r in rouge]
        # print(rouge_result)
        # break
    rouge_result = sorted(rouge_result.items(), key=lambda item: sum(item[1]), reverse=True)[:10]
    print(rouge_result)


def cal_plural_article(file_names: List[str], article_list: List[str], url: str):
    for file_name in file_names:
        file_path = rf"cr_result/{file_name}.csv"
        df = pd.read_csv(file_path, encoding="utf-8")
        tg_df = df[df["article"].isin(article_list)]
        assert len(tg_df) == len(article_list)
        print(f"{file_name} assert success")

        article, hypothesis, reference = tg_df.article, tg_df.generator, tg_df.summarize
        rouge_result = {}
        cosine_result = {}
        for a, h, r in tqdm(zip(article, hypothesis, reference)):
            # calculator rouge
            rouge = calculate_rouge([h], [r])
            rouge_result[a[:20]] = np.mean([rouge[r]["r"] for r in rouge])

            # call embedding
            result = call_deepNlp_api(
                url=url,
                text1=h,
                text2=r,
            )
            cosine_result[a[:20]] = cosine_similarity([result[0]], [result[1]])[0][0]
        rouge_result = sorted(rouge_result.items(), key=lambda item: item[1], reverse=True)
        cosine_result = sorted(cosine_result.items(), key=lambda item: item[1], reverse=True)
        print(rouge_result)
        print(cosine_result)


def cal_article_rouge_cosine(hypothesis: List[str], reference: List[str], url: str):
    rouge_result = {}
    cosine_result = {}
    for h, r in tqdm(zip(hypothesis, reference)):
        rouge = calculate_rouge([h], [r])
        rouge_result[r[:20]] = np.mean([rouge[r]["r"] for r in rouge])

        # call embedding
        result = call_deepNlp_api(
            url=url,
            text1=h,
            text2=r,
        )
        cosine_result[r[:20]] = cosine_similarity([result[0]], [result[1]])[0][0]
    rouge_result = sorted(rouge_result.items(), key=lambda item: item[1], reverse=True)
    cosine_result = sorted(cosine_result.items(), key=lambda item: item[1], reverse=True)
    print(rouge_result)
    print(cosine_result)


if __name__ == '__main__':
    # dataset_path = "nlpcc2017/train_with_summ.json"
    dataset_path = "nlpc_200.json"
    summarizer_url = "http://rd2demo.eland.com.tw/summarizer"
    # gn_summarizer_url = "http://rd2demo.eland.com.tw/gn_summarizer"
    gn_summarizer_url = "http://172.18.20.190:5050/gn_summarizer"
    embedding_url = "http://10.20.50.10:5000/embedding"
    # model_type = "pegusas"
    # model_type = "default"
    # model_type = "t5"
    model_type = "bart"

    # Content = "2019 下半年將到來，多家手機大廠的旗艦新機也將一一與消費大眾見面，爆料大神 Evan Blass 日前則揭露美國電信商 Verizon 針對三星、蘋果、Google 的新機行銷規劃。 在官方正式發表前，手機大廠的年度新機總是傳聞不斷，其中 Google 相當乾脆地公開了今年將發表的 Pixel 4 宣傳照，讓好奇的消費者搶先得知外觀設計、進一步推測可能具備的功能。 而在同一天爆料大神 Evan Blass 於個人 Twitter 帳號上揭露美國電信商 Verizon 針對重點新機的行銷規劃截圖，從下圖可看出，除了已在 5 月推出 Pixel 3a 系列以外，包括三星 Galaxy Note 10、蘋果 iPhone、以及 Google Pixel 4 都在計畫當中，預計是 8 月下旬發表 Galaxy Note 10、9 月下旬發表 iPhone、10 月中旬發表 Pixel 4份行銷規劃剛好呼應了 Google 提早 4 個月就先公開一小部份 Pixel 4 的新機資訊；此外值得注意的是，圖表上只列出「iPhone」發表，回顧去年 iPhone XR 是在 iPhone XS、iPhone XS Max 上市一個月後才開賣，或許今年推出的 3 款全新 iPhone 可能在同一時間上市。 而從 3 款新機的發表時程來看，並不令人感到意外，參考過去的發表時程如 iPhone 在 9 月份、Pixel 在 10 月份並未偏離預期範圍，唯獨只是正式發表的日期究竟在哪一天尚未公佈。 然而對於這 3 款系列機種來說，推出新機大部分著重在內部規格、相機功能等升級，並未有突出的創新設計，於是外界笑稱去年最大的創新在於價格創新高。若以最受國人矚目的 iPhone 來說，據瞭解從供應鏈端確實沒聽說新款 iPhone 有何有趣的亮點，甚至可說是沒有討論度。現今的智慧型手機已能滿足用戶的日常需求，那麼如何讓消費者願意掏錢換機，則考驗手機大廠能否創新應用服務、或提升使用體驗。"
    # print(call_deepNlp_api(url=summarizer_url, content=Content))
    # eval_flow(
    #     # url=summarizer_url,
    #     url=gn_summarizer_url,
    #     dataset_num=4,
    #     model_type=model_type
    # )

    """"""
    # import pandas as pd
    # df = pd.read_csv("pegusas_test_cr.csv")
    # hypothesis = df["generator"].tolist()
    # reference = df["summarize"].tolist()
    # hypothesis = ["華西都市報記者迅速聯繫上了與章子怡家里關系極好的知情人士，華西都市報記者為了求證章子怡懷孕消息，有關章子怡懷孕的新聞自從2013年9月份章子怡和汪峰戀情以來，"]
    # {'rouge-1': {'f': 0.21666666216805563, 'p': 0.16455696202531644, 'r': 0.3170731707317073},
    #  'rouge-2': {'f': 0.06779660568802097, 'p': 0.05128205128205128, 'r': 0.1},
    #  'rouge-l': {'f': 0.21978021499818876, 'p': 0.18181818181818182, 'r': 0.2777777777777778}}

    # hypothesis = ["張子怡是中國最著名的女演員之一，但她的個人生活一直被中國媒體炒作。"]
    # {'rouge-1': {'f': 0.16216215722059915, 'p': 0.18181818181818182, 'r': 0.14634146341463414},
    #  'rouge-2': {'f': 0.02777777283950705, 'p': 0.03125, 'r': 0.025},
    #  'rouge-l': {'f': 0.12307691813491144, 'p': 0.13793103448275862, 'r': 0.1111111111111111}}

    # hypothesis = [ 良好關係的內幕人士。"]
    # {'rouge-1': {'f': 0.18947367930415523, 'p': 0.16666666666666666, 'r': 0.21951219512195122},
    #  'rouge-2': {'f': 0.021505371441786294, 'p': 0.018867924528301886, 'r': 0.025},
    #  'rouge-l': {'f': 0.14457830834083338, 'p': 0.1276595744680851, 'r': 0.16666666666666666}}

    # hypothesis = ["“這是怎麼回事？新聞是真是假？” 23日下午8點30分，對此消息的回應是，《華西都市報》的一名記者迅速聯繫到與張子怡一家有著良好關係的內部人士。消息人士說，王峰原本打算在音樂會前宣布重要新聞，而王峰已經去上海參加音樂會了。消息人士說：“子義是"]
    # {'rouge-1': {'f': 0.12422359868832233, 'p': 0.08333333333333333, 'r': 0.24390243902439024},
    #  'rouge-2': {'f': 0.01257861258652856, 'p': 0.008403361344537815, 'r': 0.025},
    #  'rouge-l': {'f': 0.08064515716961519, 'p': 0.056818181818181816, 'r': 0.1388888888888889}}

    # hypothesis = ["最近，有媒體報導張子怡真的懷孕了！該報告還引用了熟悉此事的人士的話說：“張子怡懷孕了大約四五個月，到期日大約是今年年底，現在她不再工作了。”怎麼了？這個消息是對還是錯？對此消息的回應，是23日下午8:30，《華西都市報》的一名記者迅速聯繫到與張自三有良好關係的內幕人士。"]
    # {'rouge-1': {'f': 0.17045454188081102, 'p': 0.1111111111111111, 'r': 0.36585365853658536},
    #  'rouge-2': {'f': 0.011494249332805951, 'p': 0.007462686567164179, 'r': 0.025},
    #  'rouge-l': {'f': 0.13846153445680487, 'p': 0.09574468085106383, 'r': 0.25}}

    # hypothesis = ["最近，有媒體報導張子怡真的懷孕了！報告還引用了熟悉此事的人士的話說：“張子子懷孕了大約四個月或五個月，到期日大約是今年年底，現在她不再工作了。與章子怡一家的關係向華西都市日報記者證實，張這次懷孕了。"]
    # {'rouge-1': {'f': 0.21428571014387762, 'p': 0.15151515151515152, 'r': 0.36585365853658536},
    #  'rouge-2': {'f': 0.02898550312959521, 'p': 0.02040816326530612, 'r': 0.05},
    #  'rouge-l': {'f': 0.16981131626913504, 'p': 0.12857142857142856, 'r': 0.25}}

    # reference = ["知情人透露章子怡怀孕后，父母很高兴。章母已开始悉心照料。据悉，预产期大概是12月底"]
    # print(calculate_rouge(hypothesis, reference))

    # from nlgeval import NLGEval
    #
    # nlgeval = NLGEval()  # loads the models
    # metrics_dict = nlgeval.compute_individual_metrics(references, hypothesis)

    # from pycocoevalcap.cider.cider import Cider

    # hyp = "台積電認為他違反競業禁止協議，長江存儲是中國商武漢新芯積體電路製造公司的百分之百持股母公司，等於為台積電競爭對手武漢新芯提供服務，"
    # ref = "台積電（TSMC）的前任經理因違反競爭禁令而被判處兩年徒刑，但法院維持了該公司要求他辭職並向他支付賠償的決定，但法院維持了該公司的決定。要求他辭職並向他發出賠償，但法院維持了該公司要求他辭職並向他發出賠償的決定，但法院維持了該公司要求他辭職並向他發放賠償的決定，但法院維持了該公司的判決。決定要求他辭職並向他發出賠償，但法院維持了該公司"
    # hyp = {index: [h] for index, h in enumerate(hyp)}
    # ref = {index: [ref] for index in hyp.keys()}

    # hyp_list = ['this is the model generated sentence1 which seems good enough']
    # ref_list = [['this is one reference sentence for sentence1']]
    #
    # ref_list = [list(map(_strip, refs)) for refs in zip(*ref_list)]
    # refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
    # hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(hyp_list)}
    # # print(hyp)
    #
    # cider = Cider()
    # # print(cider.compute_score(gts=hyp, res=ref))
    # print(refs)
    # print(hyps)
    #
    # print(cider.compute_score(gts=refs, res=hyps))

    """calculator singleton article"""
    # cal_singleton_article("cr")  # 抽取式
    # [('烏法峰會將啟動印、巴加入程序<Parag', [1.0, 0.9090909090909091, 0.9743589743589743]),
    #  ('發布日期：2015-03-21<Para', [0.9655172413793104, 0.9298245614035088, 0.9743589743589743]),
    #  ('發布日期：2015-07-12<Para', [0.9482758620689655, 0.8947368421052632, 0.9565217391304348]),
    #  ('發布日期：2015-05-14<Para', [0.9491525423728814, 0.896551724137931, 0.9512195121951219]),
    #  ('發布日期：2014-11-16< Par', [0.9545454545454546, 0.8837209302325582, 0.9459459459459459]),
    #  ('2015 -07-08<Paragrap', [0.8275862068965517, 0.7719298245614035, 0.95]),
    #  ('肖雙勝近期，河北省紀委對廊坊市委常委、政', [0.88, 0.7916666666666666, 0.8695652173913043]),
    #  ('發布日期：2015-07-02 <Par', [0.8333333333333334, 0.7560975609756098, 0.8461538461538461]),
    #  ('發布日期：2015-01- 17< Pa', [0.7857142857142857, 0.6829268292682927, 0.84375]),
    #  ('三峽集團和宜昌市聯合２４日宣布：自２５日', [0.8085106382978723, 0.6304347826086957, 0.7560975609756098])]

    # cal_singleton_article("default_cr")  # distilbart
    # [('光明網05-11 <Paragraph>', [0.5625, 0.2765957446808511, 0.5714285714285714]),
    #  ('鳳凰科技訊< Paragraph>北京時', [0.5, 0.29411764705882354, 0.5]),
    #  ('發布日期：2014-11-16< Par', [0.5, 0.23255813953488372, 0.4864864864864865]),
    #  ('摘要Ⅰ<Paragraph>北京時間6月', [0.5217391304347826, 0.2222222222222222, 0.4666666666666667]),
    #  ('2015 -07-08<Paragrap', [0.46551724137931033, 0.2631578947368421, 0.475]),
    #  ('人民網昆明6月11日電<Paragrap', [0.4915254237288136, 0.25862068965517243, 0.36538461538461536]),
    #  ('中國經濟網北京6月24日綜合報導（尹彥宏', [0.43333333333333335, 0.2413793103448276, 0.4230769230769231]),
    #  ('海關總署發布最新數據顯示， 1月份我國進', [0.4791666666666667, 0.1702127659574468, 0.4418604651162791]),
    #  ('6月28日消息，據國外媒體報導， 6月2', [0.4888888888888889, 0.22727272727272727, 0.3684210526315789]),
    #  ('斑馬線只夠一人行走<Paragraph ', [0.4666666666666667, 0.2727272727272727, 0.3333333333333333])]

    # cal_singleton_article("pegusas_sum_cr")  # pegusas-sum
    # [('本報訊<Paragraph>大理將對在大', [0.7222222222222222, 0.4857142857142857, 0.6060606060606061]),
    #  ('三峽集團和宜昌市聯合２４日宣布：自２５日', [0.5531914893617021, 0.32608695652173914, 0.4878048780487805]),
    #  ('澳大利亞央行將利率降至紀錄低點，以應對疲', [0.5526315789473685, 0.2972972972972973, 0.4857142857142857]),
    #  ('新華社昆明4月24日電<Paragrap', [0.6060606060606061, 0.3125, 0.4]),
    #  ('據中國政府網網站消息，國務院發布《國務院', [0.5263157894736842, 0.35135135135135137, 0.3793103448275862]),
    #  ('圖為深圳都市報頭版深圳都市報訊<Para', [0.48717948717948717, 0.2631578947368421, 0.46875]),
    #  ('肖雙勝近期，河北省紀委對廊坊市委常委、政', [0.48, 0.25, 0.4782608695652174]),
    #  ('@現代快報#徐州突發# <Paragra', [0.45652173913043476, 0.28888888888888886, 0.45]),
    #  ('據彭博社報導，一家酒店管理公司Tianf', [0.5238095238095238, 0.21951219512195122, 0.4358974358974359]),
    #  ('提起此事，孫桂枝（化名）和患有白內障的9', [0.5714285714285714, 0.2, 0.4])]

    # cal_singleton_article("t5_cr")  # t5
    # [('烏法峰會將啟動印、巴加入程序<Parag', [0.9111111111111111, 0.5227272727272727, 0.7692307692307693]),
    #  ('本報訊<Paragraph>大理將對在大', [0.8611111111111112, 0.6571428571428571, 0.5757575757575758]),
    #  ('據彭博社報導，一家酒店管理公司Tianf', [0.7619047619047619, 0.4634146341463415, 0.6666666666666666]),
    #  ('李登輝（資料圖）【環球軍事報導】92歲的', [0.7058823529411765, 0.4, 0.6]),
    #  ('2015-06-23<Paragraph', [0.6511627906976745, 0.3333333333333333, 0.6896551724137931]),
    #  ('據上海市黃浦區人民政府新聞辦官方微博@上', [0.6666666666666666, 0.3829787234042553, 0.5945945945945946]),
    #  ('歐盟主席Donald<Paragraph', [0.64, 0.3877551020408163, 0.6097560975609756]),
    #  ('斑馬線只夠一人行走<Paragraph ', [0.6444444444444445, 0.45454545454545453, 0.4523809523809524]),
    #  ('圖為罷工現場【環球網報導<Paragra', [0.6590909090909091, 0.3023255813953488, 0.5588235294117647]),
    #  ('周仲明<Paragraph>趙風樓<Pa', [0.6578947368421053, 0.43243243243243246, 0.4])]

    """calculator plural article"""
    # test = [
    #     "烏法峰會將啟動印、巴加入程序<Paragraph >上合組織擴員大門正式打開中新網北京7月6日電<Paragraph>(記者<Paragraph>張朔)上海合作組織成員國元首理事會第十五次會議本週在俄羅斯烏法舉行，中國國家主席習近平將出席。烏法峰會將通過關於啟動接收印度、巴基斯坦加入上合組織程序的決議，上合組織擴員的大門正式打開。中國外交部6日在北京舉行中外媒體吹風會，外交部副部長程國平介紹此次與會情況，並回答記者有關上合組織擴員等方面的提問。程國平說，上合組織成立14年來，不斷深化成員國間戰略協作和團結互信，全面推進各領域合作，已成為維護本地區安全穩定、促進共同發展的重要平台。去年杜尚別峰會以來，各成員國積極落實峰會成果，推動各領域合作進一步深化，取得一系列新成果，上合組織國際影響力不斷上升。程國平表示，隨著上合組織發展影響擴大，越來越多的域內國家紛紛提出加入。順應這些國家的利益需要並根據上合組織自身發展的需要，本次烏法峰會將通過關於啟動接收印度、巴基斯坦加入上合組織程序的政治決議，這就意味著上合組織擴員的大門正式打開，印、巴加入上合組織的法律程序啟動。程國平指出，我們也看到國與國之間難免存在著這樣、那樣的分歧與矛盾，這些都是歷史形成的，但並不影響國家間發展友好關係。印、巴加入上合組織不僅將對上合組織發展發揮重要作用，也將為促進印巴之間的友好關係不斷發展和改善發揮建設性作用。(完)",
    #     "發布日期：2015-03-21<Paragraph>04:05:33岳陽市氣象台3月21日4時2分發布大霧橙色預警信號：汨羅市、臨湘市、岳陽縣、岳陽市區、湘陰縣、平江縣、華容縣已經出現能見度不足200米的霧並將持續，能見度低，請注意防範。圖例標準防禦指南6小時內可能出現能見度小於200米的濃霧，或者已經出現能見度小於200米、大於等於50米的濃霧且可能持續。1、有關部門和單位按照職責做好防霧工作；2、機場、高速公路、輪渡碼頭等單位加強調度指揮；3、駕駛人員必須嚴格控制車、船的行進速度；4、減少戶外活動。",
    #     "發布日期：2015-07-12<Paragraph>15:20:00滁州市氣像台2015年07月12日15時20分發布雷電黃色預警信號：目前我市西部有較強對流雲團向東南方向移動，預計6小時內我市部分地區將發生雷電活動，並可能伴有短時強降水、大風、局部冰雹等強對流天氣，請注意防範。圖例標準防禦指南6小時內可能發生雷電活動，可能會造成雷電災害事故。1、政府及相關部門按照職責做好防雷工作；2、密切關注天氣，盡量避免戶外活動。",
    #     "發布日期：2015-05-14<Paragraph>09:15:00咸寧市氣象台5月14日09時15分發布雷電黃色預警信號:預計未來6小時內，咸寧市區、嘉魚縣、通城縣、通山縣、崇陽縣、赤壁縣有雷電活動,可能造成雷電災害事故，請有關部門注意防範。圖例標準防禦指南6小時內可能發生雷電活動，可能會造成雷電災害事故。1、政府及相關部門按照職責做好防雷工作；2、密切關注天氣，盡量避免戶外活動。",
    #     "歐盟主席Donald<Paragraph> Tusk今日表示，歐元區領導人將於下週一在布魯塞爾召開緊急會議，討論希臘局勢。Tusk表示：“現在是在最高政治層面緊急討論希臘局勢的時候了。 ”此外，有外媒報導稱，歐洲央行管委會將於週五就希臘緊急流動性援助(ELA)舉行未經事先安排的電話會議。此前，歐盟委員會副主席Valdis<Paragraph>Dombrocskis表示，歐元區財長們沒能在會議上就希臘問題達成一致。在經過了幾個小時的討論後，希臘和其債權人依然沒能救援助計劃達成協議。債權人表示，希臘必須遵守援助協議。而希臘總理齊普拉斯對此表示不贊同。德國時代周報今日在其網絡版上報導稱，希臘的債權人計劃延長其現有的援助計劃，直到今年年底，但IMF不會參與其中。該報表示，希臘現有援助計劃中用於銀行資本重組的100億歐元將被留下，用以償還其向歐洲央行以及IMF的欠款。對此，歐盟一位外交官向外媒表示，德國時代周報關於希臘援助獲得延期的報導“和事實毫無關係”。他澄清：“這與實際情況毫無關係，這樣的提案絕對不可能通過。 ”隨著當前救助協議臨近到期，希臘與債權人的談判進入倒計時。今日是達成協議的最佳時機。華爾街見聞文章此前提到，本週四的歐元區財長會議是希臘談判爭取突破的下一個機會。美銀美林的分析師預計，本週五以前如果不能敲定協議，本月底以前達成協議就相當困難。英國《金融時報》報導則是提到，很多歐洲官員認為，本週四如果不能達成協議，持續五個月的希臘債務僵局就會進入新的關鍵階段，屆時可能沒有足夠時間讓歐元區債權國的議會——特別是德國國會批准新協議。",
    #     "本報訊<Paragraph>大理將對在大理古城內從事生產經營及旅遊活動的單位和人員收取古城維護費，專項用於古城保護。近日，大理市人民政府發布第4號公告，出台《大理市大理古城保護管理辦法》（以下簡稱《辦法》），重申熱議多年的古城維護費，但具體徵收範圍、方式、收費程序等由古城保護管理機構制定徵收辦法，報大理市人民政府批准後才實施。據悉，《辦法》將於今年7月1日起實施。昨日，記者從大理市古城保護管理局了解到，《辦法》重點明確了古城保護範圍，把古城保護區分成了重點保護區、建設控制區和環境協調區。辦法重申，對古城內從事生產經營及旅遊活動的單位和人員收取古城維護費，專項用於古城保護。此外，對於在大理古城核心保護範圍內進行商業性影視攝製的，應按照利用補償原則，由利用單位或個人與古城保護管理機構簽訂有償利用協議。據了解，《辦法》還明確規定了古城重點保護區內建築的建設、修繕、審批等程序，在古城重點保護區內的經營項目實行準營證管理制度，在古城重點保護區內從事經營活動的單位和個人應當持有古城保護管理機構核發的《準營證》。對於消防安全，《辦法》專門指出，大理古城重點保護區禁止燃放煙花爆竹、燃燒火把、放孔明燈或使用易燃易爆危險物品，古城重點保護區內的單位、居民以及經營業主應按消防要求配備相應的消防器材和設施，生產生活中用電用火應當杜絕消防安全隱患。2009年就提出收古維費遊客每人次收30元據悉，關於大理古城維護費，2009年雲南省發改委、財政廳下發了對大理古城和洱海資源保護費收費標準的通知，稱雲南擬對進入大理古城的遊客徵收每人次30元的古城維護費，以籌集古城保護資金。大理將收取古城維護費的消息很快引發爭議，2009年大理州旅遊局局長馬金鐘在接受本報記者採訪時曾明確表示，大理兩年內不再提古城維護費。今年的全省兩會上，古城維護費的事在分組討論時再次被提及。大理州委書記梁志敏提出，今年將努力把大理古城打造升級成為5A級景區，還將考慮古城收費問題，收取的費用用於維護大理古城，大理州州長何華則表示，是否收費還在調研論證。“像麗江古城收取了維護費，所以古城內一些建設比較完善，大理古城的投入有限，差距很大。保護民居的基礎設施方面投入非常巨大。我個人認為每一個到大理的人對促進當地的發展，維護當地的文化、生態、建設都負有責任。 ”何華在接受采訪時表示，大理古城是世界的古城，不只是當地人要去維護，也希望每個到大理的遊客能和當地人一起來維護。對於大理古城維護費的收取，他表示最重要的是聽取多方聲音，多次調研論證後才會做出具體佈署。",
    #     "據彭博社報導，一家酒店管理公司Tianfang<Paragraph>Hospitality<Paragraph>Management計劃將一個資產組合為中國三家酒店的信託，在本地進行首次公開售股（IPO），募集約5億元。知情人士透露，這只信託可能最早9月在新加坡上市。這只房地產投資信託的資產由三家酒店構成，分別是位於天津的麗思卡爾頓和天沐溫泉度假酒店，還有位於海南的三亞海棠灣天房洲際度假酒店。根據彭博社的數據，今年新加坡交易所IPO所籌集的資金僅4100萬美元（5526萬新元），去年同期是7億2900萬美元。我國上市房地產投資信託指數今年累計下滑0.1%，<Paragraph>而海峽時報指數則下跌0.4%。",
    #     "李登輝（資料圖）【環球軍事報導】92歲的李登輝21日起訪問日本6天，並將前往日本國會議員會館發表演講。據台灣“中央社”報導，李登輝當地時間21日下午6時左右抵達東京。此次訪日共有6天行程，他將於22日在日本國會議員會館向國會議員發表演說。李登輝稱，他演講的內容包括台灣典範的變化，以及民主化和自由化等，“讓日本人了解這些關於台灣的議題”。23日與日本外國特派員協會進行午餐會，之後將參訪福島和仙台，26日前往岩沼寺千年希望之丘，向“3·11地震”罹難者獻花。聯合新聞網稱，李登輝曾於2007年造訪他最喜愛的“奧之細道”，並以日文寫下“松島光與影，炫目之光”的俳句，妻子曾文惠也寫下俳句，被製作成“句碑”。李登輝此行將在句碑旁植樹。“中央社”稱，夫人曾文惠因為生病不能同行，孫女李坤儀陪同李登輝到訪日本。此次日本行，是李登輝卸任以來第七次訪問日本，也是首次在日本國會議員會館演講。眾所周知，李登輝有著強烈的“日本情結”，他曾表示自己20歲以前是日本人，還有個日本名字叫岩里政男。1943年李登輝在台北讀完高中後到日本京都帝國大學留學，學習農業經濟。留學期間，他加入過日軍，成為名古屋高射砲部隊陸軍少尉時迎來了日本戰敗。2001年是李登輝卸任台灣領導人後首次訪日，名義是治療心髒病;2007年訪問時甚至參拜了靖國神社，他還多次宣揚“釣魚台(指釣魚島)是日本的”論調，遭到島內輿論的抨擊。(崔明軒)",
    #     "2015-06-23<Paragraph>16:06 <Paragraph>新浪財經<Paragraph>顯示圖片新浪財經訊<Paragraph>6月23日消息，希臘債務危機現轉機，歐美股市上漲，受此利好今日港股恆指高開0.17%，曾受A股下挫拉低，但觸底之後一路上揚，收漲0.93%，升252.610點，收報27333.459點；國企指數跑贏大市，收漲1.69%，升225.79點，收報13609.471點；紅籌指數收漲，報點。全日大市成交1359.916億港元。藍籌股普遍上漲，華潤電力領漲藍籌股，漲3.828%，報21.7港元，成交1.319億港元。中煤能源漲2.857%，聯想漲1.596%，中國移動漲3.153%，長和漲1.546%，港交所漲1.058%，騰訊控股漲0.701%。航空股大漲，東方航空H股漲6.65%，報7.06港元，成交1987萬股，成交額1.38億港元，H股較A股折價55.6%。南方航空漲5.78%，中國國航漲4.73%。內銀表現不俗，招商銀行漲4.752%，重慶銀行漲6.43%，盛京銀行漲8.26%，民生銀行漲3.38%，中國銀行漲3.05%，工商銀行漲2.79%，中國光大銀行漲1.96%，建設銀行漲1.96%，交通銀行漲1.89%，農業銀行漲1.94%內房股普漲，恆大地產漲5.83%，遠洋地產漲5.43%，融創中國漲5.00%，雅居樂地產漲3.99%，深圳控股漲3.55%，龍湖地產漲3.48%，華潤置地漲2.81，合景泰富漲3.19%。名家點評耀才證券市務總監郭思治表示，港股之表現則相對為穩，恆指早市先高開45點至27081點，其後在買盤追捧下再升至27243點方漸受規限，從技術上看，其時恆指不但已初步重越10天線（26858點），且更進一步輕越20天線（27232點），即是說，大市之回穩勢頭已進一步明顯，惟最關鍵的仍是50天線（27538點），一日恆指仍未能成功重越50天線之前，整個市勢仍未能擺脫反覆不明之趨勢。指數，都創出了新高。",
    #     "據上海市黃浦區人民政府新聞辦官方微博@上海黃浦消息，依據《中華人民共和國突發事件應對法》、《上海市實施〈中華人民共和國突發事件應對法〉辦法》，黃浦區政府對遇難人員家屬和受傷人員負有依法履行救助、撫慰的法定義務。<Paragraph>本著“依法依規、合情合理、實事求是、一視同仁”的原則，黃浦區政府會同有關社會組織共同研究制定了外灘擁擠踩踏事件遇難人員家屬救助方案。對此次事件遇難人員家屬的救助撫慰金確定為人民幣80萬元。其中，50萬為政府救助撫慰金，30萬為社會幫扶金。<Paragraph>傷殘人員的救助撫慰金額，將根據傷員救治、傷情和傷殘鑑定等具體情況另行確定。",
    # ]
    # cal_plural_article(
    #     file_names=["cr", "default_cr", "pegasus_sum_cr", "t5_cr"],
    #     article_list=test,
    #     url=embedding_url
    # )

    """
    calculator list[str] rouge 
    -> 
    GPT-2
    MT5
    """
    human_answer = [
        "上合組織元首峰會將在俄羅斯烏法舉行，將通過關於啟動接收印度、巴基斯坦加入上合組織程序的決議",
        "岳陽市發布大霧橙色預警：汨羅市、臨湘市、岳陽縣、岳陽市區、湘陰縣、平江縣、華容縣已經出現能見度不足200米的霧並...",
        "滁州市發布雷電黃色預警：目前我市西部有較強對流雲團向東南方向移動，預計6小時內我市部分地區將發生雷電活動，並可能...",
        "咸寧市發布雷電黃色預警：預計未來6小時內，咸寧市區、嘉魚縣、通城縣、通山縣、崇陽縣、赤壁縣有雷電活動,可能造成雷...",
        "歐盟主席稱，歐元區領導人將召開緊急會議討論希臘局勢。歐元區財長們此前沒能在會議上就希臘問題達成一致。",
        "大理古城7月1日起開收古城維護費，對生產經營及旅遊活動的單位和人員收取。",
        "外媒曝中國3家酒店擬組信託，公開售股集資5億元；據悉，這只信託最早9月在新加坡上市。",
        "92歲李登輝昨起訪問日本6天，將向國會議員發表演講；這是他卸任後第七次訪日，2007年曾參拜靖國神社。",
        "港股恆指收漲0.93%收報27333點，國企指數跑贏大市收漲1.69%，招商銀行領漲國企股",
        "上海外灘踩踏事件遇難者家屬將獲80萬撫慰金，其中50萬元為政府救助撫慰金，30萬元為社會幫扶金。"
    ]
    translation_answer = [
        "習近平將出席烏法峰會通過關於啟動接收印度、巴基斯坦加入的決議",
        "岳陽市發布大霧橙色預警：汨羅、臨湘市、岳陽縣、岳陽市區、湘陰縣",
        "滁州市發布雷電黃色預警：目前我市西部有較強對流雲團向東南方向移動",
        "咸寧市發布雷電黃色預警：預計未來6小時內，咸寧市區、嘉魚縣",
        "歐盟主席表示，“現在是最高政治層面緊急討論希臘局勢”。",
        "大理擬對古城維護費征收辦法：明年7月1日起施行。",
        "外媒稱，一個酒店管理公司將信托為中國三家酒店的資產組合。",
        "李登輝今日起訪問日本6天，將前往日本國會議員會館演講。",
        "港股恒指收漲0.93%報27333.459點，升253.610點；紅籌領漲藍籌",
        "上海黃浦區政府：遇難人員家屬將向社會組織提供80萬元救助金，其中50"
    ]
    simplified_answer = [
        "烏法峰會將於本周五在俄羅斯開幕，習近平出席；將通過關注印度、巴基斯坦加入",
        "岳陽市發布大霧橙色預警：汨羅、臨湘市、岳陽縣、岳陽市區、湘陰縣",
        "滁州市發布雷電黃色預警：目前我市西部有較強對流雲團向東南方向移動",
        "咸寧市發布雷電黃色預警：預計未來6小時內，咸寧市區、嘉魚縣、通城縣",
        "歐洲央行管委會將於下周一在布魯塞爾召開緊急會議討論希臘局勢",
        "大理擬對在古城維護費征收辦法，將用於古城保護；具體征收範圍",
        "外媒曝中國3家酒店擬在本地首次公開售股，共計約5億元，或最先9月在新加坡上市。",
        "92歲李登輝今日起訪日發表演講",
        "港股恒指收漲1.19%，收漲0.93%報27333.459點；國企指數跑贏大市",
        "黃浦區政府對遇難人員家屬的救助撫慰金確定為80萬元，其中50萬為"
    ]
    mt5_answer = [
        "上合組織將啟動印、巴加入程序上合組織大門正式打開,印、巴加入上合組織程序正式打開,印、巴加入上合組織的法律程序啟動。",
        "岳陽市氣像台3月21日4時2分發布大霧橙色預警信號:汨羅市、臨湘市、岳陽市、岳陽市...",
        "滁州市雷電黃色預警信:目前我市西部有較強對流雲團向東南方向,預計6小時內我市部分地區將發生雷電活動,或伴有短時強降水、大風、局部冰雹等...",
        "咸寧市氣像台5月14日09時15分發布雷電黃色預警信號:預測未來6小時內,咸寧市區、嘉魚、通城、通山、崇陽、赤壁、赤壁、赤壁、赤壁、赤壁、赤壁...",
        "德媒稱希臘援助未果,希臘未能救援助推,但IMF仍未參加;希臘或在最高政治層面緊急討論希臘局勢。",
        "大理將對在大理古城內從事旅遊及旅遊的單位和人員收取古城維護費,專項用於古城保護。",
        "新加坡上市房地產投資基金或最早9月在新加坡上市,或最早9月在新加坡上市,或最早9月在新加坡上市。",
        "92歲李登耀21日起訪問日本6天,並赴日本大使演演,曾表示自己20歲前是日本人,曾有日本名字叫岩里政男。",
        "希臘股市上漲,受此利好今日港股恆指高開0.17%,受此利好今日港股恆指高開0.17%,受此利好今日港股恆指高開0.17%,受此利好今日港股恆指高開0.17%,受此利好今日港股恆指高開0.17%,受此利好今日港股恆指高開0.17%,受此利好今日港股恆指高開0.17%,受此利好今日港股恆指高開0.17%,受此利好今日港股恆指高",
        "黃浦:外灘擁擠踩踏事件遇難者家屬救助金,50萬為政府救助撫慰金,50萬為政府救助撫慰金,30萬為政府救助撫慰金。"
    ]
    # print("translation_answer")
    # cal_article_rouge_cosine(translation_answer, human_answer, url=embedding_url)

    # print("simplified_answer")
    # cal_article_rouge_cosine(simplified_answer, human_answer, url=embedding_url)

    print("mt5_answer")
    cal_article_rouge_cosine(mt5_answer, human_answer, url=embedding_url)

