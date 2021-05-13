import json

import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import datetime
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from bs4 import BeautifulSoup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import Adafactor, get_linear_schedule_with_warmup, MT5ForConditionalGeneration, T5Tokenizer
from scipy.stats import spearmanr, pearsonr
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

# train_inputs, train_targets, dev_inputs, dev_targets, test_inputs, test_targets = [], [], [], [], [], []


def load_translation_dataset(data_name: str = "sts-train.tsv"):
    data_path = r"KorNLUDatasets/KorSTS"
    train = os.path.join(data_path, data_name)
    data = pd.read_csv(train, delimiter='\t', error_bad_lines=False)

    data.score = round(data.score * 5) / 5
    data = data.applymap(str)
    data['input'] = ''
    for i in range(len(data)):
        # strs_to_join = []
        strs_to_join = ['stsb sentence1:', data.iloc[i]['sentence1'], 'sentence2:', data.iloc[i]['sentence2']]
        data['input'].iloc[i] = " ".join(strs_to_join)
    target = data.score

    return data, target


def load_lcst_dataset(data_name: str = "PART_III.html"):
    file_name = rf"LCSTS2.0/DATA/{data_name}"
    with open(file_name, "r", encoding="utf-8") as f:
        yield BeautifulSoup(f.read(), 'html.parser')
    # print(f"total data: {len(data)}")
    # return data


def load_nlpcc_dataset(data_name: str = "train_with_summ.json"):
    file_name = f"nlpcc2017/{data_name}"
    with open(file_name, encoding="utf-8") as f:
        yield json.load(f)


def convert_translation_dataset(data, targets):
    target_max_length = 2
    _inputs, _target = [], []
    out_of_max_length = 0
    for input, target in zip(data.input, targets):
        if len(input) <= data_max_length:
            tokenized_inputs = tokenizer.encode_plus(
                input,
                max_length=data_max_length,
                padding='max_length',
                return_tensors="pt").input_ids
            _inputs.append(tokenized_inputs)

            tokenized_targets = tokenizer.encode_plus(
                target,
                max_length=target_max_length,
                padding='max_length',
                return_tensors="pt").input_ids
            _target.append(tokenized_targets)
        else:
            out_of_max_length += 1

    print(f"out of max_length: {out_of_max_length}")
    inputs_ids = torch.cat(_inputs, dim=0)
    labels = torch.cat(_target, dim=0)
    return inputs_ids, labels


def convert_nlpcc_dataset(dataset, num: int = 1000):
    _article, _summary = [], []
    out_of_max_length = 0
    for data in next(dataset)[:num]:
        article = data["article"][:article_max_length]
        summary = data["summarization"][:summary_max_length]
        _article.append(
            tokenizer.encode_plus(
                article,
                max_length=article_max_length,
                padding="max_length",
                return_tensors="pt"
            ).input_ids
        )

        _summary.append(
            tokenizer.encode_plus(
                summary,
                max_length=summary_max_length,
                padding="max_length",
                return_tensors="pt"
            ).input_ids
        )
    print(f"out of max_length: {out_of_max_length}")
    inputs_ids = torch.cat(_article, dim=0)
    labels = torch.cat(_summary, dim=0)
    return inputs_ids, labels


def convert_lcst_dataset(dataset, num: int = 1000):
    dataset = next(dataset)
    article = dataset.select("doc > short_text")
    summary = dataset.select("doc > summary")

    print(f"article length: {len(article)}")
    print(f"summary length: {len(summary)}")
    print(f"first article: {article[0].text}")
    print(f"first summary: {summary[0].text}")
    print(f"last article: {article[-1].text}")
    print(f"last summary: {summary[-1].text}")

    assert len(article) == len(summary)
    article = [a.text.strip()[:article_max_length] for a in article[:num]]
    summary = [s.text.strip()[:summary_max_length] for s in summary[:num]]

    _article, _summary = [], []
    for a, s in zip(article, summary):
        _article.append(
            tokenizer.encode_plus(
                a,
                max_length=article_max_length,
                padding="max_length",
                return_tensors="pt"
            ).input_ids
        )

        _summary.append(
            tokenizer.encode_plus(
                s,
                max_length=summary_max_length,
                padding="max_length",
                return_tensors="pt"
            ).input_ids
        )
    inputs_ids = torch.cat(_article, dim=0)
    labels = torch.cat(_summary, dim=0)
    return inputs_ids, labels


def convert_custom_dataset(dataset):
    _article, _summary = [], []
    for data in dataset:
        article = data[0][:article_max_length]
        summary = data[1][:summary_max_length]
        _article.append(
            tokenizer.encode_plus(
                article,
                max_length=article_max_length,
                padding="max_length",
                return_tensors="pt"
            ).input_ids
        )

        _summary.append(
            tokenizer.encode_plus(
                summary,
                max_length=summary_max_length,
                padding="max_length",
                return_tensors="pt"
            ).input_ids
        )
    inputs_ids = torch.cat(_article, dim=0)
    labels = torch.cat(_summary, dim=0)
    return inputs_ids, labels


def convert_to_dataLoader(input_ids, labels):
    X_train, X_valid, y_train, y_valid = train_test_split(input_ids, labels, test_size=0.01, random_state=42)
    train_dataset = TensorDataset(X_train, y_train)
    # print(f"dataset: {dataset}")
    train_dataset_loader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )
    # print(f"train_dataloader: {dataset}")

    valid_dataset = TensorDataset(X_valid, y_valid)
    valid_dataset_loader = DataLoader(
        valid_dataset,  # The training samples.
        sampler=RandomSampler(valid_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )
    return train_dataset_loader, valid_dataset_loader


def train_model_flow(dataloader, dev_dataloader):
    optimizer = Adafactor(model.parameters(),
                          lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=(1e-30, 1e-3),
                          relative_step=False)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    # predictions_all = []
    seed_val = 0

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # training_stats = []
    # total_t0 = time.time()

    for epoch_i in range(0, epochs):
        #               Training
        print("")
        print(f'======== Epoch {epoch_i + 1} / {epochs} ========')

        train_model(
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_i=epoch_i
        )

        validation_model(
            dataloader=dev_dataloader,
            epoch_i=epoch_i
        )
    writer.flush()
    writer.close()
    torch.save(model.state_dict(), model_path)


def train_model(
        dataloader,
        optimizer,
        scheduler,
        epoch_i):
    print('Training...')

    t0 = time.time()
    total_train_loss = 0

    model.train()
    for step, batch in enumerate(dataloader):

        if step % 1000 == 0 and not step == 0:
            elapsed = round(time.time() - t0, 3)
            print(f'  Batch {step} of {len(dataloader)}. Elapsed: {elapsed}, Loss: {loss.item():.2f}')

        b_input_ids = batch[0].to(device)
        b_labels = batch[1].to(device)

        model.zero_grad()

        output = model(input_ids=b_input_ids, labels=b_labels, return_dict=True)
        loss = output.loss
        logits = output.logits

        total_train_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(dataloader)
    writer.add_scalar("Loss/train", avg_train_loss, epoch_i)
    training_time = round(time.time() - t0, 3)
    print("")
    print(f"  Total_train_loss: {total_train_loss:.2f}")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))


def validation_model(
        dataloader,
        epoch_i):
    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_loss = 0
    # nb_eval_steps = 0

    for batch in dataloader:
        b_input_ids = batch[0].to(device)
        b_labels = batch[1].to(device)

        with torch.no_grad():
            output = model(input_ids=b_input_ids, labels=b_labels, return_dict=True)
            loss = output.loss
            logits = output.logits

        total_eval_loss += loss.item()

        # logits = logits.detach().cpu().numpy()
        # label_ids = b_labels.to('cpu').numpy()

    avg_val_loss = total_eval_loss / len(dataloader)
    writer.add_scalar("Loss/valid", avg_val_loss, epoch_i)
    validation_time = round(time.time() - t0, 3)

    print(f"  Validation Loss: {avg_val_loss:.2f}")
    print(f"  Validation took: {validation_time}")


def load_model():
    # device = torch.device("cuda")
    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    print(f"success load state dict from: {model_path}")
    return model


def pipeline_translation(file_name: str, action: str = "train"):
    data, target = load_translation_dataset(file_name)
    inputs_ids, labels = convert_translation_dataset(data, target)
    data_loader = convert_to_dataLoader(inputs_ids, labels)

    if action == "train":
        train_model_flow(data_loader)
    elif action == "validation":
        validation_model(data_loader)


def pipeline_train_model(file_name: str, use_dataset: str = "nlpcc", num: int = 2000):
    if use_dataset == "nlpcc":
        dataset = load_nlpcc_dataset(file_name)
        inputs_ids, labels = convert_nlpcc_dataset(dataset, num)
    else:
        dataset = load_lcst_dataset(file_name)
        inputs_ids, labels = convert_lcst_dataset(dataset, num)

    train_dataset_loader, valid_dataset_loader = convert_to_dataLoader(inputs_ids, labels)
    train_model_flow(train_dataset_loader, valid_dataset_loader)


if __name__ == '__main__':
    # model_name = "google/mt5-base"
    model_name = "google/mt5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    model.cuda()

    GPU_NUM = 0
    batch_size = 2
    epochs = 4
    data_max_length = 256
    summary_max_length = 256
    article_max_length = 512

    # model_path = "model_base.pt"
    model_path = "model_small.pt"

    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()
    # # torch.cuda.set_device(device)  # change allocation of current GPU
    # # print('Current cuda device ', torch.cuda.current_device())  # check
    #
    # train_data_name = "sts-train.tsv"
    # test_data_name = "sts-test.tsv"
    # dev_data_name = "sts-dev.tsv"
    #
    # # train
    # pipeline_translation(file_name=train_data_name, action="train")
    #
    # # test
    # # print(f"first test dataset")
    # # pipeline(file_name=test_data_name, action="test")
    # #
    # pipeline_translation(file_name=test_data_name, action="test")

    nlpcc_file_name = "train_with_summ.json"
    # pipeline_train_model(nlpcc_file_name, num=50000)

    lcst_file_name = "PART_I.html"
    pipeline_train_model(lcst_file_name, use_dataset="lcst", num=-1)

    # model = load_model()
    # pipeline_nlpcc(nlpcc_file_name, action="validation", num=10)

    """test mt5 generator"""
    gitlab_test_dataset = [
        [
            "烏法峰會將啟動印、巴加入程序上合組織擴員大門正式打開中新網北京7月6日電(記者張朔)上海合作組織成員國元首理事會第十五次會議本週在俄羅斯烏法舉行，中國國家主席習近平將出席。烏法峰會將通過關於啟動接收印度、巴基斯坦加入上合組織程序的決議，上合組織擴員的大門正式打開。中國外交部6日在北京舉行中外媒體吹風會，外交部副部長程國平介紹此次與會情況，並回答記者有關上合組織擴員等方面的提問。程國平說，上合組織成立14年來，不斷深化成員國間戰略協作和團結互信，全面推進各領域合作，已成為維護本地區安全穩定、促進共同發展的重要平台。去年杜尚別峰會以來，各成員國積極落實峰會成果，推動各領域合作進一步深化，取得一系列新成果，上合組織國際影響力不斷上升。程國平表示，隨著上合組織發展影響擴大，越來越多的域內國家紛紛提出加入。順應這些國家的利益需要並根據上合組織自身發展的需要，本次烏法峰會將通過關於啟動接收印度、巴基斯坦加入上合組織程序的政治決議，這就意味著上合組織擴員的大門正式打開，印、巴加入上合組織的法律程序啟動。程國平指出，我們也看到國與國之間難免存在著這樣、那樣的分歧與矛盾，這些都是歷史形成的，但並不影響國家間發展友好關係。印、巴加入上合組織不僅將對上合組織發展發揮重要作用，也將為促進印巴之間的友好關係不斷發展和改善發揮建設性作用。(完)",
            "上合組織元首峰會將在俄羅斯烏法舉行，將通過關於啟動接收印度、巴基斯坦加入上合組織程序的決議"
        ],
        [
            "發布日期：2015-03-2104:05:33岳陽市氣象台3月21日4時2分發布大霧橙色預警信號：汨羅市、臨湘市、岳陽縣、岳陽市區、湘陰縣、平江縣、華容縣已經出現能見度不足200米的霧並將持續，能見度低，請注意防範。圖例標準防禦指南6小時內可能出現能見度小於200米的濃霧，或者已經出現能見度小於200米、大於等於50米的濃霧且可能持續。1、有關部門和單位按照職責做好防霧工作；2、機場、高速公路、輪渡碼頭等單位加強調度指揮；3、駕駛人員必須嚴格控制車、船的行進速度；4、減少戶外活動。",
            "岳陽市發布大霧橙色預警：汨羅市、臨湘市、岳陽縣、岳陽市區、湘陰縣、平江縣、華容縣已經出現能見度不足200米的霧並..."
        ],
        [
            "發布日期：2015-07-1215:20:00滁州市氣像台2015年07月12日15時20分發布雷電黃色預警信號：目前我市西部有較強對流雲團向東南方向移動，預計6小時內我市部分地區將發生雷電活動，並可能伴有短時強降水、大風、局部冰雹等強對流天氣，請注意防範。圖例標準防禦指南6小時內可能發生雷電活動，可能會造成雷電災害事故。1、政府及相關部門按照職責做好防雷工作；2、密切關注天氣，盡量避免戶外活動。",
            "滁州市發布雷電黃色預警：目前我市西部有較強對流雲團向東南方向移動，預計6小時內我市部分地區將發生雷電活動，並可能..."
        ],
        [
            "發布日期：2015-05-1409:15:00咸寧市氣象台5月14日09時15分發布雷電黃色預警信號:預計未來6小時內，咸寧市區、嘉魚縣、通城縣、通山縣、崇陽縣、赤壁縣有雷電活動,可能造成雷電災害事故，請有關部門注意防範。圖例標準防禦指南6小時內可能發生雷電活動，可能會造成雷電災害事故。1、政府及相關部門按照職責做好防雷工作；2、密切關注天氣，盡量避免戶外活動。",
            "咸寧市發布雷電黃色預警：預計未來6小時內，咸寧市區、嘉魚縣、通城縣、通山縣、崇陽縣、赤壁縣有雷電活動,可能造成雷..."
        ],
        [
            "歐盟主席Donald Tusk今日表示，歐元區領導人將於下週一在布魯塞爾召開緊急會議，討論希臘局勢。Tusk表示：“現在是在最高政治層面緊急討論希臘局勢的時候了。 ”此外，有外媒報導稱，歐洲央行管委會將於週五就希臘緊急流動性援助(ELA)舉行未經事先安排的電話會議。此前，歐盟委員會副主席ValdisDombrocskis表示，歐元區財長們沒能在會議上就希臘問題達成一致。在經過了幾個小時的討論後，希臘和其債權人依然沒能救援助計劃達成協議。債權人表示，希臘必須遵守援助協議。而希臘總理齊普拉斯對此表示不贊同。德國時代周報今日在其網絡版上報導稱，希臘的債權人計劃延長其現有的援助計劃，直到今年年底，但IMF不會參與其中。該報表示，希臘現有援助計劃中用於銀行資本重組的100億歐元將被留下，用以償還其向歐洲央行以及IMF的欠款。對此，歐盟一位外交官向外媒表示，德國時代周報關於希臘援助獲得延期的報導“和事實毫無關係”。他澄清：“這與實際情況毫無關係，這樣的提案絕對不可能通過。 ”隨著當前救助協議臨近到期，希臘與債權人的談判進入倒計時。今日是達成協議的最佳時機。華爾街見聞文章此前提到，本週四的歐元區財長會議是希臘談判爭取突破的下一個機會。美銀美林的分析師預計，本週五以前如果不能敲定協議，本月底以前達成協議就相當困難。英國《金融時報》報導則是提到，很多歐洲官員認為，本週四如果不能達成協議，持續五個月的希臘債務僵局就會進入新的關鍵階段，屆時可能沒有足夠時間讓歐元區債權國的議會——特別是德國國會批准新協議。",
            "歐盟主席稱，歐元區領導人將召開緊急會議討論希臘局勢。歐元區財長們此前沒能在會議上就希臘問題達成一致。"
        ],
        [
            "本報訊大理將對在大理古城內從事生產經營及旅遊活動的單位和人員收取古城維護費，專項用於古城保護。近日，大理市人民政府發布第4號公告，出台《大理市大理古城保護管理辦法》（以下簡稱《辦法》），重申熱議多年的古城維護費，但具體徵收範圍、方式、收費程序等由古城保護管理機構制定徵收辦法，報大理市人民政府批准後才實施。據悉，《辦法》將於今年7月1日起實施。昨日，記者從大理市古城保護管理局了解到，《辦法》重點明確了古城保護範圍，把古城保護區分成了重點保護區、建設控制區和環境協調區。辦法重申，對古城內從事生產經營及旅遊活動的單位和人員收取古城維護費，專項用於古城保護。此外，對於在大理古城核心保護範圍內進行商業性影視攝製的，應按照利用補償原則，由利用單位或個人與古城保護管理機構簽訂有償利用協議。據了解，《辦法》還明確規定了古城重點保護區內建築的建設、修繕、審批等程序，在古城重點保護區內的經營項目實行準營證管理制度，在古城重點保護區內從事經營活動的單位和個人應當持有古城保護管理機構核發的《準營證》。對於消防安全，《辦法》專門指出，大理古城重點保護區禁止燃放煙花爆竹、燃燒火把、放孔明燈或使用易燃易爆危險物品，古城重點保護區內的單位、居民以及經營業主應按消防要求配備相應的消防器材和設施，生產生活中用電用火應當杜絕消防安全隱患。2009年就提出收古維費遊客每人次收30元據悉，關於大理古城維護費，2009年雲南省發改委、財政廳下發了對大理古城和洱海資源保護費收費標準的通知，稱雲南擬對進入大理古城的遊客徵收每人次30元的古城維護費，以籌集古城保護資金。大理將收取古城維護費的消息很快引發爭議，2009年大理州旅遊局局長馬金鐘在接受本報記者採訪時曾明確表示，大理兩年內不再提古城維護費。今年的全省兩會上，古城維護費的事在分組討論時再次被提及。大理州委書記梁志敏提出，今年將努力把大理古城打造升級成為5A級景區，還將考慮古城收費問題，收取的費用用於維護大理古城，大理州州長何華則表示，是否收費還在調研論證。“像麗江古城收取了維護費，所以古城內一些建設比較完善，大理古城的投入有限，差距很大。保護民居的基礎設施方面投入非常巨大。我個人認為每一個到大理的人對促進當地的發展，維護當地的文化、生態、建設都負有責任。 ”何華在接受采訪時表示，大理古城是世界的古城，不只是當地人要去維護，也希望每個到大理的遊客能和當地人一起來維護。對於大理古城維護費的收取，他表示最重要的是聽取多方聲音，多次調研論證後才會做出具體佈署。",
            "大理古城7月1日起開收古城維護費，對生產經營及旅遊活動的單位和人員收取。"
        ],
        [
            "據彭博社報導，一家酒店管理公司TianfangHospitalityManagement計劃將一個資產組合為中國三家酒店的信託，在本地進行首次公開售股（IPO），募集約5億元。知情人士透露，這只信託可能最早9月在新加坡上市。這只房地產投資信託的資產由三家酒店構成，分別是位於天津的麗思卡爾頓和天沐溫泉度假酒店，還有位於海南的三亞海棠灣天房洲際度假酒店。根據彭博社的數據，今年新加坡交易所IPO所籌集的資金僅4100萬美元（5526萬新元），去年同期是7億2900萬美元。我國上市房地產投資信託指數今年累計下滑0.1%，而海峽時報指數則下跌0.4%。",
            "外媒曝中國3家酒店擬組信託，公開售股集資5億元；據悉，這只信託最早9月在新加坡上市。"
        ],
        [
            "李登輝（資料圖）【環球軍事報導】92歲的李登輝21日起訪問日本6天，並將前往日本國會議員會館發表演講。據台灣“中央社”報導，李登輝當地時間21日下午6時左右抵達東京。此次訪日共有6天行程，他將於22日在日本國會議員會館向國會議員發表演說。李登輝稱，他演講的內容包括台灣典範的變化，以及民主化和自由化等，“讓日本人了解這些關於台灣的議題”。23日與日本外國特派員協會進行午餐會，之後將參訪福島和仙台，26日前往岩沼寺千年希望之丘，向“3·11地震”罹難者獻花。聯合新聞網稱，李登輝曾於2007年造訪他最喜愛的“奧之細道”，並以日文寫下“松島光與影，炫目之光”的俳句，妻子曾文惠也寫下俳句，被製作成“句碑”。李登輝此行將在句碑旁植樹。“中央社”稱，夫人曾文惠因為生病不能同行，孫女李坤儀陪同李登輝到訪日本。此次日本行，是李登輝卸任以來第七次訪問日本，也是首次在日本國會議員會館演講。眾所周知，李登輝有著強烈的“日本情結”，他曾表示自己20歲以前是日本人，還有個日本名字叫岩里政男。1943年李登輝在台北讀完高中後到日本京都帝國大學留學，學習農業經濟。留學期間，他加入過日軍，成為名古屋高射砲部隊陸軍少尉時迎來了日本戰敗。2001年是李登輝卸任台灣領導人後首次訪日，名義是治療心髒病;2007年訪問時甚至參拜了靖國神社，他還多次宣揚“釣魚台(指釣魚島)是日本的”論調，遭到島內輿論的抨擊。(崔明軒)",
            "92歲李登輝昨起訪問日本6天，將向國會議員發表演講；這是他卸任後第七次訪日，2007年曾參拜靖國神社。"
        ],
        [
            "2015-06-2316:06 新浪財經顯示圖片新浪財經訊6月23日消息，希臘債務危機現轉機，歐美股市上漲，受此利好今日港股恆指高開0.17%，曾受A股下挫拉低，但觸底之後一路上揚，收漲0.93%，升252.610點，收報27333.459點；國企指數跑贏大市，收漲1.69%，升225.79點，收報13609.471點；紅籌指數收漲，報點。全日大市成交1359.916億港元。藍籌股普遍上漲，華潤電力領漲藍籌股，漲3.828%，報21.7港元，成交1.319億港元。中煤能源漲2.857%，聯想漲1.596%，中國移動漲3.153%，長和漲1.546%，港交所漲1.058%，騰訊控股漲0.701%。航空股大漲，東方航空H股漲6.65%，報7.06港元，成交1987萬股，成交額1.38億港元，H股較A股折價55.6%。南方航空漲5.78%，中國國航漲4.73%。內銀表現不俗，招商銀行漲4.752%，重慶銀行漲6.43%，盛京銀行漲8.26%，民生銀行漲3.38%，中國銀行漲3.05%，工商銀行漲2.79%，中國光大銀行漲1.96%，建設銀行漲1.96%，交通銀行漲1.89%，農業銀行漲1.94%內房股普漲，恆大地產漲5.83%，遠洋地產漲5.43%，融創中國漲5.00%，雅居樂地產漲3.99%，深圳控股漲3.55%，龍湖地產漲3.48%，華潤置地漲2.81，合景泰富漲3.19%。名家點評耀才證券市務總監郭思治表示，港股之表現則相對為穩，恆指早市先高開45點至27081點，其後在買盤追捧下再升至27243點方漸受規限，從技術上看，其時恆指不但已初步重越10天線（26858點），且更進一步輕越20天線（27232點），即是說，大市之回穩勢頭已進一步明顯，惟最關鍵的仍是50天線（27538點），一日恆指仍未能成功重越50天線之前，整個市勢仍未能擺脫反覆不明之趨勢。指數，都創出了新高。",
            "港股恆指收漲0.93%收報27333點，國企指數跑贏大市收漲1.69%，招商銀行領漲國企股"
        ],
        [
            "據上海市黃浦區人民政府新聞辦官方微博@上海黃浦消息，依據《中華人民共和國突發事件應對法》、《上海市實施〈中華人民共和國突發事件應對法〉辦法》，黃浦區政府對遇難人員家屬和受傷人員負有依法履行救助、撫慰的法定義務。本著“依法依規、合情合理、實事求是、一視同仁”的原則，黃浦區政府會同有關社會組織共同研究制定了外灘擁擠踩踏事件遇難人員家屬救助方案。對此次事件遇難人員家屬的救助撫慰金確定為人民幣80萬元。其中，50萬為政府救助撫慰金，30萬為社會幫扶金。傷殘人員的救助撫慰金額，將根據傷員救治、傷情和傷殘鑑定等具體情況另行確定。",
            "上海外灘踩踏事件遇難者家屬將獲80萬撫慰金，其中50萬元為政府救助撫慰金，30萬元為社會幫扶金。"
        ]
    ]

    # gitlab_test_dataset = [
    #     [
    #         # "中央流行疫情指揮中心今(11)日公布國內新增7例本土COVID-19確定病例，1例為案1187之接觸者(案1201)，6例感染源待釐清(案1202、案1203，案1208至案1211)，並包括1起遊藝場群聚事件。指揮中心表示，案1201為本國籍40多歲女性，為案1187同住家人，近期無出國史，亦無疑似症狀，因案1187檢驗確診，5月9日經衛生單位安排採檢後居家隔離，並於今日確診(Ct值22)。目前初步掌握個案接觸者3人，皆已匡列為案1187之接觸者，故不重複匡列。",
    #         "中央流行疫情指挥中心今(11)日公布国内新增7例本土COVID-19确定病例，1例为案1187之接触者(案1201)，6例感染源待厘清(案1202、案1203，案1208至案1211)，并包括1起游艺场群聚事件。指挥中心表示，案1201为本国籍40多岁女性，为案1187同住家人，近期无出国史，亦无疑似症状，因案1187检验确诊，5月9日经卫生单位安排采检后居家隔离，并于今日确诊(Ct值22)。目前初步掌握个案接触者3人，皆已匡列为案1187之接触者，故不重复匡列。",
    #         "增7例本土個案 6例不明感染源"
    #     ]
    # ]

    # model = load_model()
    # # dataset = load_nlpcc_dataset()
    # # inputs_ids, labels = convert_nlpcc_dataset(dataset, 10)

    inputs_ids, labels = convert_custom_dataset(gitlab_test_dataset)
    test_path = "test1.txt"
    with open(test_path, "w", encoding="utf-8") as f:
        for i in range(10):
            output = model.generate(inputs_ids[i].cuda().reshape(1, -1), min_length=30, max_length=130, num_beams=1)
            generate_text = tokenizer.decode(output[0], skip_special_tokens=True)
            summary_text = tokenizer.decode(labels[i], skip_special_tokens=True)
            print(f"generate text: {generate_text}")
            print(f"summary text:  {summary_text}")
            f.write(f"generate text: {generate_text}\n")
            f.write(f"summary text: {summary_text}\n")
