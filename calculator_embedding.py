#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

import requests


def call_embedding_api(text1: str, text2: str) -> List[float]:
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

    url = "http://10.20.50.10:5000/embedding"
    resp = requests.post(url=url,
                         json=input_json,
                         headers={'Content-Type': 'application/json'})
    if resp.status_code == 200:
        result = resp.json()["result_list"]
        return [r["embedding"] for r in result]
    else:
        print(f"Error request status_code : {resp.status_code}\n resp is : {resp}")


if __name__ == '__main__':
    # text1 = "蘋果下半年新款iPhone設計外界關注。分析師預期，6.1吋新款LCD版iPhone的上天線，可能維持MPI與LCP混合設計；新iPhone的NFC軟板天線，也會升級到4層板。天風國際證券分析師郭明錤報告預估，今年下半年6.1吋新款LCD版iPhone的上天線（UAT），可能仍維持軟板異質PI（ModifiedPI）與液晶聚合物LCP（Liquidcrystalpolymer）混合的設計。報告指出，原因在於新6.1吋LCD版iPhone是下半年新機型中的低階機種，MPI成本較LCP低；此外若全採用LCP設計，日本廠商村田製作所（Murata）將成獨家供應商，供應風險高；再者調整設計後，MPI仍能符合蘋果技術標準。在軟板設計部分，報告預期，今年下半年新款iPhone的近距離無線通訊（NFC）軟板天線，會從2層板升級到4層板，價格也明顯增加。展望明年新款iPhone設計，報告預期明年下半年新款iPhone將支援5G通訊、且LCP用量將增加，因此預期蘋果需要更多LCP供應商，降低供應風險。此外，明年下半年高階新款有機發光二極體（OLED）版iPhone，將採用Y-OCTA面板觸控技術與COP（ChipOnPi）封裝技術，可降低成本、減少面板厚度與縮小邊框，有利外觀設計。今年新款iPhone規格功能各界關注，分析師預估相機鏡頭升級是今年下半年新iPhone最大賣點之一。其中6.5吋OLED版、5.8吋OLED版和6.1吋LCD版3款，後置相機分別升級到3顆鏡頭、3顆鏡頭與雙鏡頭。市場一般預期，今年新iPhone可能推出6.5吋OLED螢幕、5.8吋OLED螢幕以及6.1吋LCD螢幕等3款機種，螢幕上方的「瀏海」面積與去年機種相同。今年新iPhone將支援雙向無線充電，可能均採用Lightning連接線，並無支援USBType-C功能。"
    # text2 = "2019下半年將到來，多家手機大廠的旗艦新機也將一一與消費大眾見面，爆料大神EvanBlass日前則揭露美國電信商Verizon針對三星、蘋果、Google的新機行銷規劃。在官方正式發表前，手機大廠的年度新機總是傳聞不斷，其中Google相當乾脆地公開了今年將發表的Pixel4宣傳照，讓好奇的消費者搶先得知外觀設計、進一步推測可能具備的功能。而在同一天爆料大神EvanBlass於個人Twitter帳號上揭露美國電信商Verizon針對重點新機的行銷規劃截圖，從下圖可看出，除了已在5月推出Pixel3a系列以外，包括三星GalaxyNote10、蘋果iPhone、以及GooglePixel4都在計畫當中，預計是8月下旬發表GalaxyNote10、9月下旬發表iPhone、10月中旬發表Pixel4份行銷規劃剛好呼應了Google提早4個月就先公開一小部份Pixel4的新機資訊；此外值得注意的是，圖表上只列出「iPhone」發表，回顧去年iPhoneXR是在iPhoneXS、iPhoneXSMax上市一個月後才開賣，或許今年推出的3款全新iPhone可能在同一時間上市。而從3款新機的發表時程來看，並不令人感到意外，參考過去的發表時程如iPhone在9月份、Pixel在10月份並未偏離預期範圍，唯獨只是正式發表的日期究竟在哪一天尚未公佈。然而對於這3款系列機種來說，推出新機大部分著重在內部規格、相機功能等升級，並未有突出的創新設計，於是外界笑稱去年最大的創新在於價格創新高。若以最受國人矚目的iPhone來說，據瞭解從供應鏈端確實沒聽說新款iPhone有何有趣的亮點，甚至可說是沒有討論度。現今的智慧型手機已能滿足用戶的日常需求，那麼如何讓消費者願意掏錢換機，則考驗手機大廠能否創新應用服務、或提升使用體驗。"

    text1 = "上合組織元首峰會將在俄羅斯烏法舉行，將通過關於啟動接收印度、巴基斯坦加入上合組織程序的決議"
    # text2 = "烏法峰會將啟動印、巴加入程序上合組織擴員大門正式打開中新網北京7月6日電(記者張朔)上海合作組織成員國元首理事會第十五次會議本週在俄羅斯烏法舉行，烏法峰會將通過關於啟動接收印度、巴基斯坦加入上合組織程序的決議，本次烏法峰會將通過關於啟動接收印度、巴基斯坦加入上合組織程序的政治決議，"
    # text2 = "7月6日，中國新聞社啟動了擴大上海合作組織聲援之門，並啟動了國務會議國務委員會第十五次會議，國家元首，國務會議，國務會議。俄羅斯國務委員會，印度國務委員會，巴基斯坦國務委員會，中國國務委員會，俄羅斯國務委員會，印度國務委員會，中國國務委員會，俄羅斯國，印度國務委員會，巴基斯坦國務委員會，中國國務委員會，國務委員會"
    # text2 = "上海合作組織國家元首理事會第十五次會議將於本週在俄羅斯烏法舉行。阿聯酋首腦會議將通過一項決議，啟動啟動接受印度和巴基斯坦加入上海合作組織的程序。"
    # text2 = "烏法首腦會議將開始印度和巴基斯坦的加入程序。上合組織成員擴大的大門正式拉開。 7月6日，北京，中國新聞網（記者張碩）上海合作組織國家元首理事會第十五次會議本週將在俄羅斯烏法舉行。中國國家主席習近平將出席。烏法首腦會議將通過一項決議，啟動啟動接受印度和巴基斯坦加入上海合作組織的程序，並正式開啟大門。中國外交部為中國人舉行了情況通報會"
    text2 = "上海合作組織國家元首理事會第十五次會議將於本週在俄羅斯烏法舉行。上海合作組織是維護該地區安全與穩定，促進共同發展的重要平台。"

    embedding = call_embedding_api(text1=text1, text2=text2)
    cos = cosine_similarity([embedding[0]], [embedding[1]])
    print(cos)

    # extract [[0.70441658]]
    # pegusas-sum [[0.48648079]]
    # t5 [[0.77415469]]
    # bart-base [[0.78222111]]
    # distilbart [[0.55600899]]
