from transformers import BertJapaneseTokenizer, BertModel
import torch
import scipy.spatial
import statistics
import pandas as pd
import fugashi
import ipadic
import glob
import os
import re
from striprtf.striprtf import rtf_to_text

"""
2022/7/26 実装完了　sentence, query内の単語・文章をいろいろ変えて実験①
2022/8/2 sentence, query内の単語・文章をいろいろ変えて実験②
2022/10/14　全ての実装完了。testファイルでの実行問題なし。
2024/04/23 微修正: list内の0.05を下回る数値をperseverationとして捕捉。list内演算はそれを除いて実行
コサイン距離：0　意味が近い　1　意味が遠い

2024/04/23 10:10　実行
実行時：
・sBERTによるcosine similarity計算
・Path内に配置されたUNDERPIN dataに対して集計を行う
"""


class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest",
                                                             truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)


model = SentenceBertJapanese("sonoisa/sentence-bert-base-ja-mean-tokens")


def readPatientPart(file):
    with open(file, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        lines_strip = [line.rstrip() for line in lines]  # 改行文字を除く
        line_num = [i for i in range(len(lines_strip))]
        p_line_num = [i for i, moji in enumerate(lines_strip) if '●患者' in moji]  # 「患者」を含む行番号を取得
        d_line_num = [i for i, moji in enumerate(lines_strip) if '●医師' in moji]
        target_num = p_line_num
        for i in range(len(p_line_num)):
            values = [e for e in d_line_num if e > p_line_num[i]]
            if values:
                if p_line_num[i] == (min(values) - 1):
                    continue
                else:
                    add_num = [j for j in range(len(lines_strip)) if (p_line_num[i] < j < min(values))]
                    target_num = target_num + add_num
            else:
                larger_num = [y for y in line_num if (y > p_line_num[i])]
                target_num = target_num + larger_num
        target_num = sorted(target_num)
        patient_lines = [s.replace('●患者', '') if s.startswith('●患者') else s for s in
                         [moji for i, moji in enumerate(lines_strip) if i in target_num]]
        patient_lines = [re.sub("\(.+?\)", "", s) if '(' in s else s for s in patient_lines]
        return patient_lines


def countSimilarity(x, y):
    X = model.encode([x])
    Y = model.encode([y])
    distance = scipy.spatial.distance.cdist(X, Y, metric="cosine")[0]
    similarity = distance
    return similarity

"""
def gogoSimilarity(list):
    distance = scipy.spatial.distance.cdist(list[0], list[1], metric="cosine")[0]
    return distance
"""

def adjacentTwo(list):
    # list: list of type(str)
    # 隣接する文章のセットを作る。
    ad_list = [[list[i], list[i + 1]] for i in (range(len(list) - 1))]
    return ad_list

def countFeatures(data, subjectID, visit, part):
    #　コサイン距離なので、意味的に近いほど0に近く。
    # 10^-10以下の値の個数をカウント
    perseveration = sum(1 for x in data if x <= 0.05)

    # 1未満の値のみを含む新しいリストを作成
    new_list = [x for x in data if x > 0.05]
    print('new_list:', new_list)

    # 新しいリストに対して統計量を計算
    mean = statistics.mean(new_list) if new_list else float('nan')
    median = statistics.median(new_list) if new_list else float('nan')
    variance = statistics.variance(new_list) if len(new_list) > 1 else float('nan')
    SD = statistics.stdev(new_list) if len(new_list) > 1 else float('nan')
    MAX = max(new_list) if new_list else float('nan')
    MIN = min(new_list) if new_list else float('nan')

    # 結果をデータフレームにまとめる
    dataframe = pd.DataFrame({
        'ID': [subjectID],
        'Visit': [visit],
        'Part': [part],
        'senti_sim_mean': [mean],
        'senti_sim_var': [variance],
        'senti_sim_sd': [SD],
        'senti_sim_max': [MAX],
        'senti_sim_min': [MIN],
        'senti_sim_med': [median],
        'perseveration': [perseveration]
    })
    return dataframe


def fileSortForBERT(path):
    df = pd.DataFrame({'ID': [], 'Visit': [], 'Part': [],
        'senti_sim_mean': [],
        'senti_sim_var': [],
        'senti_sim_sd': [],
        'senti_sim_max': [],
        'senti_sim_min': [],
        'senti_sim_med': [],
        'perseveration':[]})
    os.chdir(path)
    file = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]
    '''
    id = 'U*'
    visit = '*'
    part = '*'
    format = '*.txt'
    name = id + '-' + visit + '_' + part + format
    file = glob.glob(path + '/**/' + name, recursive=True)
    '''
    print('file:',file)

    for i in range(len(file)):
        target1 = '.txt'
        target2 = 'Kana.txt'
        if ((target1 in file[i]) == True) or ((target2 in file[i]) == True):
            try:
                catch_id = re.findall(r'U[A-Z]{2}[0-9]{3}.[1-5].[1-3]', file[i])
                subid = catch_id[0][0:6]
                subvisit = catch_id[0][7]
                subpart = catch_id[0][9]
                df_sub = pd.DataFrame({'ID': [subid], 'Visit': [subvisit], 'Part': [subpart]})
                #df = pd.concat([df, df_sub])
                #print('df', df)
                # ここからBERT処理
                print('file:', file[i])
                try:
                    PATIENT_lines = readPatientPart(file[i])
                    print('PATIENT_lines:',PATIENT_lines)
                    #subID = 'experiment'
                    ad_list = adjacentTwo(PATIENT_lines)
                    print('ad_list:',ad_list)
                    similarity_list = [float(countSimilarity(ad_list[i][0], ad_list[i][1])) for i in range(len(ad_list))]
                    print('similarity_list', similarity_list)
                    df_Features = countFeatures(similarity_list, subid, subvisit, subpart)
                    print('df_Features:', df_Features)
                    df = pd.concat([df,df_Features])
                except:
                    print('Errorです。', file[i])
            except:
                print('stop@:',catch_id)
                print('stop@2',subid)
        else:
            print('Errorを拾っている可能性があります。', file[i])
    return df


def checkFileAnalyzedOrNot(directory_path, dataframe):
    # dataframeからID, Visit, Partを読み取り、file_name = "ID-Visit_Part"を作成
    analyzed_file_list = [f"{row['ID']}-{row['Visit']}_{row['Part']}" for index, row in dataframe.iterrows()]

    # 正規表現パターン
    pattern = r'U[A-Z]{2}[0-9]{3}.[1-5].[1-3]'

    # '.txt' 形式のファイルのパスを格納するリスト
    all_txt_file = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.txt')]

    # ディレクトリ内の全てのファイルのパスを格納するリスト
    all_path = [os.path.join(directory_path, f) for f in os.listdir(directory_path)]

    # all_pathから正規表現にマッチするID部分を抽出し、リスト化
    all_id_list = []
    for path in all_path:
        match = re.search(pattern, path)
        if match:
            all_id_list.append(match.group(0))

    # all_pathに含まれ、all_txt_fileに含まれない要素のみを持つリスト
    rest_path_txt_all = [f for f in all_path if f not in all_txt_file]

    # all_id_listに含まれ、analyzed_file_listに含まれない要素のみを持つリスト
    rest_path_analyzed_all = [id for id in all_id_list if id not in analyzed_file_list]

    print("TXT Files:", all_txt_file)
    print("txtファイル数:", len(all_txt_file))
    # print("All Files:", all_path)
    print("rest files(all-.txt):", rest_path_txt_all)
    print("実際に解析対象となったファイル数:", len(analyzed_file_list))
    print("rest files(all-analyzed):", rest_path_analyzed_all)

    return rest_path_analyzed_all

if __name__ == '__main__':
    PATH = '/Users/psychetmdu/Desktop/RecordVoice_yg_R6Nov22'
    df = fileSortForBERT(PATH)
    print('df:', df)
    df.to_csv('Dataframe_sBERT_R6Nov22.csv')
    print(checkFileAnalyzedOrNot(PATH, df))