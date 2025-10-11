import os
import re
import statistics
import MeCab
import pandas as pd
from gensim.models import KeyedVectors

# MeCabオブジェクトの生成
neologd_path = "/usr/local/lib/mecab/dic/mecab-ipadic-neologd" # インストール時に表示されたパス
mt = MeCab.Tagger("-d " + neologd_path)
#mt.parse('')

# w2v model
model_path = '/Users/psychetmdu/Desktop/w2V/entity_vector/entity_vector.model.txt'
wv = KeyedVectors.load_word2vec_format(model_path, binary=False)

'''
2024/11/26　cosine similarity between two adjacent content words vectorized by word2Vec.
This code is based on "cosineSimilarity.py" which uses sentenceBERT.

This is one of the NLP #1 analyses, in the part of word2Vec.
It has two ways: round-robin fashion and adjacent.
This code is the way of adjacent.
'''

def readPatientPart(file):
    with open(file, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        #print("lines:", lines)
        lines_strip = [line.rstrip() for line in lines]  # 改行文字を除く
        #print("lines_strip:", lines_strip)
        line_num = [i for i in range(len(lines_strip))]
        #print("行番号:", line_num)
        p_line_num = [i for i, moji in enumerate(lines_strip) if '●患者' in moji]  # 「患者」を含む行番号を取得
        #print("患者の行番号:", p_line_num)
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
        #print("patient_lines(line52):", patient_lines)
        return patient_lines


def extractContentWords(PATIENT_lines):
    text_list = []
    for line in PATIENT_lines:
        node = mt.parseToNode(line.strip())
        #print('mt:', mt)
        #print('node:', node)
        while node:
            fields = node.feature.split(",")
            #print('fields:', fields)
            if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞' or fields[0] == '副詞':
                #print('node.surface:', node.surface)
                #print('node.feature:', node.feature)
                if len(fields) > 6:
                    lemma = fields[6]  # 原形情報はfeatureの6番目の要素
                    #print('fields[6]:', lemma)
                else:
                    print('No lemma information available.')
                    lemma = node.surface  # 元の形態素をlemmaとして使用
                #print('lemma:', lemma)
                text_list.append(lemma)
                #print('text:', text_list)
                #print('node:', node)
            node = node.next
    return text_list


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

'''
def calculate_adjacent_similarity(word_list, wv):
    similarity_list = []
    try:
        for i in range(len(word_list) - 1 ):
            #　隣接する単語の類似度を計算
            sim = wv.similarity(word_list[i], word_list[i+1])
            similarity_list.append(sim)
    except Exception as e:
        print(f"エラーが発生しました: {e}")
    return similarity_list
'''

def calculate_adjacent_similarity(word_list, wv):
    similarity_list = []
    for i in range(len(word_list) - 1):
        try:
            # 辞書に単語が存在するか確認
            if word_list[i] in wv and word_list[i+1] in wv:
                # 隣接する単語の類似度を計算
                sim = wv.similarity(word_list[i], word_list[i+1])
                similarity_list.append(sim)
            else:
                # 辞書に単語が存在しない場合はスキップ
                print(f"単語が辞書に存在しません: {word_list[i]}, {word_list[i+1]}")
        except Exception as e:
            print(f"エラーが発生しました: {e}")
    return similarity_list


def calculate_features_duplicates(similarity_list):
    # similarity = 1, 重複する要素の数を数える
    duplicates_adj = similarity_list.count(1)
    # duplicates_adjの割合を調べる
    duplicates_rate_adj = duplicates_adj / len(similarity_list)
    # 重複を除いたsimilarity_listを作成
    similarity_list_ex_duplicates_adj = [value for value in similarity_list if value != 1]
    return duplicates_adj,duplicates_rate_adj, similarity_list_ex_duplicates_adj


def countFeatures(similarity_list_ex_duplicates_adj, duplicates_adj, duplicates_rate_adj, unique_num_adj, subjectID, visit, part) :
    alist  = [float(item) for item in similarity_list_ex_duplicates_adj]
    try:
        mean = statistics.mean(alist)
        median = statistics.median(alist)
        variance = statistics.variance(alist,mean)
        SD = statistics.stdev(alist,mean)
        MAX = max(alist)
        MIN = min(alist)
    except:
        mean = 0
        median = 0
        variance = 0
        SD = 0
        MAX = max(alist)
        MIN = min(alist)

    feature_dic ={
        'ID': subjectID,
        'Visit': visit,
        'Part': part,
        'duplicates_adj': duplicates_adj,
        'duplicates_rate_adj': duplicates_rate_adj,
        'w2V_adj_mean': mean,
        'w2V_adj_variance': variance,
        'w2V_adj_SD': SD,
        'w2V_adj_MAX': MAX,
        'w2V_adj_MIN': MIN,
        'w2V_adj_median': median,
        'unique_num_adj': unique_num_adj}

    return feature_dic

def fileSortForBERT(path, dataframe):
    os.chdir(path)
    file = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]
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
                # ここからw2V処理
                print('file:', file[i])
                try:
                    # 患者部分のみ発話パート抽出してリスト化
                    PATIENT_lines = readPatientPart(file[i])
                    print('PATIENT_lines:',PATIENT_lines)
                    #　内容語のみ抽出してリスト化
                    text_list = extractContentWords(PATIENT_lines)
                    print('text_list:', text_list)
                    #  隣接する内容語間の類似度を計算
                    similarity_list = calculate_adjacent_similarity(word_list=text_list, wv=wv)
                    print("similarity_list:", similarity_list)
                    #  重複、重複の割合、重複を除いた類似度のリストを算出
                    duplicates_adj, duplicates_rate_adj, similarity_list_ex_duplicates_adj = calculate_features_duplicates(similarity_list)
                    #  固有の内容語数を計算
                    unique_text_list = set(text_list)
                    unique_num_adj = len(unique_text_list)
                    print("unique_num_adj:", unique_num_adj)
                    features_1 = {"duplicates_adj":duplicates_adj,
                                  "duplicates_rate_adj":duplicates_rate_adj,
                                  "unique_num_adj":unique_num_adj}
                    dic = {'ID': subid, 'Visit': subvisit, 'Part': subpart}
                    print("dic:", dic)
                    dic.update(**features_1)
                    df_Features = countFeatures(similarity_list_ex_duplicates_adj=similarity_list_ex_duplicates_adj,
                                                duplicates_adj=duplicates_adj, duplicates_rate_adj=duplicates_rate_adj,
                                                unique_num_adj=unique_num_adj, subjectID=subid, visit=subvisit,
                                                part=subpart)
                    dic.update(**df_Features)
                    print(dic)
                    df = pd.DataFrame(data=dic, index=['val', ])
                    dataframe = pd.concat([df, dataframe], axis=0)
                except:
                    print('Errorです。', file[i])
            except:
                print('stop@:',catch_id)
                print('stop@2',subid)
        else:
            print('Errorを拾っている可能性があります。', file[i])
    return dataframe




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
    #　例文解析
    text="これは例文です。"
    # 　内容語のみ抽出してリスト化
    ex_text_list = extractContentWords(text)
    print('text_list:', ex_text_list)
    #  隣接する内容語間の類似度を計算
    ex_similarity_list = calculate_adjacent_similarity(word_list=ex_text_list, wv=wv)
    print("similarity_list:", ex_similarity_list)
    #  重複、重複の割合、重複を除いた類似度のリストを算出
    duplicates_adj, duplicates_rate_adj, similarity_list_ex_duplicates_adj = calculate_features_duplicates(ex_similarity_list)
    #  固有の内容語数を計算
    unique_text_list = set(ex_text_list)
    unique_num_adj = len(unique_text_list)
    print("unique_num_adj:", unique_num_adj)
    ex_features_1 = {"duplicates_adj": duplicates_adj,
                  "duplicates_rate_adj": duplicates_rate_adj,
                  "unique_num_adj": unique_num_adj}
    ex_dic = {'ID': 0, 'Visit': 0, 'Part': 0}
    print("ex_dic:", ex_dic)
    ex_dic.update(**ex_features_1)
    ex_Features = countFeatures(similarity_list_ex_duplicates_adj=similarity_list_ex_duplicates_adj,
                                duplicates_adj=duplicates_adj, duplicates_rate_adj=duplicates_rate_adj,
                                unique_num_adj=unique_num_adj, subjectID=0, visit=0,
                                part=0)
    ex_dic = {'ID': 'example', 'Visit': 0, 'Part': 0}
    print("ex_features:", ex_Features)
    ex_dic.update(**ex_Features)
    ex_df = pd.DataFrame(data=ex_dic, index=['val',])
    print(ex_df)

    PATH = '/Users/psychetmdu/Desktop/RecordVoice_yg_R6Nov22'
    df = fileSortForBERT(PATH, ex_df)
    print('df:', df)
    df.to_csv('Dataframe_w2v_adj_R6Dec6.csv')
    print(checkFileAnalyzedOrNot(PATH, df))