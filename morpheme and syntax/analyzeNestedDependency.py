# -*- coding: utf-8 -*-
import spacy
import pandas as pd
import glob
import os
import re
from exJapaneseFeatures import JapaneseFeatureExtractor

"""
2023/08/15
https://note.com/npaka/n/n5c3e4ca67956
https://dev.classmethod.jp/articles/try-parsing-using-ginza/
https://megagonlabs.github.io/ginza/
を参照。
UNDERPIN fileに対してGINZA analysisを実行してデータフレームを出力

2024/01/11
GINZA解析はexJapaneseFeatures.pyから引っ張ってくるように変更
解析チェックの関数を追加。

2024/01/26
GINZA: max_bytes=49149
長文すぎるとGinzaは解析できない。
しかし、arrowsが短すぎると構文解析できずにarrows＝[]となる。
そのため、以下のようにした。

Patient_linesを用いて、1応答ずつ解析。
arrowsが短すぎる場合にはdepths = 0　とする。
"""

def analyze_sentence(sentence, nlp):
    # 短い文の場合は features を 0 で初期化
    if len(sentence.strip()) == 0:
        return {"max_nesting_relation": 0, "max_nesting_depth": 0}
    try:
        # 日本語モデルをロード, nlp
        # 解析を実行
        doc = nlp(sentence)

        #　係り受け関係　は考慮せず
        # 係り受け関係を格納
        arrows = [(token.i, token.head.i) for token in doc if token.i < token.head.i]
        if not arrows:
            # 矢印リストが空の場合は、features を 0 で返す
            return {"max_nesting_relation": 0, "max_nesting_depth": 0}

        # 各矢印についてnesting_relationを計算
        nesting_relations = calculate_nesting_relations(arrows)
        # 最大のnesting_relationを求める
        max_nesting_relation = max(nesting_relations) if nesting_relations else 0
        # 最大の入れ子構造の深さを求める
        max_nesting_depth = max(calculate_nesting_depth(arrows, start, end) for start, end in arrows)

        features = {
            "max_nesting_relation": max_nesting_relation,
            "max_nesting_depth": max_nesting_depth
        }
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"max_nesting_relation": 0, "max_nesting_depth": 0}
    return features

def analyze_text_and_get_max_features(sentence_list, nlp):
    # 最大値を格納するための変数を初期化
    max_nesting_depth = 0
    max_nesting_relation = 0

    # リスト内の各文に対してanalyze_sentenceを実行
    for sentence in sentence_list:
        features = analyze_sentence(sentence, nlp)
        # 最大値を更新
        max_nesting_depth = max(max_nesting_depth, features["max_nesting_depth"])
        max_nesting_relation = max(max_nesting_relation, features["max_nesting_relation"])

    features = {
            "max_nesting_relation":max_nesting_relation,
            "max_nesting_depth":max_nesting_depth
        }
    return features

def calculate_nesting_relations(arrows):
    nesting_relations = []
    for i, (src, tgt) in enumerate(arrows):
        nesting_relation = 0
        for src2, tgt2 in arrows:
            #print("src, src2, tgt2, tgt:", src,src2,tgt2,tgt)
            if src2 <= tgt2 and src <= tgt and (
                    (src < src2 and tgt > tgt2) or (src == src2 and tgt > tgt2) or (tgt == tgt2 and src < src2)):
                nesting_relation += 1
            elif src2 >= tgt2 and src >= tgt and ((src == src2 and tgt < tgt2) or (tgt == tgt2 and src > src2)):
                nesting_relation += 1
        nesting_relations.append(nesting_relation)
    return nesting_relations

def calculate_nesting_depth(arrows, start, end, depth=0):
    nested_depths=[depth]
    for src, tgt in arrows:
        # 同じ向き同士の係り受け関係の入れ子構造のみ考慮
        if start <= end and src <= tgt and ((src < start and tgt > end) or (src == start and tgt > end) or (tgt == end and src < start)):
            nested_depths.append(calculate_nesting_depth(arrows, src, tgt, depth + 1))
            #print("src, start, end, tgt:",src,start,end,tgt)
        elif start >= end and src >= tgt and ((src > start and tgt < end) or(src == start and tgt < end) or (tgt == end and src > start)):
            nested_depths.append(calculate_nesting_depth(arrows, src, tgt, depth + 1))
            #print("src, start, end, tgt:",src, start, end, tgt)
    return max(nested_depths)

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


def fileSortForAnalaysisNestedDependency(path, dataframe, feature_extractor, nlp):
    # ディレクトリを移動
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
                print('file:', file[i])
                try:
                    PATIENT_lines = readPatientPart(file[i])
                    print('PATIENT_lines:',PATIENT_lines)
                    features = analyze_text_and_get_max_features(PATIENT_lines, nlp)
                    dic = {'ID': subid, 'Visit': subvisit, 'Part':subpart}
                    dic.update(**features)
                    print(dic)
                    df = pd.DataFrame(data=dic, index=['val', ])
                    dataframe = pd.concat([df, dataframe], axis=0)
                except:
                    print('Errorです。')
            except:
                print('stop@:',catch_id)
                print('stop@2',subid)
        else:
            print('Errorを拾っている可能性があります。')
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

def filtered_sentence(sentence, nlp):
    doc = nlp(sentence)
    # フィラー、"INTJ"、 "reparandum" 依存関係のトークンを除外
    filtered_tokens = [token.text for token in doc if
                       token.tag_ != "感動詞-フィラー" and token.pos_ != "INTJ" and token.dep_ != "reparandum"]
    result = ''.join(filtered_tokens)
    # 読点を削除
    result = result.replace("、", "")
    result = result.replace("。", "")
    return result

if __name__ == '__main__':
    # JapaneseFeatureExtractor インスタンスを作成
    feature_extractor = JapaneseFeatureExtractor()
    nlp = feature_extractor.nlp_speech #話し言葉用
    #nlp = feature_extractor.nlp #書き言葉用
    # 例の使用法
    sentence = "私はああそうだニュースは今日が徳川家康の誕生日だと伝えたようだね困ったものだなあと思ったものだよ私は彼が鹿だという話を彼から聞いた"
    features = analyze_sentence(sentence, nlp)
    print(features)

    #　例文解析
    text="これは例文です。"
    ex_features = analyze_sentence(text, nlp)  # sentenceを解析
    ex_dic = {'ID': 'example', 'Visit': 0, 'Part': 0}
    #ex_dic1 = countMorpheme(text)
    #ex_dic2 = countEntity(text)
    #ex_dic3 = countSyntaxFeatures(text)
    print("ex_features:", ex_features)
    ex_dic.update(**ex_features)
    ex_df = pd.DataFrame(data=ex_dic, index=['val',])
    print(ex_df)

    # UNDERPIN解析
    PATH = '/Users/psychetmdu/Desktop/RecordVoice'
    #　インスタンス生成は、pythonProject以下で行うようにすること。注意が必要。
    os.chdir(PATH)
    dataframe_answer = fileSortForAnalaysisNestedDependency(PATH, ex_df, feature_extractor,nlp)
    print('dataframe_answer:', dataframe_answer)
    dataframe_answer.to_csv('Dataframe_analyzeNestedDependency_R6Jan26_RecordVoice.csv')
    print(checkFileAnalyzedOrNot(PATH, dataframe_answer))
    print("解析から外れたファイル数:", len(checkFileAnalyzedOrNot(PATH, dataframe_answer)))
