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

2024/04/26　実行

2024/12/06　発話長が長くなりすぎない場合はこちらのファイルで十分対応可能
"""

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


def fileSortForAnalaysisGINZA(path, dataframe):
    # JapaneseFeatureExtractor インスタンスを作成
    feature_extractor = JapaneseFeatureExtractor()

    # ディレクトリを移動
    os.chdir(path)
    '''
    id = 'U*'
    visit = '*'
    part = '*'
    format = '*.txt'
    name = id + '-' + visit + '_' + part + format
    file = glob.glob(path + '/**/' + name, recursive=True)
    '''
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
                    #print('PATIENT_lines:',PATIENT_lines)
                    patient_part_lines = ''.join(PATIENT_lines)
                    print('patient_part_lines:', patient_part_lines)
                    sentence = str(patient_part_lines)  # セルの値を文字列に変換してsentenceとして扱う
                    features = feature_extractor.extract_features(sentence)  # sentenceを解析
                    print("features:", features)
                    dic = {'ID': subid, 'Visit': subvisit, 'Part':subpart}
                    print("dic:", dic)
                    #dic1 = countMorpheme(patient_part_lines)
                    #dic2 = countEntity(patient_part_lines)
                    #dic3 = countSyntaxFeatures(patient_part_lines)
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

if __name__ == '__main__':
    # JapaneseFeatureExtractor インスタンスを作成
    feature_extractor = JapaneseFeatureExtractor()

    #　例文解析
    text="これは例文です。"
    ex_features = feature_extractor.extract_features(text)  # sentenceを解析
    ex_dic = {'ID': 'example', 'Visit': 0, 'Part': 0}
    #ex_dic1 = countMorpheme(text)
    #ex_dic2 = countEntity(text)
    #ex_dic3 = countSyntaxFeatures(text)
    print("ex_features:", ex_features)
    ex_dic.update(**ex_features)
    ex_df = pd.DataFrame(data=ex_dic, index=['val',])
    print(ex_df)

    # UNDERPIN解析
    PATH = '/Users/psychetmdu/NLP/RecordVoice_yg_R6Nov22'
    #　インスタンス生成は、pythonProject以下で行うようにすること。注意が必要。
    #os.chdir(PATH)
    dataframe_answer = fileSortForAnalaysisGINZA(PATH, ex_df)
    print('dataframe_answer:', dataframe_answer)
    dataframe_answer.to_csv('Dataframe_AnalysisGINZA_R7Aug16.csv')
    print(checkFileAnalyzedOrNot(PATH, dataframe_answer))
    print("解析から外れたファイル数:", len(checkFileAnalyzedOrNot(PATH, dataframe_answer)))