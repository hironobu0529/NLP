import glob
import os
import re
import statistics
from gensim.models import KeyedVectors
import pandas as pd

"""
2022/10/23　重複を省くように実装し直した。
2023/06/13　実装修正。path入力-> duplicates, similarity計算 -> dataframe
https://www.nogawanogawa.com/entry/gensim_intro
http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/
2024/04/25 実行。 RecordVoice_vf_tokに対して。
"""

model_path = '/Users/psychetmdu/Desktop/w2V/entity_vector/entity_vector.model.txt'
wv = KeyedVectors.load_word2vec_format(model_path, binary=False)

def load_token_list(path):
    token_list=[]
    with open(path, 'r') as token_file:
        for line in token_file:
            line_list = re.split('\t', line)
            #print('line_list',line_list)
            token_list = token_list + line_list
            #print('token_list', token_list)
    token_file.close()
    #'\n'を含む場合、それを除外する
    for i in range(len(token_list)):
        token_list[i] = token_list[i].replace("\n",'')
    return token_list


def wvMain(data_path):
    df = pd.DataFrame({'ID': [],
                       'Visit': [],
                       'Part': [],
                       'duplicates': [],
                       'duplicate_rate': [],
                       'w2Vsim_mean': [],
                       'w2Vsim_variance': [],
                       'w2Vsim_SD': [],
                       'w2Vsim_MAX': [],
                       'w2Vsim_MIN': [],
                       'w2Vsim_median': [],
                       'unique_num': []})

    os.chdir(data_path)
    model_path = '/Users/psychetmdu/Desktop/w2V/entity_vector/entity_vector.model.txt'
    wv = KeyedVectors.load_word2vec_format(model_path, binary=False)
    id = 'U*'
    visit = '*'
    part = '*'
    format = '*.tok'
    name = id + '-' + visit + '_' + part + format
    file_list= glob.glob(data_path + '/**/' + name, recursive=True)
    print('file_list:', file_list)

    for f in file_list:
        print(f)
        try:
            catch_id = re.findall(r'U[A-Z]{2}[0-9]{3}.[1-5].[1-3]', f)
            subid = catch_id[0][0:6]
            subvisit = catch_id[0][7]
            subpart = catch_id[0][9]
        except:
            subid=f
            subvisit=f
            subpart=f

        f=str(f)
        token_list = load_token_list(f)
        #print('token_list:', token_list)
        token_len = len(token_list)
        #　重複するものは省いてリスト化
        token_set = set(token_list)
        token_list_unique = list(set(token_list))
        token_list_unique = [i for i in token_list_unique if i != '']
        print('token_list_unique:', token_list_unique)
        token_unique_len = len(token_list_unique)

        #　重複の数を計算
        print('token_len:', token_len)
        print('token_unique_len:', token_unique_len)
        unique_num = token_unique_len
        dp = token_len - token_unique_len
        print('duplicates:', dp)
        duplicate_rate = dp / token_len
        print('duplicate_rate:', duplicate_rate)
        similarity_list=[]

        for i in range(token_unique_len):
            for j in range(token_unique_len):
                if (i == j):
                    continue
                try:
                    result = wv.similarity(token_list_unique[i],token_list_unique[j])
                    #print(token_list_unique[i], "*", token_list_unique[j])
                    #print("result:", result)
                    similarity_list.append(result)
                except:
                    continue
        #print("similarity_list:", similarity_list)
        df_Features = countFeatures(alist=similarity_list, duplicates=dp, duplicate_rate=duplicate_rate, unique_num=unique_num, subjectID=subid, visit=subvisit, part=subpart)
        df = pd.concat([df, df_Features])
    return df


def countFeatures(alist, duplicates, duplicate_rate, unique_num, subjectID, visit, part) :
    alist  = [float(item) for item in alist]
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


    dataframe = pd.DataFrame({
        'ID': subjectID,
        'Visit': visit,
        'Part': part,
        'duplicates': duplicates,
        'duplicate_rate': duplicate_rate,
        'w2Vsim_mean': mean,
        'w2Vsim_variance': variance,
        'w2Vsim_SD': SD,
        'w2Vsim_MAX': MAX,
        'w2Vsim_MIN': MIN,
        'w2Vsim_median': median,
        'unique_num': unique_num
    }, index=[0])

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

if __name__ == "__main__":
    data_path = '/Users/psychetmdu/Desktop/RecordVoice_yg_R6Nov29_tok'
    DataFrame = wvMain(data_path)
    DataFrame.to_csv('w2Vsimilarity_r&b_R6Nov29.csv')
    print(checkFileAnalyzedOrNot(data_path, DataFrame))
    print("解析から外れたファイル数:", len(checkFileAnalyzedOrNot(data_path,DataFrame)))