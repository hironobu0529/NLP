import glob
import re
import os
import MeCab
from gensim.models.ldamodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from collections import defaultdict

import pandas as pd
from statistics import mean, median,variance,stdev
import traceback

# MeCabオブジェクトの生成
neologd_path = "/usr/local/lib/mecab/dic/mecab-ipadic-neologd" # インストール時に表示されたパス
mt = MeCab.Tagger("-d " + neologd_path)
#mt.parse('')

'''
2022/10/24
LDA for All　を基に、内容のみ抽出するファイルを作成する。
2hr くらいでできた。
'◯患者'に注意が必要。'◯'を認識しない。
2024/01/26　確認
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

def printOutContentWords(file_path, text_list):
    list_len = len(text_list)
    if list_len != 0:
        ftok_path = file_path.replace('.txt', '.tok')
        print('ftok_path:', ftok_path)
        try:
            f_tok = open(ftok_path, 'x')
            text_len = len(text_list)
            for i in range(text_len):
                if i%10 !=0:
                    f_tok.write(text_list[i] + "\t")
                else:
                    f_tok.write(text_list[i] + "\n")
            f_tok.close()
        except:
            print('error:',file_path)
            traceback.print_exc()

def sortFile(path):
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
    print('file数:',len(file))

    for i in range(len(file)):
        target1 = '.txt'
        target2 = 'Kana.txt'
        if ((target1 in file[i]) == True) or ((target2 in file[i]) == True):
            catch_id = re.findall(r'U[A-Z]{2}[0-9]{3}.[1-5].[1-3]', file[i])
            #subid = catch_id[0][0:6]
            #subvisit = catch_id[0][7]
            #subpart = catch_id[0][9]
            print('file:',file[i])
            try:
                #print("ここがおかしいと予想, str(file[i]):", str(file[i]))
                PATIENT_lines = readPatientPart(str(file[i]))
                #print('PATIENT_lines:',PATIENT_lines)
                text_list = extractContentWords(PATIENT_lines)
                #print("text_list:", text_list)
                printOutContentWords(file[i], text_list)
            except:
                print('Error①です。')
                print('error:', file[i])
                traceback.print_exc()
        else:
            print('Error②を拾っている可能性があります。')

if __name__ == "__main__":
    this_path = '/Users/psychetmdu/Desktop/RecordVoice_yg_R6Nov22'
    os.chdir(this_path)
    # MeCabオブジェクトの生成
    neologd_path = "/usr/local/lib/mecab/dic/mecab-ipadic-neologd"  # インストール時に表示されたパス
    mt = MeCab.Tagger("-d " + neologd_path)
    # 実行部分
    sortFile(this_path)

    file = [os.path.join(this_path, f) for f in os.listdir(this_path) if f.endswith('.txt')]
    tok_file = [os.path.join(this_path, f) for f in os.listdir(this_path) if f.endswith('.tok')]
    print("tokのファイル数:", len(tok_file))

    # file と tok_file から拡張子を除いた基本名を抽出
    base_names_file = set(os.path.splitext(os.path.basename(f))[0] for f in file)
    base_names_tok_file = set(os.path.splitext(os.path.basename(f))[0] for f in tok_file)

    # file にあって tok_file にない基本名を見つける
    unique_to_file = base_names_file - base_names_tok_file

    # 結果を表示
    print("fileにあってtok_fileにないID:", unique_to_file)



