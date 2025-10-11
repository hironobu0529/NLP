import glob
import os
import traceback
import re
import codecs
"""
2023/05/21　実装終了
'○'と'◯'を'●'に書き換えるコード
RecordVoiceデータで実験済み
読み込んで書き換えて新規に保存という形をとるため、必ずバックアップを取ること！！！

2023/08/17　修正
“●医師”、”●医者”の２種類あり、後者が圧倒的少数派。
“●医師”で統一することとした。

2023/12/26　修正
再発注のデータがエンコーディングが変わっているため、下記のようにエンコーディング候補を併記して読み込み、最終的にはutf-8で保存するように変更
空白スペースと"()"を削除するように追加した。
"""

def readPatientPart(file_name):
    # 使用するエンコーディングの候補
    #encodings_to_try = ["utf-8", "shift-jis", "euc-jp", "utf-16"]  # お好みのエンコーディングを追加
    encodings_to_try = [
        "utf-8", "shift-jis", "euc-jp", "utf-16", "iso-8859-1", "windows-1252", "utf-8-sig", "macroman", "us-ascii",
        "big5", "euc-kr", "gb2312", "iso-2022-jp", "iso-2022-kr", "iso-8859-2", "iso-8859-5", "iso-8859-7",
        "iso-8859-9",
        "koi8-r", "utf-32", "utf-32-be", "utf-32-le", "utf-16-be", "utf-16-le", "utf-7", "utf-7-imap", "windows-1250",
        "windows-1251", "windows-1253", "windows-1254", "windows-1255", "windows-1256", "windows-1257", "windows-1258",
        "iso-8859-15", "iso-8859-16", "mac-cyrillic", "mac-romanian", "mac-turkish", "mac-greek", "mac-iceland",
        "mac-croatian", "mac-ce", "hp-roman8", "hp-greek8", "hp-hebrew", "hp-turkish8", "hp-thai8", "hp-southl1",
        "hp-southl2", "hp-czech", "hp-polish", "hp-hungarian", "hp-slovenian", "hp-bulgarian", "hp-ukraine",
        "hp-estonia",
        "hp-lithuania", "koi8-u", "koi8-t", "iso-8859-13", "iso-8859-14", "iso-8859-10", "iso-8859-11", "iso-8859-4",
        "iso-8859-6", "iso-8859-8", "iso-8859-3", "iso-8859-14", "tis-620", "hz-gb-2312", "euc-cn", "euc-tw",
        "mac-japanese"
    ]

    for encoding in encodings_to_try:
        try:
            with codecs.open(file_name, "r", encoding=encoding) as f:
                data_lines = f.read()
                print(f"Successfully read using encoding: {encoding}")
            break  # エンコーディングが正しく特定された場合はループを終了
        except UnicodeDecodeError:
            print(f"Failed to read using encoding: {encoding}")
            continue  # 次のエンコーディングを試す

    # 文字列置換
    new_data_lines = replaceToken(data_lines)

    # 同じファイル名で保存
    with codecs.open(file_name, "w", encoding="utf-8") as f:
        f.write(new_data_lines)

    return new_data_lines

def remove_brackets(text):
    # 括弧とその内部を正規表現で取り除く
    cleaned_text = re.sub(r'\([^)]*\)', '', text)
    return cleaned_text

def replaceToken(text):
    # ○が4種類あるので4段構えとする。
    # "医者"　→　"医師"で統一
    answer=text.replace('◯','●')
    answer1 = answer.replace('○','●')
    answer2 = answer1.replace('○','●')
    answer3 = answer2.replace('○','●').replace('〇', '●')
    last_answer = answer3.replace('●医者','●医師').replace(' ', '').replace('　', '')
    last_answer = remove_brackets(last_answer)
    return last_answer

"""
def readPatientPart(file_name):
    with open(file_name, encoding="utf-8") as f:
        data_lines = f.read()
        #print('data_lines:', data_lines)
    # 文字列置換
    new_data_lines = replaceToken(data_lines)
    #print("new_data_lines:", new_data_lines)
    # 同じファイル名で保存
    with open(file_name, mode="w", encoding="utf-8") as f:
        f.write(new_data_lines)
    return new_data_lines
"""

def sortFile(path):
    os.chdir(path)
    id = 'U***'
    visit = '*'
    part = '*'
    format = '*.txt'
    name = id + '-' + visit + '_' + part + format
    file = glob.glob(path + '/**/' + name, recursive=True)
    print('file:',file)

    for i in range(len(file)):
        target1 = '.txt'
        target2 = 'Kana.txt'
        if ((target1 in file[i]) == True) or ((target2 in file[i]) == True):
            catch_id = re.findall('[U][A-Z][A-Z][0-9][0-9][0-9][-][1-5][_][1-3]', file[i])
            #subid = catch_id[0][0:6]
            #subvisit = catch_id[0][7]
            #subpart = catch_id[0][9]
            print('file:', file[i])
            try:
                filename=str(file[i])
                text = readPatientPart(filename)
                print('text:',text)
            except:
                print('Error①です。')
                print('error:', file[i])
                traceback.print_exc()
        else:
            print('Error②を拾っている可能性があります。')



if __name__ == '__main__':
    test_text = "○丸がたくさんあって困る。正解は◯。◯。◯。○。◯。"
    answer = replaceToken(test_text)
    print("answer:", answer)
    print("=============ここまで実験内容です。これ以下が本番==================")

    this_path = '/Users/psychetmdu/Desktop/RecordVoice_yg_R6Nov29'
    os.chdir(this_path)
    # 実行部分
    sortFile(this_path)
