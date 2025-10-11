from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.font_manager as fm
import japanize_matplotlib
import os
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
import os


def load_tfidf_model(path, part):
    """
    指定されたパートに基づいて保存されたTF-IDFモデルを読み込み、ベクトルライザーを返す。

    Args:
    - path: モデルが保存されたディレクトリのパス
    - part: モデルが関連付けられたパート

    Returns:
    - vectorizer: 読み込まれたTF-IDFベクトルライザー
    """
    model_filename = os.path.join(path, f'tfidf_model_part{part}.pkl')

    try:
        with open(model_filename, 'rb') as file:
            vectorizer = pickle.load(file)
        print(f"TF-IDFモデルが'{model_filename}'から読み込まれました。")
        return vectorizer
    except Exception as e:
        print(f"Error during loading the model from '{model_filename}': {e}")
        return None

def train_tfidf_model(path, part):
    # 正規表現パターンの調整（Kana.tokファイルも含む）
    pattern = r'U[A-Z]{2}[0-9]{3}\.[1-5]\.' + str(part) + r'(\.tok|Kana\.tok)'

    if not os.path.isdir(path):
        print(f"Error: '{path}' is not a valid directory.")
        return None

    documents = []
    for filename in os.listdir(path):
        if re.match(pattern, filename):
            try:
                with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
                    document = []
                    for line in file:
                        words = re.split('\t|\s', line.strip())
                        document.extend(words)
                    documents.append(' '.join(document))
            except Exception as e:
                print(f"Error reading file {filename}: {e}")

    if not documents:
        print("No documents found for the specified part. Exiting function.")
        return None

    print(f"Documents for part {part}:", documents)

    try:
        vectorizer = TfidfVectorizer()
        vectorizer.fit(documents)
        model_filename = f'tfidf_model_part{part}.pkl'
        with open(model_filename, 'wb') as file:
            pickle.dump(vectorizer, file)
        print(f"TF-IDFモデルがトレーニングされ、'{model_filename}'として保存されました。")
    except Exception as e:
        print(f"Error during training or saving the model: {e}")
        return None

    return vectorizer


# word_listから重要な単語リストを生成する関数
def generate_important_word_list(model, word_list):
    # word_listの単語をスペース区切りの文書として扱う
    document = " ".join(word_list)
    # トレーニング済みモデルを使用してTF-IDF変換
    tfidf_matrix = model.transform([document])
    tfidf_scores = tfidf_matrix.toarray().flatten()
    # 単語とそのスコアを辞書に格納
    feature_names = model.get_feature_names_out()
    word_scores = {word: tfidf_scores[idx] for idx, word in enumerate(feature_names) if word in word_list}
    print("word_scores:", word_scores)
    # スコアに基づいて単語をソート
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    # 上位10%の単語を選出
    top_10_percent = int(len(sorted_words) * 0.1)
    important_word_list = [word for word, score in sorted_words[:top_10_percent]]
    print("important_word_list:", important_word_list)
    # TF-IDFモデルにない単語を追加
    for word in word_list:
        if word not in feature_names:
            important_word_list.append(word)

    important_word_list = list(set(important_word_list))
    print("important_word_list:", important_word_list)
    return important_word_list


# 類似度が閾値以上の単語リストを作成し、グラフに追加する関数
def extend_graph(wv, G, word_list, graph_word_list, processed_words, similarity_cache, sth, N, ih):
    new_graph_word_list = set()

    for wi in graph_word_list:
        if wi in processed_words:
            continue  # 既に処理された単語はスキップ

        over_sth_word_list = []

        for wj in word_list:
            if wi != wj:
                # 類似度のキャッシュをチェック
                if (wi, wj) in similarity_cache:
                    similarity = similarity_cache[(wi, wj)]
                else:
                    try:
                        similarity = wv.similarity(wi, wj) if wi in wv.key_to_index and wj in wv.key_to_index else ih
                        similarity_cache[(wi, wj)] = similarity  # 類似度をキャッシュに保存
                    except KeyError:
                        similarity = ih  # エラーが発生した場合のデフォルト類似度

                if similarity >= sth:
                    over_sth_word_list.append(wj)

        if len(over_sth_word_list) >= N:
            G.add_node(wi)
            for wh in over_sth_word_list[:N]:
                G.add_edge(wi, wh)
            new_graph_word_list.update(over_sth_word_list[:N])

        print("wi, over_sth_word_list:", wi, over_sth_word_list)
        processed_words.add(wi)

    return new_graph_word_list, processed_words, similarity_cache


def analyzeNetwork(tf_idf_vectorizer, word_list, sth, N, ih):
    """
    ネットワーク解析を行い、特徴量を計算する関数。

    Args:
    - tf_idf_vectorizer: TF-IDFベクトルライザー（モデル）
    - word_list: 解析対象の単語リスト
    - sth: 特定の閾値またはパラメータ
    - N: ネットワーク拡張のためのノード数の閾値
    - ih: 重要な単語を識別するための閾値またはパラメータ

    Returns:
    - features: 計算された特徴量の辞書
    """
    try:
        # important_word_listの作成
        important_word_list = generate_important_word_list(model=tf_idf_vectorizer, word_list=word_list)
        print("important_word_list:", important_word_list)

        # ネットワークグラフの作成
        G = nx.Graph()

        # キーワードは最初にグラフノードとして設定
        for wi in important_word_list:
            G.add_node(wi)

        # word_list, graph_word_listを用意
        graph_word_list = important_word_list  # 初期値はimportant_word_list

        # メインのwhileループ
        processed_words = set()  # 処理済みの単語のリスト
        similarity_cache = {}  # 類似度計算のキャッシュ
        iteration_count = 0
        while True:
            new_graph_word_list, processed_words, similarity_cache = extend_graph(G, word_list, graph_word_list,
                                                                                  processed_words, similarity_cache,
                                                                                  sth, N, ih)
            if len(new_graph_word_list) < N or iteration_count >= MAX_ITERATIONS:
                break  # 新たなワードリストの要素数がN未満または最大繰り返し回数に達したら終了
            graph_word_list = new_graph_word_list
            iteration_count += 1

        # 特徴量の計算
        if len(G.nodes) == 0:
            return {
                "number_of_nodes": 0,
                "words_nodes_ratio": 0,
                "cluster_coefficient": np.nan,
                "average_closeness_centrality": np.nan,
                "average_distance": np.nan,
                "diameter": np.nan,
                "density": np.nan,
            }

        # 以下、特徴量計算の処理...
        else:
            cluster_coefficient = nx.average_clustering(G)
            closeness_centrality = nx.closeness_centrality(G)
            average_closeness_centrality = np.mean(list(closeness_centrality.values()))
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            density = nx.density(G)

            try:
                average_distance = nx.average_shortest_path_length(G)
            except nx.NetworkXError:
                try:
                    largest_cc = max(nx.connected_components(G), key=len)
                    largest_subgraph = G.subgraph(largest_cc).copy()
                    average_distance = nx.average_shortest_path_length(largest_subgraph)
                except (ValueError, nx.NetworkXError):
                    average_distance = np.nan

            try:
                diameter = nx.diameter(G)
            except nx.NetworkXError:
                try:
                    largest_cc = max(nx.connected_components(G), key=len)
                    largest_subgraph = G.subgraph(largest_cc).copy()
                    diameter = nx.diameter(largest_subgraph)
                except (ValueError, nx.NetworkXError):
                    diameter = np.nan

            features = {
                "number_of_nodes": len(G.nodes),
                "words_nodes_ratio": len(G.nodes) / len(word_list) if word_list else 0,
                "cluster_coefficient": cluster_coefficient,
                "average_closeness_centrality": average_closeness_centrality,
                # "degree_centrality": degree_centrality,
                # "betweenness_centrality": betweenness_centrality,
                "average_distance": average_distance,
                "diameter": diameter,
                "density": density,
            }
    except Exception as e:
        print(f"An error occurred in analyzeNetwork: {e}")
        return {
            "number_of_nodes": 0,
            "words_nodes_ratio": 0,
            "cluster_coefficient": np.nan,
            "average_closeness_centrality": np.nan,
            "average_distance": np.nan,
            "diameter": np.nan,
            "density": np.nan,
        }

    del G
    return features

def load_token_list(path):
    token_list=[]
    with open(path, 'r') as token_file:
        for line in token_file:
            line_list = re.split('\t', line)
            #print('line_list',line_list)
            token_list = token_list + line_list
            #print('token_list', token_list)
    #'\n'を含む場合、それを除外する
    for i in range(len(token_list)):
        token_list[i] = token_list[i].replace("\n",'')
    return token_list


def wvGraphAnalyzer(data_path, part, tf_idf_vectorizer, sth, N, ih):
    os.chdir(data_path)
    dataframe = pd.DataFrame()
    pattern = r'U[A-Z]{2}[0-9]{3}\.[1-5]\.' + str(part) + r'(\.tok|Kana\.tok)'

    for filename in os.listdir(data_path):
        if re.match(pattern, filename):
            file_path = os.path.join(data_path, filename)
            try:
                token_list = load_token_list(file_path)
                features = analyzeNetwork(tf_idf_vectorizer, token_list, sth, N, ih, MAX_ITERATIONS)
                catch_id = re.findall(r'U[A-Z]{2}[0-9]{3}.[1-5].[1-3]', filename)[0]
                subid = catch_id[0:6]
                subvisit = catch_id[7]
                subpart = catch_id[9]

                dic = {'ID': subid, 'Visit': subvisit, 'Part': subpart}
                dic.update(**features)
                df = pd.DataFrame(data=dic, index=['val', ])
                dataframe = pd.concat([df, dataframe], axis=0)
            except Exception as e:
                print(f'Error processing file {filename}: {e}')
    return dataframe

def checkFileAnalyzedOrNot(directory_path, dataframe):
    # dataframeからID, Visit, Partを読み取り、file_name = "ID-Visit_Part"を作成
    analyzed_file_list = [f"{row['ID']}-{row['Visit']}_{row['Part']}" for index, row in dataframe.iterrows()]
    # 正規表現パターン
    pattern = r'U[A-Z]{2}[0-9]{3}.[1-5].[1-3]'
    # '.txt' 形式のファイルのパスを格納するリスト
    all_txt_file = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.tok')]
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
"""
2024/01/19　w2Vのベクトルからグラフを作成して特徴量を算出
・計算量が多くなるため、どのワードをノードとして取り出すかが重要。
2024/01/26　pip install japanize-matplotlib　グラフの日本語表記
"""

if __name__ == '__main__':
    # Tf-idfモデル作成
    part = 1  # partを指定
    data_path = '/Users/psychetmdu/Desktop/RecordVoice_tok'
    tf_idf_vectorizer = train_tfidf_model(data_path, part=part)
    tf_idf_vectorizer = load_tfidf_model(data_path,part)

    # w2v model作成
    model_path = '/Users/hironobu/Desktop/w2vGraphNetwork/entity_vector/entity_vector.model.txt'
    wv = KeyedVectors.load_word2vec_format(model_path, binary=False)

    # 類似度の閾値と上位N個を取り出すための数
    sth = 0.3  # 例えば 0.5
    N = 10  # 上位N個
    ih = 0  # word2Vecモデルに含まれていない単語についての類似度の代替値

    MAX_ITERATIONS = 10  # 最大繰り返し回数の定数

    # ここまではグローバル変数

    #　例文解析
    ex_word_list = ["沖縄", "私", "パラダイス", "洗浄", "遊び", "リレー", "カレーライス", "戦争", "ドッキリ", "海", "海水浴", "ダンス",
                 "剣道", "精神", "ラッキー", "悲しい", "困る", "落書き", "シンデレラ", "話",
                 "すごい", "お母さん", "親", "いない", "シンデレラ", "まま", "母", "育てる", "義理", "姉", "愛す"]  # あなたのテキストデータ
    ex_features = analyzeNetwork(tf_idf_vectorizer, ex_word_list, sth, N, ih)  # word_listを解析
    ex_dic = {'ID': 'example', 'Visit': 0, 'Part': 0}
    print("ex_features:", ex_features)
    ex_dic.update(**ex_features)
    ex_df = pd.DataFrame(data=ex_dic, index=['val',])
    print(ex_df)

    # ここから解析・特徴量計算・データフレーム化
    DataFrame = wvGraphAnalyzer(data_path, part, tf_idf_vectorizer, sth, N, ih)
    DataFrame.to_csv('wvGraphAnalyzer_R6Jan27.csv')
    print(checkFileAnalyzedOrNot(data_path, DataFrame))
    print("解析から外れたファイル数:", len(checkFileAnalyzedOrNot(data_path, DataFrame)))

'''
    # 特徴量
    Eigenvector Centrality

    # 日本語フォントの設定
    # fm.get_font_config()
    #import matplotlib
    import japanize_matplotlib
    # plt.rcParams['font.family'] = 'Hiragino Maru Gothic Pro'
    #matplotlib.rc('font', family='Helvetica')

    # グラフの可視化
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', font_family='Hiragino Maru Gothic Pro')
    plt.title("Network Graph")
    plt.show()


    # グラフオブジェクトを削除
    #del G

    # matplotlibの現在の図（figure）をクローズ
    #plt.close('all')
'''
