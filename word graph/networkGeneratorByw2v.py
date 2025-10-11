import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.font_manager as fm
import japanize_matplotlib


def train_tfidf_model(path):
    documents = []
    for filename in os.listdir(path):
        if filename.endswith('.tok'):
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
                # ファイルごとにドキュメントを作成
                document = []
                for line in file:
                    words = re.split('\t|\s', line.strip())  # タブとスペースで分割
                    document.extend(words)
                documents.append(' '.join(document))
    print("tokファイル全体のdocuments:", documents)
    # TF-IDFベクトルライザの初期化とトレーニング
    vectorizer = TfidfVectorizer()
    vectorizer.fit(documents)
    # モデルの保存
    with open('tfidf_model.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)
    print("TF-IDFモデルがトレーニングされ、'tfidf_model.pkl'として保存されました。")
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
    print("word_scores:",word_scores)
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

    important_word_list=list(set(important_word_list))
    print("important_word_list:", important_word_list)
    return important_word_list


# 類似度が閾値以上の単語リストを作成し、グラフに追加する関数
def extend_graph(G, word_list, graph_word_list, processed_words, sth, N, ih):
    new_graph_word_list = set()

    for wi in graph_word_list:
        if wi in processed_words:
            continue  # 既に処理された単語はスキップする

        over_sth_word_list = []

        for wj in word_list:
            if wi != wj:
                similarity = wv.similarity(wi, wj) if wi in wv.key_to_index and wj in wv.key_to_index else ih
                if similarity >= sth:
                    over_sth_word_list.append(wj)

        if len(over_sth_word_list) >= N:
            G.add_node(wi)
            for wh in over_sth_word_list[:N]:
                G.add_edge(wi, wh)
            new_graph_word_list.update(over_sth_word_list[:N])
        print("wi, over_sth_word_list:", wi, over_sth_word_list)
        processed_words.add(wi)
    return new_graph_word_list, processed_words

"""
2024/01/19　w2Vのベクトルからグラフを作成して特徴量を算出
・計算量が多くなるため、どのワードをノードとして取り出すかが重要。
2024/01/26　pip install japanize-matplotlib　グラフの日本語表記
"""

if __name__ == '__main__':
    # Tf-idfモデル作成
    path="/Users/hironobu/Desktop/w2vGraphNetwork/tryagaindata"
    tf_idf_vectorizer = train_tfidf_model(path)

    # w2v model作成
    model_path = '/Users/hironobu/Desktop/w2vGraphNetwork/entity_vector/entity_vector.model.txt'
    wv = KeyedVectors.load_word2vec_format(model_path, binary=False)

    # 類似度の閾値と上位N個を取り出すための数
    sth = 0.3  # 例えば 0.5
    N = 10  # 上位N個
    ih = 0 # word2Vecモデルに含まれていない単語についての類似度の代替値

    # テキストデータの読み込み（例）
    word_list = ["沖縄","私","パラダイス","洗浄","遊び","リレー","カレーライス","戦争","ドッキリ","海","海水浴","ダンス",
                 "剣道","精神","ラッキー","悲しい","困る","落書き","シンデレラ", "話",
                 "すごい", "お母さん", "親", "いない","シンデレラ","まま","母","育てる","義理","姉","愛す"]# あなたのテキストデータ

    # important_word_listの作成
    important_word_list= generate_important_word_list(model=tf_idf_vectorizer, word_list=word_list)
    print("important_word_list:", important_word_list)

    # ネットワークグラフの作成
    G = nx.Graph()

    # キーワードは最初にグラフノードとして設定
    for wi in important_word_list:
        G.add_node(wi)

    # word_list, graph_word_listを用意
    graph_word_list = important_word_list  # 初期値はimportant_word_list
    #word_list = list(set(word_list))

    # メインのwhileループ
    processed_words = set()
    iteration_count = 0
    max_iterations = 10  # 最大繰り返し回数
    while True:
        new_graph_word_list = extend_graph(G, word_list, graph_word_list, processed_words, sth, N, ih)
        if len(new_graph_word_list) < N or iteration_count >= max_iterations:
            break  # 新たなワードリストの要素数がN未満または最大繰り返し回数に達したら終了
        graph_word_list = new_graph_word_list
        iteration_count += 1

    # 特徴量の計算
    if len(G.nodes) == 0:
        print("グラフにノードが存在しないため、特徴量を計算できません。")
        features = {
            "number_of_nodes": 0,
            "words_nodes_ratio": 0,
            "cluster_coefficient": np.nan,
            "average_closeness_centrality": np.nan,
            #"degree_centrality": np.nan,
            #"betweenness_centrality": np.nan,
            "average_distance": np.nan,
            "diameter": np.nan,
            "density": np.nan,
        }

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
            #"degree_centrality": degree_centrality,
            #"betweenness_centrality": betweenness_centrality,
            "average_distance": average_distance,
            "diameter": diameter,
            "density": density,
        }

    # 特徴量を出力
    print(features)

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
