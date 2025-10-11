import ginza
import spacy
import traceback

class JapaneseFeatureExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load('ja_ginza_electra') #基本は書き言葉としておく
            self.nlp_speech = spacy.load('ja_cejc_gsd_bert_v2')# CEJC。話し言葉用。
            self.nlp_speech_parse = spacy.load('ja_cejc_parse_bert_v2') # spaCy dependency analysis model trained on CEJC
            self.nlp_speech_CEJCminus = spacy.load('ja_cejc_dropped_morph_parser-3.4.3')
            #self.nlp = spacy.load('ja_cejc_gsd_bert_v2')  # 話し言葉用
        except Exception as e:
            print("Failed to load one or more Ginza models:", str(e))
            raise

    def validate_input(self, text):
        if not text or not isinstance(text, str):
            raise ValueError("Invalid input: text must be a non-empty string")

    def extract_features(self, text):
        try:
            self.validate_input(text)
            # 既存の特徴量抽出関数を呼び出す
            morpheme_features = self.count_morpheme(text)
            entity_features = self.count_entity(text)
            syntax_features = self.count_syntax_features(text)
            dependency_features = self.calculate_dependency_features(text)
            avg_sentence_length, avg_word_length, num_sentences = self.calculate_text_lengths(text)
            redundancy = self.calculate_redundancy_with_ginza(text)

            # 特徴量の結合
            features = {
                **morpheme_features,
                **entity_features,
                **syntax_features,
                **dependency_features,
                'avg_sentence_length': avg_sentence_length,
                'avg_word_length': avg_word_length,
                'num_sentences': num_sentences,
                'redundancy': redundancy,
            }
            return features
        except Exception as e:
            print("Error occurred during extract_features in feature extraction:", str(e))
            traceback.print_exc()
            return {}

    def calculate_text_lengths(self, text):
        try:
            self.validate_input(text)
            # 文長については書き言葉用を用いる。
            doc = self.nlp(text)
            sentence_lengths = [len(sent) for sent in doc.sents]
            word_lengths = [len(token) for token in doc]
            avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
            avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
            #sentence_length_distribution = {length: sentence_lengths.count(length) for length in set(sentence_lengths)}
            # 文章の数もここでカウント
            num_sentences = len(list(doc.sents))
            return avg_sentence_length, avg_word_length, num_sentences
        except Exception as e:
            print("Error occurred during calculate text lengths in feature extraction:", str(e))
            traceback.print_exc()
            return {}

    # ここに以前の関数をメソッドとして組み込む
    def count_morpheme(self, text):
        try:
            self.validate_input(text)
            # countMorpheme関数の内容をここにコピー
            total_morphemes = 0
            num_noun = 0
            num_propn = 0
            num_verb = 0
            num_adj = 0
            num_adv = 0
            num_intj = 0
            num_pron = 0
            num_num = 0
            num_aux = 0
            num_cconj = 0
            num_sconj = 0
            num_det = 0
            num_adp = 0
            num_part = 0
            num_punct = 0
            num_sym = 0
            num_intj_general = 0
            num_intj_filler = 0
            num_kakujoshi = 0
            num_fukujoshi = 0
            num_kakarijoshi = 0
            num_setuzokujoshi = 0
            num_shuujoshi = 0
            num_rentaishi = 0
            num_settouji = 0
            num_setsubiji = 0

            doc = self.nlp_speech(text)
            for sent in doc.sents:
                for token in sent:
                    # print(token)
                    total_morphemes += 1
                    if token.pos_ == "NOUN":
                        num_noun += 1
                    if token.pos_ == "PROPN":
                        num_propn += 1
                    if token.pos_ == "VERB":
                        num_verb += 1
                    if token.pos_ == "ADJ":
                        num_adj += 1
                    if token.pos_ == "ADV":
                        num_adv += 1
                    if token.pos_ == "INTJ":
                        num_intj += 1
                    if token.pos_ == "PRON":
                        num_pron += 1
                    if token.pos_ == "NUM":
                        num_num += 1
                    if token.pos_ == "AUX":
                        num_aux += 1
                    if token.pos_ == "CCONJ":
                        num_cconj += 1
                    if token.pos_ == "SCONJ":
                        num_sconj += 1
                    if token.pos_ == "DET":
                        num_det += 1
                    if token.pos_ == "ADP":
                        num_adp += 1
                    if token.pos_ == "PART":
                        num_part += 1
                    if token.pos_ == "PUNCT":
                        num_punct += 1
                    if token.pos_ == "SYM":
                        num_sym += 1
                    if token.tag_ == "感動詞-一般":
                        num_intj_general += 1
                    elif token.tag_ == "感動詞-フィラー":
                        num_intj_filler += 1
                    elif token.tag_ == "助詞-格助詞":
                        num_kakujoshi += 1
                    elif token.tag_ == "助詞-副助詞":
                        num_fukujoshi += 1
                    elif token.tag_ == "助詞-係助詞":
                        num_kakarijoshi += 1
                    elif token.tag_ == "助詞-接続助詞":
                        num_setuzokujoshi += 1
                    elif token.tag_ == "助詞-終助詞":
                        num_shuujoshi += 1
                    if '連体詞' in token.tag_:
                        num_rentaishi += 1
                    if '接頭辞' in token.tag_:
                        num_settouji += 1
                    if '接尾辞' in token.tag_:
                        num_setsubiji += 1
            return {"num_morhpheme": total_morphemes,
                    "num_NOUN": num_noun,
                    "num_PROPN": num_propn,
                    "num_VERB": num_verb,
                    "num_ADJ": num_adj,
                    "num_ADV": num_adv,
                    "num_INTJ": num_intj,
                    "num_PRON": num_pron,
                    "num_NUM": num_num,
                    "num_AUX": num_aux,
                    "num_CCONJ": num_cconj,
                    "num_SCONJ": num_sconj,
                    "num_DET": num_det,
                    "num_ADP": num_adp,
                    "num_PART": num_part,
                    "num_PUNCT": num_punct,
                    "num_SYM": num_sym,
                    "num_intj_general": num_intj_general,
                    "num_intj_filler": num_intj_filler,
                    "num_kakujoshi": num_kakujoshi,
                    "num_fukujoshi": num_fukujoshi,
                    "num_kakarijoshi": num_kakarijoshi,
                    "num_setuzokujoshi": num_setuzokujoshi,
                    "num_shuujoshi": num_shuujoshi,
                    "num_rentaishi": num_rentaishi,
                    "num_settouji": num_settouji,
                    "num_setsubiji": num_setsubiji
                    }
        except Exception as e:
            print("Error occurred during count morpheme in feature extraction:", str(e))
            traceback.print_exc()
            return {}

    def count_entity(self, text):
        try:
            self.validate_input(text)
            # countEntity関数の内容をここにコピー
            num_person = 0
            num_god = 0
            num_place = 0
            num_time = 0
            doc = self.nlp(text)#固有表現抽出は書き言葉用
            for ent in doc.ents:
                #print(ent.text, ent.label_)
                # print(ent.label_)
                if ent.label_ == "Person":
                    num_person += 1
                    #print("人名の固有名詞:",ent)
                if ent.label_ == "God":
                    num_god += 1
                    #print("神の固有名詞:",ent)
                if ent.label_ == "Country":
                    num_place += 1
                    #print("場所の固有名詞:",ent)
                if ent.label_ == "Region":
                    num_place += 1
                    #print("場所の固有名詞:",ent)
                if ent.label_ == "Region_Other":
                    num_place += 1
                    #print("場所の固有名詞:",ent)
                if ent.label_ == "Province":
                    num_place += 1
                    #print("場所の固有名詞:",ent)
                if ent.label_ == "City":
                    num_place += 1
                    #print("場所の固有名詞:",ent)
                if ent.label_ == "Location":
                    num_place += 1
                    #print("場所の固有名詞:",ent)
                if ent.label_ == "Location_Other":
                    num_place += 1
                    #print("場所の固有名詞:",ent)
                if ent.label_ == "Timex":
                    num_time += 1
                    #print("時間の固有名詞:",ent)
                if ent.label_ == "Timex_Other":
                    num_time += 1
                    #print("時間の固有名詞:",ent)
                if ent.label_ == "Timeex" :
                    num_time += 1
                    #print("時間の固有名詞:",ent)
                if ent.label_ == "Timeex_Other":
                    num_time += 1
                    #print("時間の固有名詞:",ent)
                if ent.label_ == "Time":
                    num_time += 1
                    #print("時間の固有名詞:",ent)
                if ent.label_ == "Date":
                    num_time += 1
                    #print("時間の固有名詞:",ent)
                if ent.label_ == "Era":
                    num_time += 1
                    #print("時間の固有名詞:",ent)
            return {"num_person": num_person,
                    "num_god": num_god,
                    "num_place": num_place,
                    "num_time": num_time}
        except Exception as e:
            print("Error occurred during count entity in feature extraction:", str(e))
            traceback.print_exc()
            return {}

    def count_syntax_features(self, text):
        try:
            self.validate_input(text)
            # countSyntaxFeatures関数の内容をここにコピー
            num_acl = 0
            num_advcl = 0
            num_advmod = 0
            num_amod = 0
            num_appos = 0
            num_aux = 0
            num_case = 0
            num_cc = 0
            num_ccomp = 0
            num_clf = 0
            num_compound = 0
            num_conj = 0
            num_cop = 0
            num_csubj = 0
            num_dep = 0
            num_det = 0
            num_discourse = 0
            num_dislocated = 0
            num_expl = 0
            num_fixed = 0
            num_flat = 0
            num_goeswith = 0
            num_iobj = 0
            num_list = 0
            num_mark = 0
            num_nmod = 0
            num_nsubj = 0
            num_nummod = 0
            num_obj = 0
            num_obl = 0
            num_orphan = 0
            num_parataxis = 0
            num_punct = 0
            num_reparandum = 0
            num_root = 0
            num_vocative = 0
            num_xcomp = 0

            doc = self.nlp_speech_parse(text)
            for sent in doc.sents:
                for token in sent:
                    if token.dep_ == "acl":
                        num_acl += 1
                    if token.dep_ == "advcl":
                        num_advcl += 1
                    if token.dep_ == "advmod":
                        num_advmod += 1
                    if token.dep_ == "amod":
                        num_amod += 1
                    if token.dep_ == "appos":
                        num_appos += 1
                    if token.dep_ == "aux":
                        num_aux += 1
                    if token.dep_ == "case":
                        num_case += 1
                    if token.dep_ == "cc":
                        num_cc += 1
                    if token.dep_ == "ccomp":
                        num_ccomp += 1
                    if token.dep_ == "clf":
                        num_clf += 1
                    if token.dep_ == "compound":
                        num_compound += 1
                    if token.dep_ == "conj":
                        num_conj += 1
                    if token.dep_ == "cop":
                        num_cop += 1
                    if token.dep_ == "csubj":
                        num_csubj += 1
                    if token.dep_ == "dep":
                        num_dep += 1
                    if token.dep_ == "det":
                        num_det += 1
                    if token.dep_ == "discourse":
                        num_discourse += 1
                    if token.dep_ == "dislocated":
                        num_dislocated += 1
                    if token.dep_ == "expl":
                        num_expl += 1
                    if token.dep_ == "fixed":
                        num_fixed += 1
                    if token.dep_ == "flat":
                        num_flat += 1
                    if token.dep_ == "goeswith":
                        num_goeswith += 1
                    if token.dep_ == "iobj":
                        num_iobj += 1
                    if token.dep_ == "list":
                        num_list += 1
                    if token.dep_ == "mark":
                        num_mark += 1
                    if token.dep_ == "nmod":
                        num_nmod += 1
                    if token.dep_ == "nsubj":
                        num_nsubj += 1
                    if token.dep_ == "nummod":
                        num_nummod += 1
                    if token.dep_ == "obj":
                        num_obj += 1
                    if token.dep_ == "obl":
                        num_obl += 1
                    if token.dep_ == "orphan":
                        num_orphan += 1
                    if token.dep_ == "parataxis":
                        num_parataxis += 1
                    if token.dep_ == "punct":
                        num_punct += 1
                    if token.dep_ == "reparandum":
                        num_reparandum += 1
                    if token.dep_ == "ROOT":
                        num_root += 1
                    if token.dep_ == "vocative":
                        num_vocative += 1
                    if token.dep_ == "xcomp":
                        num_xcomp += 1

            return {
                "acl": num_acl,
                "advcl": num_advcl,
                "advmod": num_advmod,
                "amod": num_amod,
                "appos": num_appos,
                "aux": num_aux,
                "case": num_case,
                "cc": num_cc,
                "ccomp": num_ccomp,
                "clf": num_clf,
                "compound": num_compound,
                "conj": num_conj,
                "cop": num_cop,
                "csubj": num_csubj,
                "dep": num_dep,
                "det": num_det,
                "discourse": num_discourse,
                "dislocated": num_dislocated,
                "expl": num_expl,
                "fixed": num_fixed,
                "flat": num_flat,
                "goeswith": num_goeswith,
                "iobj": num_iobj,
                "list": num_list,
                "mark": num_mark,
                "nmod": num_nmod,
                "nsubj": num_nsubj,
                "nummod": num_nummod,
                "obj": num_obj,
                "obl": num_obl,
                "orphan": num_orphan,
                "parataxis": num_parataxis,
                "punct": num_punct,
                "reparandum": num_reparandum,
                "root": num_root,
                "vocative": num_vocative,
                "xcomp": num_xcomp, }
        except Exception as e:
            print("Error occurred during count syntax feature in feature extraction:", str(e))
            traceback.print_exc()
            return {}

    def calculate_dependency_features(self, text):
        try:
            self.validate_input(text)
            # calculate_dependency_features関数の内容をここにコピー
            """
                Calculate various dependency features of a sentence:
                - Maximum dependency depth
                - Number of nodes
                - Number of leaves
                - Total dependency distance
                """
            # GiNZAを用いた日本語モデルのロード
            #　前処理
            # 文の解析
            doc = self.nlp_speech(text)

            # フィラー、"INTJ"、 "reparandum" 依存関係のトークンを除外
            filtered_tokens = [token.text for token in doc if token.tag_ != "感動詞-フィラー" and token.pos_ != "INTJ" and token.dep_ != "reparandum"]
            result = ''.join(filtered_tokens)
            # 読点を削除
            result = result.replace("、", "")
            result = result.replace("。", "")

            # 文の解析
            doc = self.nlp_speech_CEJCminus(result)

            max_depth = 0
            total_distance = 0
            num_leaves = 0
            num_nodes = len(doc)

            # A dictionary to keep count of child nodes for each token
            child_counts = {token: 0 for token in doc}

            for token in doc:
                depth = 0
                current = token

                # Calculate depth and total dependency distance
                while current.head != current:
                    depth += 1
                    current = current.head
                    total_distance += 1

                max_depth = max(max_depth, depth)

                # Increment child count for the parent token
                child_counts[current.head] += 1

            # Count the number of leaves (tokens with no children)
            num_leaves = sum(1 for count in child_counts.values() if count == 0)

            return {'max_depth': max_depth,
                    "num_nodes": num_nodes,
                    "num_leaves": num_leaves,
                    "total_distance": total_distance}
        except Exception as e:
            print("Error occurred during calculate dependency features in feature extraction:", str(e))
            traceback.print_exc()
            return {}

    # 新しい冗長度計算関数
    def calculate_redundancy_with_ginza(self, text):
        try:
            self.validate_input(text)
            doc = self.nlp_speech(text)
            tokens = [token.text for token in doc]
            N = len(tokens)
            V = len(set(tokens))
            redundancy = N / V if V > 0 else float('inf')
            return redundancy
        except Exception as e:
            print("Error occurred during calculate redundancy with ginza in feature extraction:", str(e))
            traceback.print_exc()
            return {}

    def process_large_text(self, text):
        try:
            split_texts = split_text(text)
            print(f"Text split into {len(split_texts)} parts")

            features_list = []
            for part in split_texts:
                features = self.extract_features(part)
                features_list.append(features)

            combined_features = merge_features(features_list)
            return combined_features
        except Exception as e:
            print("Error processing large text:", str(e))
            traceback.print_exc()
            return {}
####ここでクラス終了

# 以下は分割文の処理に必要な関数
def split_text(text, max_byte=49149):
    encoded_text = text.encode('utf-8')
    if len(encoded_text) <= max_byte:
        return [text]

    split_index = max_byte
    while split_index > 0 and encoded_text[split_index] & 0xC0 == 0x80:
        split_index -= 1

    part1 = encoded_text[:split_index].decode('utf-8')
    part2 = encoded_text[split_index:].decode('utf-8')
    return [part1, part2]

def merge_features(features_list):
    merged = {}
    for features in features_list:
        for key, value in features.items():
            if isinstance(value, (int, float)):
                merged[key] = merged.get(key, 0) + value
            elif isinstance(value, list):
                merged[key] = merged.get(key, []) + value
            else:
                merged[key] = value
    return merged

if __name__ == '__main__':
    # 使用例
    extractor = JapaneseFeatureExtractor()
    text = "いろいろ自然言語処理の実装が進んできました。銀座は南瓜と違ってメンテナンスされているようです。ここまでは、句点があるようば文章でした。ここからは句点がないような文章を並べます。銀座でランチをご一緒しましょうそういえば、六本木は江戸時代にできた町？知らないなあどうしようか迷うところがあるね" \
         "ああ、だけど2000年の六本木も悪くないね綺麗な建物が並んでいたり樹が生えていたり今日は晴れです。それはそうなんだけど、今日どこ行った？うーん、どうしようかなあ。教えてあげてもいいけど、まー、うーん、そういうことはね、あんまり興味がないかあ。いつだっていいけどさ、予定教えて欲しいんだけど。さあ？知らないよお" \
         "自己紹介をします。えーと、うーん、と、あのさあ、名前を忘れてしまった。どうしよう。東京。六本木。新宿。昨日。明日。明後日。一昨日。平成。令和。昭和。キリスト。ブッダ。ゴッド。ヒンズー教。2020年。兄。アインシュタイン。" \
           "えーと、その、あの、うーん、と、あのさあ、そうだね、えっと、なんていうか、まあ、そういうことだよね。まあさー、でもね、君は素晴らしい人だと思うんだけど、うー、あのー、えーと、' \
           'けれどもねー、いくつかの問題点を指摘されたんだよね、上司に。"
    features = extractor.extract_features(text)
    print(features)
