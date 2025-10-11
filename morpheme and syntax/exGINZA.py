import spacy
import ginza

"""
2023/08/07
GINZAの実験。
https://note.com/npaka/n/n5c3e4ca67956
https://dev.classmethod.jp/articles/try-parsing-using-ginza/
https://megagonlabs.github.io/ginza/
を参照。
"""

if __name__ == '__main__':
    text="田中さん。神様。キリスト。ここは六本木。いろいろ自然言語処理の実装が進んできました。銀座は南瓜と違ってメンテナンスされているようです。ここまでは、句点があるようば文章でした。ここからは句点がないような文章を並べます。銀座でランチをご一緒しましょうそういえば、六本木は江戸時代にできた町？知らないなあどうしようか迷うところがあるね" \
         "ああ、だけど2000年の六本木も悪くないね綺麗な建物が並んでいたり樹が生えていたり今日は晴れです。それはそうなんだけど、今日どこ行った？うーん、どうしようかなあ。教えてあげてもいいけど、まー、うーん、そういうことはね、あんまり興味がないかあ。いつだっていいけどさ、予定教えて欲しいんだけど。さあ？知らないよお" \
         "自己紹介をします。えーと、うーん、と、あのさあ、名前を忘れてしまった。どうしよう。えーと、その、あの、うーん、と、あのさあ、そうだね、えっと、なんていうか、まあ、そういうことだよね。まあさー、でもね、君は素晴らしい人だと思うんだけど、うー、あのー、えーと、' \
           'けれどもねー、いくつかの問題点を指摘されたんだよね、上司に。そして、いつからか、あたたと、しかしながら、困った、と、言われても。そこで、おばあさんが、おじいさんと、一緒に、嘆くのか。"
    nlp = spacy.load('ja_ginza_electra')
    #nlp = spacy.load('ja_cejc_gsd_bert_v2')
    doc = nlp(text)
    print("doc.sents", doc.sents)
    for sent in doc.sents:
        for token in sent:
            print(
                token.i,
                token.orth_,
                token.lemma_,
                token.norm_,
                token.morph.get("Reading"),
                token.pos_,
                token.morph.get("Inflection"),
                token.tag_,
                token.dep_,
                token.head.i
            )
        print('EOS')

    print("ginza.bunsetu_spans(doc):",ginza.bunsetu_spans(doc))
    print(ginza.bunsetu_phrase_spans(doc))
    for phrase in ginza.bunsetu_phrase_spans(doc):
        print(phrase, phrase.label_)

    sentence = list(doc.sents)[0]
    for sentence in doc.sents:
        for relation, bunsetu in ginza.sub_phrases(sentence.root, ginza.bunsetu):
            print(relation, bunsetu)
        print("root", ginza.bunsetu(sentence.root))

    # 単語間の係り受け解析
    for sent in doc.sents:
        for token in sent:
            print(token.text + ' ← ' + token.head.text + ', ' + token.dep_)

    # グラフ表示
    #displacy.render(doc, style='dep', jupyter=True, options={'compact': True, 'distance': 90})

    # 固有表現抽出
    for ent in doc.ents:
        print(
            ent.text + ',' +  # テキスト
            ent.label_ + ',' +  # ラベル
            str(ent.start_char) + ',' +  # 開始位置
            str(ent.end_char))  # 終了位置

    # 文境界解析
    for sent in doc.sents:
        print(sent)


    # 話し言葉用のGiNZA
    print("===================ここから話し言葉用のGiNZAによる解析=====================")
    #nlp_speech = spacy.load('ja_cejc_gsd_bert_v2')
    nlp_speech = spacy.load('ja_cejc_parse_bert_v2')
    #nlp_speech = spacy.load('ja_cejc_gsd_bert_v2-0.1.0')
    doc_speech = nlp_speech(text)
    print("doc_speech.sents", doc_speech.sents)
    for sent in doc_speech.sents:
        for token in sent:
            print(
                token.i,
                token.orth_,
                token.lemma_,
                token.norm_,
                token.morph.get("Reading"),
                token.pos_,
                token.morph.get("Inflection"),
                token.tag_,
                token.dep_,
                token.head.i
            )
        print('EOS')


    '''
    sentence_speech = list(doc_speech.sents)[0]
    for sentence_speech in doc_speech.sents:
        for relation, bunsetu in ginza.sub_phrases(sentence_speech.root, ginza.bunsetu):
            print(relation, bunsetu)
        print("root", ginza.bunsetu(sentence_speech.root))
    '''

    # 単語間の係り受け解析
    for sent in doc_speech.sents:
        for token in sent:
            print(token.text + ' ← ' + token.head.text + ', ' + token.dep_)

    # グラフ表示
    #displacy.render(doc, style='dep', jupyter=True, options={'compact': True, 'distance': 90})

    # 固有表現抽出
    for ent in doc_speech.ents:
        print(
            ent.text + ',' +  # テキスト
            ent.label_ + ',' +  # ラベル
            str(ent.start_char) + ',' +  # 開始位置
            str(ent.end_char))  # 終了位置

    # 文境界解析
    for sent in doc_speech.sents:
        print(sent)

"""
    print(ginza.bunsetu_spans(doc_speech))
    print(ginza.bunsetu_phrase_spans(doc_speech))
    for phrase in ginza.bunsetu_phrase_spans(doc_speech):
        print(phrase, phrase.label_)

"""
