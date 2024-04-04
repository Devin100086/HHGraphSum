import argparse
import os
import json
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel, LdaMulticore
from tqdm import tqdm
from gensim.test.utils import datapath
import gc


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="data", type=str)

# parser.add_argument("--topic_num", type=str)
args = parser.parse_args()
data_root = args.data_path
topic_num = 100
stop_words = stopwords.words('english') + ['!',',','.','?','-s','-ly','</s>','s', '(', ")", "@", "[", "]", "/", "_"]

def clean_sentence(sen):
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(sen)
    # print(words)
    return [w for w in words if w.lower() not in stop_words]

def readJson(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

data_names = ["MultiNews"]
splits = [ "train", "val", "test"]


for dataset_name in data_names:
    corpus = []
    for split in splits:
        # read data
        data_path = os.path.join(data_root, "{}".format(dataset_name), "{}.label.jsonl".format(split))
        docs = readJson(data_path)
        for doc in tqdm(docs):
            article_text = doc["summary"]
            if isinstance(doc["summary"], list) and isinstance(doc["summary"][0], list):  # multi document
                article_text = []
                for doc in doc["summary"]:
                    article_text.extend(doc)
            for sen in article_text:
                corpus.append(clean_sentence(sen))
    del docs
    gc.collect()
    # print("corpus:{}".format(len(corpus)))
    dictionary = Dictionary(corpus)
    input_text = [dictionary.doc2bow(text) for text in corpus]
    # print("input_text:{}".format(len(input_text)))

    lda = LdaMulticore(input_text, num_topics=topic_num, id2word=dictionary, passes=1, workers=8)
    lda_save_path = datapath("model")
    lda.save(lda_save_path)
    # print(lda.print_topics(50, 3))
    # print(lda.get_document_topics(input_text[0]))

    doc_lda_cluster = {}
    for split in splits:
        data_path = os.path.join(data_root, "{}".format(dataset_name), "{}.label.jsonl".format(split))
        docs = readJson(data_path)
        idx = 0
        doc_id = 0
        for doc in tqdm(docs):
            # doc_id = doc["article_id"]
            article_text = doc["summary"]
            if isinstance(doc["summary"], list) and isinstance(doc["summary"][0], list):  # multi document
                article_text = []
                for doc in doc["summary"]:
                    article_text.extend(doc)
            clst = [[] for i in range(0, topic_num)]
            for sen_id, sen in enumerate(article_text):
                doc_topic = lda.get_document_topics(input_text[idx])
                if len(doc_topic) > 0:
                    topic_idx = doc_topic[0][0]
                    clst[topic_idx].append(sen_id)
                idx += 1
            doc_lda_cluster[doc_id] = clst
            doc_id += 1
    # print(doc_lda_cluster)
    lda_clst_save_path = os.path.join(data_root, dataset_name, "lda_clst.npy")
    np.save(lda_clst_save_path, doc_lda_cluster)
    print("Saved:{}".format(dataset_name))
print("finished")





