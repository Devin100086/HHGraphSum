import os
import argparse
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
 
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

def GetType(path):
    filename = path.split("/")[-1]
    return filename.split(".")[0]

def catDoc(textlist):
    res = []
    for tlist in textlist:
        res.extend(tlist)
    return res

def get_tfidf_embedding(text):
    """
    
    :param text: list, sent_number * word
    :return: 
        vectorizer: 
            vocabulary_: word2id
            get_feature_names(): id2word
        tfidf: array [sent_number, max_word_number]
    """
    vectorizer = CountVectorizer(lowercase=True)
    word_count = vectorizer.fit_transform(text)
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(word_count)
    tfidf_weight = tfidf.toarray()
    return vectorizer, tfidf_weight
    
def compress_array(a, id2word):
    """
    
    :param a: matrix, [N, M], N is document number, M is word number
    :param id2word: word id to word
    :return: 
    """
    d = {}
    for i in range(len(a)):
        d[i] = {}
        for j in range(len(a[i])):
            if a[i][j] != 0:
                d[i][id2word[j]] = a[i][j]
    return d
    

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='data/CNNDM/test.label.jsonl', help='File to deal with')
    parser.add_argument('--dataset', type=str, default='CNNDM', help='dataset name')

    args = parser.parse_args()

    save_dir = os.path.join("cache", args.dataset)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    fname = GetType(args.data_path) + ".s2s.sim.jsonl"
    saveFile = os.path.join(save_dir, fname)
    print("Save topic2sent features of dataset %s to %s" % (args.dataset, saveFile))

    model = SentenceTransformer('/data/wcq/HeterGraph/all-mpnet-base-v2') 
    fout = open(saveFile, "w")
    with open(args.data_path) as f:
        num_lines = sum(1 for line in open(args.data_path))
        pbar = tqdm(total=num_lines)
        for line in f:
            e = json.loads(line)
            if isinstance(e["text"], list) and isinstance(e["text"][0], list):
                sents = catDoc(e["text"])
            else:
                sents = e["text"]
            sentence_embeddings = model.encode(sents, convert_to_numpy=True)
            cosine_scores = util.pytorch_cos_sim(sentence_embeddings, sentence_embeddings).tolist()
            fout.write(json.dumps(cosine_scores) + "\n")
            pbar.update(1)  
        pbar.close()

if __name__ == '__main__':
    main()
        
