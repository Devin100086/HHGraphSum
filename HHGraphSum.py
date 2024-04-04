import numpy as np

import torch
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

import dgl

# from module.GAT import GAT, GAT_ffn
from module.Encoder import sentEncoder
from module.GAT import WSWGAT
from module.PositionEmbedding import get_sinusoid_encoding_table
from module.Transformer import *


class HHGraphSum(nn.Module):
    """ without sent2sent and add residual connection """
    def __init__(self, hps, embed):
        """

        :param hps: 
        :param embed: word embedding
        """
        super().__init__()

        self._hps = hps
        self._n_iter = hps.n_iter
        self._embed = embed
        self.embed_size = hps.word_emb_dim


        # sent node feature
        self._init_sn_param()
        self._TFembed = nn.Embedding(10, hps.feat_embed_size)   # box=10
        self._SIMembed = nn.Embedding(10,hps.feat_embed_size)   # box=10
        self.n_feature_proj = nn.Linear(hps.n_feature_size * 2, hps.hidden_size, bias=False)

        # word -> sent
        embed_size = hps.word_emb_dim
        self.word2sent = WSWGAT(in_dim=embed_size,
                                out_dim=hps.hidden_size,
                                num_heads=hps.n_head,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="W2S"
                                )

        # sent -> word
        self.sent2word = WSWGAT(in_dim=hps.hidden_size,
                                out_dim=embed_size,
                                num_heads=6,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="S2W"
                                )
        
        # sent -> sent
        self.sent2sent = WSWGAT(in_dim=hps.hidden_size,
                                out_dim=hps.hidden_size,
                                num_heads=hps.n_head,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="S2S",
                                )
        
        # word -> word
        self.word2word = WSWGAT(in_dim=embed_size,
                                out_dim=embed_size,
                                num_heads=6,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="W2W",
                                )

        # node classification
        self.n_feature = hps.hidden_size
        # self.MultiHeadAttention = MultiHeadAttention(self.n_feature, hps.n_head)
        # self.LayerNorm = nn.LayerNorm(self.n_feature)
        # self.Wc = nn.Linear(self.n_feature, self.n_feature, bias=False)
        # self.Wr = nn.Linear(self.n_feature, self.n_feature, bias=False)
        self.lstmout = nn.LSTM(self.n_feature,self.lstm_hidden_state, num_layers=self._hps.lstm_layers, dropout=0.1,
                            batch_first=True, bidirectional=False)
        self.wh = nn.Linear(self.n_feature, 2)

    def forward(self, graph):
        """
        :param graph: [batch_size] * DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
            edge:
                word2sent, sent2word:  tffrac=int, type=0
        :return: result: [sentnum, 2]
        """
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        # word node init
        word_feature = self.set_wnfeature(graph)    # [wnode, embed_size]

        sent_feature = self.n_feature_proj(self.set_snfeature(graph))    # [snode, n_feature_size]

        # the start state
        word_state = word_feature
        sent_state = self.word2sent(graph, word_feature, sent_feature)

        for i in range(self._n_iter):
            # word -> word
            word_state = self.word2word(graph, word_state, word_state)
            # sent -> word
            word_state = self.sent2word(graph, word_state, sent_state)
            # word -> sent
            sent_state = self.word2sent(graph, word_state, sent_state)
            #sent -> sent
            sent_state = self.sent2sent(graph, sent_state, sent_state)

        graph.nodes[snode_id].data["state"] = sent_state
        feature, glen = get_snode_feat(graph, feat="state")
        sent_state = self._sent_lstm_out_feature(feature ,glen)
        result = torch.sigmoid(self.wh(sent_state))

        return result

    def _init_sn_param(self):
        self.sent_pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self._hps.doc_max_timesteps + 1, self.embed_size, padding_idx=0),
            freeze=True)
        self.cnn_proj = nn.Linear(self.embed_size, self._hps.n_feature_size)
        self.lstm_hidden_state = self._hps.lstm_hidden_state
        self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden_state, num_layers=self._hps.lstm_layers, dropout=0.1,
                            batch_first=True, bidirectional=self._hps.bidirectional)
        if self._hps.bidirectional:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state * 2, self._hps.n_feature_size)
        else:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state, self._hps.n_feature_size)

        self.transformerEncoder = Encoder(self._hps.transformer_max_len, self._hps.n_feature_size, self._hps.ffn_inner_hidden_size,
                                          self._hps.transformer_head, self._hps.transformer_num_layers, self._hps.transformer_dropout, "cuda:0")
        self.transformer_proj = nn.Linear(self._hps.n_feature_size, self._hps.n_feature_size)

        self.ngram_enc = sentEncoder(self._hps, self._embed)

    def _sent_cnn_feature(self, graph, snode_id):
        ngram_feature = self.ngram_enc.forward(graph.nodes[snode_id].data["words"])  # [snode, embed_size]
        graph.nodes[snode_id].data["sent_embedding"] = ngram_feature
        snode_pos = graph.nodes[snode_id].data["position"].view(-1)  # [n_nodes]
        position_embedding = self.sent_pos_embed(snode_pos)
        cnn_feature = self.cnn_proj(ngram_feature + position_embedding)
        return cnn_feature

    def _sent_lstm_feature(self, features, glen):
        pad_seq = rnn.pad_sequence(features, batch_first=True)
        lstm_input = rnn.pack_padded_sequence(pad_seq, glen, batch_first=True)
        lstm_output, _ = self.lstm(lstm_input)
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output, batch_first=True)
        lstm_embedding = [unpacked[i][:unpacked_len[i]] for i in range(len(unpacked))]
        lstm_feature = self.lstm_proj(torch.cat(lstm_embedding, dim=0))  # [n_nodes, n_feature_size]
        return lstm_feature

    def _sent_transformer_feature(self, features, glen):
        pad_seq = rnn.pad_sequence(features, batch_first=True)
        mask = get_encoder_mask(glen).to(pad_seq.device)
        transformer_feature = self.transformerEncoder(pad_seq, mask)
        transformer_feature = self.transformer_proj(torch.cat([transformer_feature[i][:pack] for i,pack in enumerate(glen)]))

        return transformer_feature

    def _sent_lstm_out_feature(self, features, glen):
        pad_seq = rnn.pad_sequence(features, batch_first=True)
        lstm_input = rnn.pack_padded_sequence(pad_seq, glen, batch_first=True)
        lstm_output, _ = self.lstmout(lstm_input)
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output, batch_first=True)
        lstm_embedding = [unpacked[i][:unpacked_len[i]] for i in range(len(unpacked))]
        lstm_feature = torch.cat(lstm_embedding, dim=0)  # [n_nodes, n_feature_size]
        return lstm_feature

    def set_wnfeature(self, graph):
        wnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"]==0)
        # wwedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 2)
        wsedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 0)   # for word to supernode(sent&doc)
        ssedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 1)   
        wid = graph.nodes[wnode_id].data["id"]  # [n_wnodes]
        w_embed = self._embed(wid)  # [n_wnodes, D]
        graph.nodes[wnode_id].data["embed"] = w_embed
        wsetf = graph.edges[wsedge_id].data["tffrac"]
        ssetf = graph.edges[ssedge_id].data["simfrac"]
        # wwetf = graph.edges[wwedge_id].data["nextfrac"]
        # graph.edges[wwedge_id].data["nextidembed"] = self._Nextembed(wwetf)
        graph.edges[wsedge_id].data["tfidfembed"] = self._TFembed(wsetf)
        graph.edges[ssedge_id].data["simidfembed"] = self._SIMembed(ssetf)
        return w_embed

    def set_snfeature(self, graph):
        # node feature
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        cnn_feature = self._sent_cnn_feature(graph, snode_id)
        features, glen = get_snode_feat(graph, feat="sent_embedding")
        lstm_feature = self._sent_lstm_feature(features, glen)
        graph.nodes[snode_id].data["lstm_feature"] = lstm_feature
        # TODO : Get Transformer feature
        lstm_features, glen = get_snode_feat(graph, feat="lstm_feature")
        transformer_feature = self._sent_transformer_feature(lstm_features, glen)
        
        node_feature = torch.cat([cnn_feature, transformer_feature], dim=1)  # [n_nodes, n_feature_size * 2]
        return node_feature


class HHDocGraphSum(HHGraphSum):
    """
        without sent2sent and add residual connection
        add Document Nodes
    """

    def __init__(self, hps, embed):
        super().__init__(hps, embed)
        self.dn_feature_proj = nn.Linear(hps.hidden_size, hps.hidden_size, bias=False)
        self.MultiHeadAttention = MultiHeadAttention(2 * self.n_feature, hps.n_head)
        self.LayerNorm = nn.LayerNorm(self.n_feature * 2)
        self.Wc = nn.Linear(self.n_feature * 2, self.n_feature * 2, bias=False)
        self.Wr = nn.Linear(self.n_feature * 2, self.n_feature * 2, bias=False)
        self.fg = nn.Linear(self.n_feature * 4, self.n_feature * 2, bias=False)
        self.lstmout = nn.LSTM(self.n_feature * 2, self.n_feature * 2, num_layers=self._hps.lstm_layers, dropout=0.1,
                            batch_first=True, bidirectional=False)
        self.wh = nn.Linear(self.n_feature * 2, 2)

    def forward(self, graph):
        """
        :param graph: [batch_size] * DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
                document: unit=1, dtype=2
            edge:
                word2sent, sent2word: tffrac=int, type=0
                word2doc, doc2word: tffrac=int, type=0
                sent2doc: type=2
        :return: result: [sentnum, 2]
        """

        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        dnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        supernode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 1)

        # word node init
        word_feature = self.set_wnfeature(graph)    # [wnode, embed_size]
        sent_feature = self.n_feature_proj(self.set_snfeature(graph))    # [snode, n_feature_size]

        # sent and doc node init
        graph.nodes[snode_id].data["init_feature"] = sent_feature
        doc_feature, snid2dnid = self.set_dnfeature(graph)
        doc_feature = self.dn_feature_proj(doc_feature)
        graph.nodes[dnode_id].data["init_feature"] = doc_feature

        # the start state
        word_state = word_feature
        sent_state = graph.nodes[supernode_id].data["init_feature"]

        for i in range(self._n_iter):
            # word -> word
            word_state = self.word2word(graph, word_state, word_state)
            # sent -> word
            word_state = self.sent2word(graph, word_state, sent_state)
            # word -> sent
            sent_state = self.word2sent(graph, word_state, sent_state)
            #sent -> sent
            sent_state = self.sent2sent(graph, sent_state, sent_state)

        graph.nodes[supernode_id].data["hidden_state"] = sent_state

        # extract sentence nodes
        s_state_list = []
        for snid in snode_id:
            d_state = graph.nodes[snid2dnid[int(snid)]].data["hidden_state"]
            s_state = graph.nodes[snid].data["hidden_state"]
            s_state = torch.cat([s_state, d_state], dim=-1)
            s_state_list.append(s_state)

        s_state = torch.cat(s_state_list, dim=0)
        graph.nodes[snode_id].data["state"] = s_state
        feature, glen = get_snode_feat(graph, feat="state")
        s_state = self._sent_lstm_out_feature(feature ,glen)
        result = self.wh(s_state)
        return result

    def set_dnfeature(self, graph):
        """ init doc node by mean pooling on the its sent node (connected by the edges with type=1) """
        dnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        node_feature_list = []
        snid2dnid = {}
        for dnode in dnode_id:
            snodes = [nid for nid in graph.predecessors(dnode) if graph.nodes[nid].data["dtype"]==1]
            doc_feature = graph.nodes[snodes].data["init_feature"].mean(dim=0)
            assert not torch.any(torch.isnan(doc_feature)), "doc_feature_element"
            node_feature_list.append(doc_feature)
            for s in snodes:
                snid2dnid[int(s)] = dnode
        node_feature = torch.stack(node_feature_list)
        return node_feature, snid2dnid

def get_snode_feat(G, feat):
    glist = dgl.unbatch(G)
    feature = []
    glen = []
    for g in glist:
        snode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        feature.append(g.nodes[snode_id].data[feat])
        glen.append(len(snode_id))
    return feature, glen
