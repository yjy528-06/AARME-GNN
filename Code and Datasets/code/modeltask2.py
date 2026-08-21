import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import os


class Att_Adapter(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.down_proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.up_proj = nn.Linear(latent_dim, input_dim)

        self.channel_attention = nn.Sequential(
            nn.Linear(input_dim, max(4, input_dim // 8)),
            nn.ReLU(),
            nn.Linear(max(4, input_dim // 8), input_dim),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        identity = x

        down = self.down_proj(x)
        up = self.up_proj(down)

        ca_weights = self.channel_attention(x)
        adapted = up * ca_weights

        output = identity + self.dropout(adapted)

        return output


class RouterNetwork(nn.Module):

    def __init__(self, input_dim, num_experts, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
        self.num_experts = num_experts

    def forward(self, x):
        logits = self.net(x)
        weights = F.softmax(logits, dim=-1)
        return weights


class GNN1(nn.Module):
    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, **kwargs):
        super(GNN1, self).__init__(**kwargs)
        self.kg, self.dict1 = dataset["dataset1"], dict1
        self.drug_name, self.args = drug_name, args
        self.drug_embed = nn.Embedding(num_embeddings=572, embedding_dim=args.embedding_num)
        self.rela_embed = nn.Embedding(num_embeddings=relation_len["dataset1"], embedding_dim=args.embedding_num)
        self.ent_embed = nn.Embedding(num_embeddings=tail_len["dataset1"], embedding_dim=args.embedding_num)
        self.W1 = nn.Parameter(torch.randn(size=(572, args.embedding_num, args.embedding_num)))
        self.b1 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.W2 = nn.Parameter(torch.randn(size=(572, args.embedding_num, args.embedding_num)))
        self.b2 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.Linear1 = nn.Sequential(nn.Linear(args.embedding_num * 2, args.embedding_num),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(args.embedding_num))

        self.aux_head = nn.Linear(args.embedding_num, args.event_num)

        self.adapter = Att_Adapter(
            input_dim=args.embedding_num,
            latent_dim=64
        )

        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, datas):
        kg, dict1, args, drug_name = self.kg, self.dict1, self.args, self.drug_name
        adj_tail, adj_relation = self.arrge(kg, dict1, args.neighbor_sample_size)
        drug_name = torch.LongTensor(drug_name)
        adj_tail = torch.LongTensor(adj_tail)
        adj_relation = torch.LongTensor(adj_relation)
        drug_embedding = self.drug_embed(drug_name)
        rela_embedding = self.rela_embed(adj_relation)
        ent_embedding = self.ent_embed(adj_tail)
        drug_rel = drug_embedding.reshape((572, 1, args.embedding_num)) * rela_embedding
        drug_rel_weigh = drug_rel.matmul(self.W1) + self.b1
        drug_rel_weigh = self.relu(drug_rel_weigh)
        drug_rel_weigh = drug_rel_weigh.matmul(self.W2) + self.b2
        drug_rel_score = torch.sum(drug_rel_weigh, axis=-1, keepdims=True)
        drug_rel_score = self.soft(drug_rel_score)
        weighted_ent = drug_rel_score.reshape((572, 1, args.neighbor_sample_size)).matmul(ent_embedding)
        drug_e = torch.cat(
            [weighted_ent.reshape(572, args.embedding_num), drug_embedding.reshape((572, args.embedding_num))], dim=1)
        drug_f = self.Linear1(drug_e)

        calibrated_drug_f = self.adapter(drug_f)

        aux_output = self.aux_head(calibrated_drug_f)

        idx, train_or_test, test_adj = datas[0], datas[1], datas[2]
        if train_or_test == 1:
            for i in test_adj[0].keys():
                pos = test_adj[0][i][0]
                length = len(pos)
                calibrated_drug_f[i] = torch.sum(calibrated_drug_f[pos], dim=0) / length
                aux_output[i] = torch.sum(aux_output[pos], dim=0) / length

        return calibrated_drug_f, aux_output, idx, test_adj, train_or_test

    def arrge(self, kg, drug_name_id, neighbor_sample_size, n_drug=572):
        adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        for i in drug_name_id:
            all_neighbors = kg[drug_name_id[i]]
            n_neighbor = len(all_neighbors)
            sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
                replace=False if n_neighbor >= neighbor_sample_size else True
            )
            adj_tail[drug_name_id[i]] = np.array([all_neighbors[i][0] for i in sample_indices])
            adj_relation[drug_name_id[i]] = np.array([all_neighbors[i][1] for i in sample_indices])
        return adj_tail, adj_relation


class GNN2(nn.Module):
    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, **kwargs):
        super(GNN2, self).__init__(**kwargs)
        self.kg, self.dict1 = dataset["dataset2"], dict1
        self.drug_name, self.args = drug_name, args
        self.drug_embed = nn.Embedding(num_embeddings=572, embedding_dim=args.embedding_num)
        self.rela_embed = nn.Embedding(num_embeddings=relation_len["dataset2"], embedding_dim=args.embedding_num)
        self.ent_embed = nn.Embedding(num_embeddings=tail_len["dataset2"], embedding_dim=args.embedding_num)
        self.W1 = nn.Parameter(torch.randn(size=(572, args.embedding_num, args.embedding_num)))
        self.b1 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.W2 = nn.Parameter(torch.randn(size=(572, args.embedding_num, args.embedding_num)))
        self.b2 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.Linear1 = nn.Sequential(nn.Linear(args.embedding_num * 2, args.embedding_num),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(args.embedding_num))

        self.aux_head = nn.Linear(args.embedding_num, args.event_num)

        self.adapter = Att_Adapter(
            input_dim=args.embedding_num,
            latent_dim=64
        )

        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, arguments):
        kg, dict1, args, drug_name = self.kg, self.dict1, self.args, self.drug_name
        gnn1_embedding, gnn1_aux, idx, test_adj, train_or_test = arguments

        adj_tail, adj_relation = self.arrge(kg, dict1, args.neighbor_sample_size)
        drug_name = torch.LongTensor(drug_name)
        adj_tail = torch.LongTensor(adj_tail)
        adj_relation = torch.LongTensor(adj_relation)
        drug_embedding = self.drug_embed(drug_name)
        rela_embedding = self.rela_embed(adj_relation)
        ent_embedding = self.ent_embed(adj_tail)
        drug_rel = drug_embedding.reshape((572, 1, args.embedding_num)) * rela_embedding
        drug_rel_weigh = drug_rel.matmul(self.W1) + self.b1
        drug_rel_weigh = self.relu(drug_rel_weigh)
        drug_rel_weigh = drug_rel_weigh.matmul(self.W2) + self.b2
        drug_rel_score = torch.sum(drug_rel_weigh, axis=-1, keepdims=True)
        drug_rel_score = self.soft(drug_rel_score)
        weighted_ent = drug_rel_score.reshape((572, 1, args.neighbor_sample_size)).matmul(ent_embedding)
        drug_e = torch.cat(
            [weighted_ent.reshape(572, args.embedding_num), drug_embedding.reshape((572, args.embedding_num))], dim=1)
        drug_f = self.Linear1(drug_e)

        calibrated_drug_f = self.adapter(drug_f)

        aux_output = self.aux_head(calibrated_drug_f)

        if train_or_test == 1:
            for i in test_adj[1].keys():
                pos = test_adj[1][i][0]
                length = len(pos)
                calibrated_drug_f[i] = torch.sum(calibrated_drug_f[pos], dim=0) / length
                aux_output[i] = torch.sum(aux_output[pos], dim=0) / length

        return calibrated_drug_f, aux_output, gnn1_embedding, gnn1_aux, idx, test_adj, train_or_test

    def arrge(self, kg, drug_name_id, neighbor_sample_size, n_drug=572):
        adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        for i in drug_name_id:
            all_neighbors = kg[drug_name_id[i]]
            n_neighbor = len(all_neighbors)
            sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
                replace=False if n_neighbor >= neighbor_sample_size else True
            )
            adj_tail[drug_name_id[i]] = np.array([all_neighbors[i][0] for i in sample_indices])
            adj_relation[drug_name_id[i]] = np.array([all_neighbors[i][1] for i in sample_indices])
        return adj_tail, adj_relation


class GNN3(nn.Module):
    def __init__(self, dataset, tail_len, relation_len, args, dict1, drug_name, **kwargs):
        super(GNN3, self).__init__(**kwargs)
        self.kg, self.dict1 = dataset["dataset3"], dict1
        self.drug_name, self.args = drug_name, args
        self.drug_embed = nn.Embedding(num_embeddings=572, embedding_dim=args.embedding_num)
        self.rela_embed = nn.Embedding(num_embeddings=67, embedding_dim=args.embedding_num)
        self.ent_embed = nn.Embedding(num_embeddings=572, embedding_dim=args.embedding_num)
        self.W1 = nn.Parameter(torch.randn(size=(572, args.embedding_num, args.embedding_num)))
        self.b1 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.W2 = nn.Parameter(torch.randn(size=(572, args.embedding_num, args.embedding_num)))
        self.b2 = nn.Parameter(torch.randn(size=(args.neighbor_sample_size, args.embedding_num)))
        self.Linear1 = nn.Sequential(nn.Linear(args.embedding_num * 2, args.embedding_num),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(args.embedding_num))

        self.aux_head = nn.Linear(args.embedding_num, args.event_num)

        self.adapter = Att_Adapter(
            input_dim=args.embedding_num,
            latent_dim=64
        )

        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, arguments):
        kg, dict1, args, drug_name = self.kg, self.dict1, self.args, self.drug_name
        gnn2_embedding, gnn2_aux, gnn1_embedding, gnn1_aux, idx, test_adj, train_or_test = arguments

        adj_tail, adj_relation = self.arrge(kg, dict1, args.neighbor_sample_size)
        drug_name = torch.LongTensor(drug_name)
        adj_tail = torch.LongTensor(adj_tail)
        adj_relation = torch.LongTensor(adj_relation)
        drug_embedding = self.drug_embed(drug_name)
        rela_embedding = self.rela_embed(adj_relation)
        ent_embedding = self.ent_embed(adj_tail)
        drug_rel = drug_embedding.reshape((572, 1, args.embedding_num)) * rela_embedding
        drug_rel_weigh = drug_rel.matmul(self.W1) + self.b1
        drug_rel_weigh = self.relu(drug_rel_weigh)
        drug_rel_weigh = drug_rel_weigh.matmul(self.W2) + self.b2
        drug_rel_score = torch.sum(drug_rel_weigh, axis=-1, keepdims=True)
        drug_rel_score = self.soft(drug_rel_score)
        weighted_ent = drug_rel_score.reshape((572, 1, args.neighbor_sample_size)).matmul(ent_embedding)
        drug_e = torch.cat(
            [weighted_ent.reshape(572, args.embedding_num), drug_embedding.reshape((572, args.embedding_num))], dim=1)
        drug_f = self.Linear1(drug_e)

        calibrated_drug_f = self.adapter(drug_f)

        aux_output = self.aux_head(calibrated_drug_f)

        if train_or_test == 1:
            for i in test_adj[3].keys():
                pos = test_adj[3][i][0]
                length = len(pos)
                calibrated_drug_f[i] = torch.sum(calibrated_drug_f[pos], dim=0) / length
                aux_output[i] = torch.sum(aux_output[pos], dim=0) / length

        return calibrated_drug_f, aux_output, gnn2_embedding, gnn2_aux, gnn1_embedding, gnn1_aux, idx

    def arrge(self, kg, drug_name_id, neighbor_sample_size, n_drug=572, tails_num=570, relations_num=65):
        drug_number = []
        drug_list = []
        for i in drug_name_id:
            drug_number.append(drug_name_id[i])
        for key in kg:
            drug_list.append(key)
        surplus = set(drug_number).difference(set(drug_list))
        for i in list(surplus):
            kg[i].append((tails_num + 1, relations_num + 1))
        adj_tail = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        adj_relation = np.zeros(shape=(n_drug, neighbor_sample_size), dtype=np.int64)
        for i in drug_name_id:
            all_neighbors = kg[drug_name_id[i]]
            n_neighbor = len(all_neighbors)
            sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
                replace=False if n_neighbor >= neighbor_sample_size else True
            )
            adj_tail[drug_name_id[i]] = np.array([all_neighbors[i][0] for i in sample_indices])
            adj_relation[drug_name_id[i]] = np.array([all_neighbors[i][1] for i in sample_indices])
        return adj_tail, adj_relation


class FusionLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.router = RouterNetwork(input_dim=args.embedding_num * 6, num_experts=3, hidden_dim=64)

        self.fullConnectionLayer = nn.Sequential(
            nn.Linear(args.embedding_num * 2, args.embedding_num * 4),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 4),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 4, args.embedding_num * 2),
            nn.ReLU(),
            nn.BatchNorm1d(args.embedding_num * 2),
            nn.Dropout(args.dropout),
            nn.Linear(args.embedding_num * 2, 65)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, arguments):
        gnn3_embedding, gnn3_aux, gnn2_embedding, gnn2_aux, gnn1_embedding, gnn1_aux, idx = arguments

        idx = idx.numpy().tolist()
        drugA = []
        drugB = []
        for i in idx:
            drugA.append(i[0])
            drugB.append(i[1])

        emb_A_mod1 = gnn1_embedding[drugA]
        emb_A_mod2 = gnn2_embedding[drugA]
        emb_A_mod3 = gnn3_embedding[drugA]
        emb_B_mod1 = gnn1_embedding[drugB]
        emb_B_mod2 = gnn2_embedding[drugB]
        emb_B_mod3 = gnn3_embedding[drugB]
        router_input = torch.cat([
            emb_A_mod1, emb_A_mod2, emb_A_mod3,
            emb_B_mod1, emb_B_mod2, emb_B_mod3
        ], dim=1)

        modality_weights = self.router(router_input)

        weighted_emb_A_mod1 = emb_A_mod1 * modality_weights[:, 0].unsqueeze(1)
        weighted_emb_A_mod2 = emb_A_mod2 * modality_weights[:, 1].unsqueeze(1)
        weighted_emb_A_mod3 = emb_A_mod3 * modality_weights[:, 2].unsqueeze(1)
        weighted_emb_B_mod1 = emb_B_mod1 * modality_weights[:, 0].unsqueeze(1)
        weighted_emb_B_mod2 = emb_B_mod2 * modality_weights[:, 1].unsqueeze(1)
        weighted_emb_B_mod3 = emb_B_mod3 * modality_weights[:, 2].unsqueeze(1)

        fused_emb_A = weighted_emb_A_mod1 + weighted_emb_A_mod2 + weighted_emb_A_mod3
        fused_emb_B = weighted_emb_B_mod1 + weighted_emb_B_mod2 + weighted_emb_B_mod3

        fused_pair_representation = torch.cat([fused_emb_A, fused_emb_B], dim=1)

        main_output = self.fullConnectionLayer(fused_pair_representation)

        auxiliary_outputs = {
            'mod1': gnn1_aux[drugA],
            'mod2': gnn2_aux[drugA],
            'mod3': gnn3_aux[drugA]
        }

        return main_output, auxiliary_outputs, modality_weights