import esm

import config

from tqdm import tqdm
import os
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

from utils import collate_fn

class ESMDataset(Dataset):
    def __init__(self,row, datatype):
        super().__init__()
        self.seq = row[f'{datatype}_seq']
        self.aa = row[f'{datatype}_aa']
        self.gene_id = row['record_id']
        self.aa_index = row['new_index']
        self.label = row['label']
    def __len__(self):
        return len(self.seq)
    def __getitem__(self, idx):
        return (self.label[idx],self.seq[idx],self.aa[idx],self.gene_id[idx],self.aa_index[idx])


def _build_wt_cache(df, model, batch_converter, device):
    unique_wt_seqs = df['wt_seq'].unique().tolist()
    wt_unique_dataset = [(str(i), seq) for i, seq in enumerate(unique_wt_seqs)]
    bs = config.batch_size_for_embed_gen
    softmax = nn.Softmax(dim=-1)
    wt_cache = {}

    print(f"****** Building WT cache: {len(unique_wt_seqs)} unique sequences "
          f"({len(df) - len(unique_wt_seqs)} redundant ESM passes saved)...")

    for batch_start in tqdm(range(0, len(wt_unique_dataset), bs)):
        batch = wt_unique_dataset[batch_start: batch_start + bs]
        _, _, tokens = batch_converter(batch)

        with torch.no_grad():
            result = model(tokens.to(device), repr_layers=[33])

        reprs   = result["representations"][33].detach().cpu()
        logits  = result['logits'][:, :, 4:24].detach().cpu()

        for idx_in_batch, (_, seq) in enumerate(batch):
            seq_logits = logits[idx_in_batch]
            wt_cache[seq] = {
                'repr':   reprs[idx_in_batch],
                'logits': seq_logits,
                'probs': softmax(seq_logits),
            }
    return wt_cache


def _lookup_wt(wt_seq_batch, aa_index_batch, wt_cache):
    wt_repr_list = []
    wt_probs_list = []

    for seq, idx in zip(wt_seq_batch, aa_index_batch):
        cached = wt_cache[seq]
        wt_repr_list.append(cached['repr'][idx])
        wt_probs_list.append(cached['probs'][idx])

    wt_repr       = torch.stack(wt_repr_list)
    wt_probs      = torch.stack(wt_probs_list)

    return wt_repr, wt_probs


def _run_mt_batches(df, model, batch_converter, wt_cache, esm_dict, device):
    mt_dataset    = ESMDataset(df, datatype="mt")
    mt_dataloader = DataLoader(
        mt_dataset,
        batch_size=config.batch_size_for_embed_gen,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False
    )

    record_to_idx = {str(row['record_id']): i for i, row in df.iterrows()}

    label_for_embeds = []
    gene_id_list = []
    concat_batches = []
    logits_batches = []

    print(f"****** Running MT embeddings: ")

    for batch in tqdm(mt_dataloader):
        labels_batch, _, mt_batch_tokens = batch_converter(batch[0])
        mt_aa_batch, gene_id_batch, aa_index_batch = batch[1], batch[2], batch[3]

        row_indices = [record_to_idx[str(id)] for id in gene_id_batch]
        wt_seq_batch = [df.at[i, 'wt_seq'] for i in row_indices]
        wt_aa_batch = [df.at[i, 'wt_aa'] for i in row_indices]

        label_for_embeds.append(labels_batch)
        gene_id_list.append(gene_id_batch)

        aa_index = torch.tensor(aa_index_batch)
        batch_indices = torch.arange(len(aa_index))

        with torch.no_grad():
            mt_result = model(mt_batch_tokens.to(device), repr_layers=[33])

        mt_repr = mt_result["representations"][33][batch_indices, aa_index].detach().cpu()

        wt_repr, wt_probs = _lookup_wt(wt_seq_batch, aa_index_batch, wt_cache)

        esm_aa_ids_wt = torch.tensor([esm_dict[a] - 4 for a in wt_aa_batch])
        esm_aa_ids_mt = torch.tensor([esm_dict[a] - 4 for a in mt_aa_batch])
        row_idx = torch.arange(len(wt_seq_batch))

        wt_prob = wt_probs[row_idx, esm_aa_ids_wt]
        mt_prob = wt_probs[row_idx, esm_aa_ids_mt]

        combined = torch.cat((wt_repr, mt_repr), dim=1)

        llr = torch.log(mt_prob / wt_prob).unsqueeze(1)

        concat_batches.append(combined)
        logits_batches.append(llr)

    concat = torch.cat(concat_batches, dim=0)
    logits_list = torch.cat(logits_batches, dim=0)
    gene_list = [str(x) for tup in gene_id_list for x in tup]
    label_list = [x for item in label_for_embeds for x in item]

    return concat, logits_list, label_list, gene_list


def generate_embeds_and_save(df, save_path, data_class, device=config.device):
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    esm_dict     = alphabet.tok_to_idx
    batch_converter = alphabet.get_batch_converter()
    model        = model.to(device)

    wt_cache = _build_wt_cache(df, model, batch_converter, device)
    concat, logits_list, label_list, gene_list = _run_mt_batches(
        df, model, batch_converter, wt_cache, esm_dict, device
    )

    final_result = {
        'x':         concat,
        'logits':    logits_list,
        'label':     label_list,
        'record_id': gene_list,
    }

    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, data_class)
    print(f"****** {data_class} embeddings saved to: {save_file}.pt ******")
    torch.save(final_result, f'{save_file}.pt')
    return f'{save_file}.pt'
