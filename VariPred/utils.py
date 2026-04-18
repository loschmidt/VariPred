from pathlib import Path

import config

from tqdm import tqdm
import os
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

# Data process part
def get_truncation(chop):

    chop.reset_index(drop=True, inplace=True)

    print(' The amount of sequences need to be truncated: ', len(chop))

    for index, seq in tqdm(chop.iterrows(), total=chop.shape[0]):

        if seq['aa_index'] < 1022:
            select_wt = seq['wt_seq'][0:1022]
            select_mt = seq['mt_seq'][0:1022]
            chop.loc[index, 'wt_seq'] = select_wt
            chop.loc[index, 'mt_seq'] = select_mt
            chop.loc[index,'new_index'] = seq['aa_index']

        elif seq['aa_index'] > seq['Length'] - 1022:
            select_wt = seq['wt_seq'][-1022:]
            select_mt = seq['mt_seq'][-1022:]
            chop.loc[index, 'wt_seq'] = select_wt
            chop.loc[index, 'mt_seq'] = select_mt
            chop.loc[index,'new_index'] = seq['aa_index']-seq['Length']+1022
        else:
            select_wt = seq['wt_seq'][seq['aa_index'] -
                                      511:seq['aa_index'] + 511]
            select_mt = seq['mt_seq'][seq['aa_index'] -
                                      511:seq['aa_index'] + 511]
            chop.loc[index, 'wt_seq'] = select_wt
            chop.loc[index, 'mt_seq'] = select_mt
            chop.loc[index,'new_index'] = 511
    chop["new_index"] = chop["new_index"].astype(int)

    return chop


def df_process(df):
    remain_df = df[df['Length'] <= 1022]
    trunc_df = df[df['Length'] > 1022]

    remain_df['new_index'] = remain_df['aa_index']

    truncated_df = get_truncation(trunc_df)

    truncated_result = pd.concat(
        [remain_df, truncated_df]).reset_index(drop=True)

    return truncated_result

        
def collate_fn(batch):
    labels, sequences, aa, gene_id, aa_index = zip(*batch)
    return list(zip(labels, sequences)), aa, gene_id, aa_index


# model training part:

# fetch the embeddings
def unpickler(ds_name):

    path = f'{config.esm_storage_path}/{ds_name}.pt'

    pt_embeds = torch.load(path)
    data_X = np.array(pt_embeds['x'])
    logits = np.array(pt_embeds['logits']).reshape(-1, 1)
    
    data_y = pt_embeds['label']
    record_id = pt_embeds['record_id']

    data_X = np.hstack((data_X, logits))

    return data_X, data_y, record_id


# Prepare datasets for models
class VariPredDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()

        self.seq = torch.tensor(X)
        self.label = torch.tensor(y)

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        return self.seq[index], self.label[index]


# model architecture setup
class MLPClassifier_LeakyReLu(nn.Module):
    """Simple MLP Model for Classification Tasks.
    """

    def __init__(self, num_input, num_hidden, num_output):
        super(MLPClassifier_LeakyReLu, self).__init__()

        # Instantiate an one-layer feed-forward classifier
        self.hidden = nn.Linear(num_input, num_hidden)
        self.predict = nn.Sequential(
            nn.Dropout(0.5),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_hidden, num_output)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.predict(x)
        x = self.softmax(x)

        return x


# train the model
def flat_accuracy(preds, labels):
    preds = preds.detach().cpu().numpy()
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def trainer(train_loader, val_loader, model, device=config.device, early_stop=config.early_stop, n_epochs=config.n_epochs):

    criterion = nn.BCELoss(reduction='sum')  # Define the loss function

    # Define the optimization algorithm.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=0)
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                         num_warmup_steps= 0,
    #                                         num_training_steps= len(train_loader)*n_epochs)

    n_epochs, best_loss, step, early_stop_count = n_epochs, math.inf, 0, early_stop

    for epoch in range(n_epochs):
        model.train()  # Set the model to train mode.
        loss_record = []

        train_pbar = tqdm(train_loader, position=0, leave=True)

        x = []
        for batch in train_pbar:
            optimizer.zero_grad()               # Set gradient to zero.
            # Move the data to device.
            b_seq, b_labels = tuple(t.to(device) for t in batch)

            pred = model(b_seq.float())
            b_labels = b_labels.float()
            loss = criterion(pred[:, 0], b_labels)

            # Compute gradient(backpropagation).
            loss.backward()

            optimizer.step()                    # Update parameters.
            # scheduler.step()

            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)

        ########### =========================== Evaluation=========================################
        print('\n\n###########=========================== Evaluating=========================################\n\n')

        model.eval()  # Set the model to evaluation mode.
        loss_record = []
        total_eval_accuracy = 0

        preds = []
        labels = []

        val_pbar = tqdm(val_loader, position=0, leave=True)
        for batch in val_pbar:

            # Move your data to device.
            b_seq, b_labels = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                b_labels = b_labels.float()
                pred = model(b_seq.float())
                loss = criterion(pred[:, 0], b_labels)

                # preds.append(pred[:,0].detach().cpu()[0].tolist())
                # labels.append(b_labels.detach().cpu()[0].tolist())

            loss_record.append(loss.item())
            total_eval_accuracy += flat_accuracy(pred, b_labels)

            val_pbar.set_description(f'Evaluating [{epoch + 1}/{n_epochs}]')
            val_pbar.set_postfix({'evaluate loss': loss.detach().item()})

        # For selecting the best MCC threshold
        # breakpoint()
        # y_true_np = np.array(labels)
        # pred_np = np.array(preds)
        # for label, pred_value in zip(y_true_np, pred_np):
        #     with open(f'./threhold_pick.txt', 'a+') as f:
        #         f.write(f'{label}\t{pred_value}\n')

        mean_valid_loss = sum(loss_record)/len(loss_record)
        avg_val_accuracy = total_eval_accuracy / len(val_loader)

        print(f'\nEpoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss

            storage_path = f'./model'

            if not os.path.isdir(f'{storage_path}'):
                # Create directory of saving models.
                os.mkdir(f'{storage_path}')

            torch.save({
                'model_state_dict': model.state_dict(), },
                f'{storage_path}/model.ckpt')  # Save the best model

            print('\nSaving model with loss {:.3f}...'.format(best_loss))

            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= early_stop:
            print('\nModel is not improving, so we halt the training session.')

            return


def predict(test_loader, model, device):
    model.eval()  # Set the model to evaluation mode.
    preds = []
    labels = []
    for batch in tqdm(test_loader):
        b_seq, b_labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            pred = model(b_seq.float())
            preds.append(pred[:, 0].detach().cpu())
            labels.append(b_labels.detach().cpu())
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)

    return preds, labels


def predict_results(y_true, preds, record_id, train=False, output_name=f'../example/output_results'):
    result_file = Path(output_name)
    result_path = result_file.parent.absolute()

    if not os.path.exists(f'{result_path}'):
        os.makedirs(result_path)

    if train:

        label_names = {'0': 0, '1': 1}

        auc_value = roc_auc_score(y_true, preds)
        print('AUC score: ', auc_value)

        y_true_np = np.array(y_true)
        preds = np.array(preds >= 0.2, dtype=int)

        MCC = matthews_corrcoef(y_true_np, preds)
        print('MCC: ', MCC)

        report = classification_report(
            y_true_np, preds, target_names=label_names)
        print(report)

        # Saving the prediction results for each test data
        if not os.path.exists(f'{result_path}/model_eval_result.txt'):
            header = "target_id\tlabel\tprediction\n"
            with open(f'{result_path}/model_eval_result.txt', 'a') as file_writer:
                file_writer.write(header)

        for ids, label, pred_value in zip(record_id, y_true_np, preds):
            with open(f'{result_path}/model_eval_result.txt', 'a+') as f:
                f.write(f'{ids}\t{label}\t{pred_value}\n')

        with open(f'{result_path}/model_performance.txt', 'a') as file_writer:
            file_writer.write(f'MCC: {MCC}\nroc_auc_score: {auc_value}\n')

    else:

        preds = np.array(preds >= 0.2, dtype=int)

        if not result_file.exists():
            header = "target_id\tprediction\n"
            with open(output_name, 'a') as file_writer:
                file_writer.write(header)

        for ids, pred_value in zip(record_id, preds):
            with open(result_file, 'a+') as f:
                f.write(f'{ids}\t{pred_value}\n')
