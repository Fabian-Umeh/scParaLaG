"""
----------------------------------------------------------------------------------------------------------------------------------------
Description:                                                                                                                            |
    This module is used for the custom training of scParaLaG models.                                                                    |
                                                                                                                                        |
Copyright:                                                                                                                              |
    Copyright © 2024. All rights reserved.                                                                                              |
                                                                                                                                        |
License:                                                                                                                                |
    This script is licensed under the MIT License.                                                                                      |
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software                                       |
    and associated documentation files (the "Software"), to deal in the Software without restriction,                                   |
    including without limitation the rights to use, copy, modify, merge, publish, distribute,                                           |
    sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,                   |
    subject to the following conditions:                                                                                                |
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.      |
                                                                                                                                        |
Disclaimer:                                                                                                                             |
    This software is provided 'as is' and without any express or implied warranties.                                                    |
    The author or the copyright holders make no representations about the suitability of this software for any purpose.                 |
                                                                                                                                        |
Contact:                                                                                                                                |
    For any queries or issues related to this script, please contact fchumeh@gmail.com.                                                 |
----------------------------------------------------------------------------------------------------------------------------------------
"""
import sys
import os
import math
import warnings
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.scParaLaG import scParaLaG, scParaLaG_Hierarchical
from scipy.stats import pearsonr, spearmanr
from decimal import Decimal
from sklearn.metrics import mean_absolute_error
import gc

# Ignore all warnings
warnings.filterwarnings("ignore")

class CustomEarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, rate_threshold=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.rate_threshold = rate_threshold
        self.best_loss = float('inf')
        self.best_epoch = -1
        self.epochs_since_improvement = 0
        self.prev_loss = float('inf')
    def update(self, current_loss, current_epoch):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.best_epoch = current_epoch
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1
        rate_of_improvement = (self.prev_loss - current_loss) / self.prev_loss if self.prev_loss != 0 else 0
        if rate_of_improvement < self.rate_threshold:
            self.epochs_since_improvement += 1
        print(f'Patience grace period: {self.patience - self.epochs_since_improvement}')
        self.prev_loss = current_loss
    def should_stop(self):
        return self.epochs_since_improvement >= self.patience

# Unified wrapper that selects a model variant based on args.variant.
class scParaLaGWrapper:
    def __init__(self, args):
        self.args = args
        if hasattr(self.args, 'variant'):
            variant = self.args.variant
        else:
            variant = "original"
        if variant == "original":
            ModelClass = scParaLaG
        elif variant == "hierarchical":
            ModelClass = scParaLaG_Hierarchical
        else:
            raise ValueError(f"Unknown variant: {variant}")
        self.model = ModelClass(self.args).to(self.args.device)

    def predict(self, graph, idx=None):
        self.model.eval()
        with torch.no_grad():
            if idx is not None:
                graph = graph.subgraph(idx).to(self.args.device)
            else:
                graph = graph.to(self.args.device)
            outputs = self.model(graph, graph.ndata['feat'])
        return outputs

    def score(self, graph, labels, idx=None):
        self.model.eval()
        with torch.no_grad():
            preds = self.predict(graph, idx)
            if idx is not None:
                preds = preds[idx]
                labels = labels[idx]
            mse_loss = F.mse_loss(preds, labels.to(self.args.device).float())
            rmse = math.sqrt(mse_loss.item())
            mae = mean_absolute_error(labels.cpu().numpy(), preds.cpu().numpy())
            return rmse, mae

    def fit(self, train_graph, val_graph, test_graph, train_label, val_label,
            test_label, num_epochs=500, batch_size=520, verbose=True,
            es_patience=20, es_min_delta=0.01, es_rate_threshold=0.001,
            learning_rate=0.000028, sample=True, l2_lambda=0.0):
        torch.manual_seed(self.args.seed)
        if self.args.device == "cuda":
            torch.cuda.manual_seed_all(self.args.seed)
        early_stopping = CustomEarlyStopping(patience=es_patience,
                                            min_delta=es_min_delta,
                                            rate_threshold=es_rate_threshold)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
        save_dir = "model_checkpoints"
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        num_nodes = len(train_graph.ndata['feat'])
        indices = torch.randperm(num_nodes) if sample else torch.arange(num_nodes)
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            for i in range(0, num_nodes, batch_size):
                batch_indices = indices[i:i+batch_size]
                subgraph = train_graph.subgraph(batch_indices).to(self.args.device)
                batch_labels = train_label[batch_indices].to(self.args.device).float()
                optimizer.zero_grad()
                outputs = self.model(subgraph, subgraph.ndata['feat'])
                loss = F.mse_loss(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = math.sqrt(epoch_loss / (len(indices)/batch_size))
            val_rmse, val_mae = self.score(val_graph, val_label.float())
            test_rmse, test_mae = self.score(test_graph, test_label.float())
            if verbose:
                print('---------------------------------')
                print(f'Epoch {epoch+1}/{num_epochs}')
                print(f'Train RMSE: {avg_loss:.4f}')
                print(f'Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}')
                print(f'Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}')
                print('---------------------------------')
            if val_rmse < best_val_loss:
                best_val_loss = val_rmse
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                print(f'Model saved at epoch {epoch+1} with val RMSE {val_rmse:.4f}')
            early_stopping.update(val_rmse, epoch)
            if early_stopping.should_stop():
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        self.model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
        self.model.eval()
        final_test_rmse, final_test_mae = self.score(test_graph, test_label.float())
        print(f'Final Test RMSE: {final_test_rmse:.4f}, Final Test MAE: {final_test_mae:.4f}')
        # Compute correlation.
        test_graph = test_graph.to(self.args.device)
        test_feat = test_graph.ndata['feat'].to(self.args.device)
        test_label = test_label.to(self.args.device)
        outputs = self.model(test_graph, test_feat).to(self.args.device)
        rmse = math.sqrt(F.mse_loss(outputs, test_label))
        pred_cpu = outputs.cpu().detach().numpy()
        test_cpu = test_label.cpu().detach().numpy()
        pearson_corr, p_value_pearson = pearsonr(pred_cpu.flatten(), test_cpu.flatten())
        spearman_corr, p_value_spearman = spearmanr(pred_cpu.flatten(), test_cpu.flatten())
        p_value_pearson = Decimal(str(p_value_pearson)).quantize(Decimal('1.0000e+0')) if not math.isnan(p_value_pearson) else "NaN"
        p_value_spearman = Decimal(str(p_value_spearman)).quantize(Decimal('1.0000e+0')) if not math.isnan(p_value_spearman) else "NaN"
        print('Pearson Corr:', pearson_corr)
        print('Spearman Corr:', spearman_corr)
        print('p-rmse:', rmse)
        result = pd.DataFrame({
            'rmse': [rmse],
            'mae': [final_test_mae],
            'seed': [self.args.seed],
            'subtask': [self.args.subtask],
            'method': [self.args.variant if hasattr(self.args, 'variant') else 'original'],
            'pearson': [pearson_corr],
            'p_value_pearson': [p_value_pearson],
            'spearman': [spearman_corr],
            'p_value_spearman': [p_value_spearman]
        })
        print(result)
        # Clean up.
        del train_graph, val_graph, test_graph
        del train_label, val_label, test_label
        del outputs, test_feat, pred_cpu, test_cpu
        if self.args.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()