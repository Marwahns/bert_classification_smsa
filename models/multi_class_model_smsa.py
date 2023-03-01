import random
import pandas as pd

from statistics import mean

import torch
import torch.nn as nn

import pytorch_lightning as pl

from transformers import BertModel
from sklearn.metrics import classification_report

from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, PrecisionRecallCurve

from sklearn.metrics import f1_score, accuracy_score

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

class MultiClassModel(pl.LightningModule):
    def __init__(self,
                 dropout,
                 n_out,
                 lr) -> None:
        super(MultiClassModel, self).__init__()

        torch.manual_seed(1)
        random.seed(1)

        self.num_classes = n_out

        self.bert = BertModel.from_pretrained('indolem/indobert-base-uncased')
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(dropout)

        ## Jumlah label = 5
        self.classifier = nn.Linear(768, n_out)

        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()

        self.accuracy = MulticlassAccuracy(task="multiclass", num_classes = self.num_classes)

        # self.f1 = MulticlassF1Score(task = "multiclass", 
        #                   average = "micro", 
        #                   multidim_average = "global",
        #                   num_classes = self.num_classes)
        # self.precission_recall = PrecisionRecallCurve(task = "multiclass", num_classes = self.num_classes)

        self.train_score = {
            "loss": [],
            "F1" : [],
            "accuracy": []
        }

        self.validation_score = {
            "loss": [],
            "F1" : [],
            "accuracy": []
        }

    ## Model
    def forward(self, input_ids):
        bert_out = self.bert(input_ids = input_ids)
        hidden_state = bert_out[0]
        pooler = hidden_state[:, 0]
        ## Outout size (batch size = 30 baris, sequence length = 100 kata / token, hidden_size = 768 tensor jumlah vector representation dari bert)

        pooler = self.pre_classifier(pooler)
        ## pre classifier untuk mentransfer weight output ke epoch selanjutnya
        pooler = torch.nn.Tanh()(pooler)
        ## kontrol hasil pooler min -1 max 1


        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        ## classifier untuk memprojeksikan hasil pooler (768) ke jumlah label (5)

        return output

    def configure_optimizers(self):
        ## Proses training lebih cepat
        ## Tidak memakan memori berlebih
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x_input_ids, y = train_batch
        
        out = self(input_ids = x_input_ids)
        ## ke tiga parameter di input dan diolah oleh method / function forward

        loss = self.criterion(out, target = y.float())

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()
        f1_s = f1_score(true, pred, average='micro')

        # predict = out.argmax(1).cpu().detach().numpy()
        avg_pred = sum(pred)/len(pred)
        predict = avg_pred.cpu().detach().numpy()

        # pred = out
        # true = y

        # acc = self.accuracy(out, y)
        acc = accuracy_score(pred, true)
        # precission, recall, _ = self.precission_recall(out, y)
        # report = classification_report(true, pred, output_dict = True, zero_division = 0)

        self.log("accuracy", acc, prog_bar = True)
        self.log("f1_score", f1_s, prog_bar = True)
        self.log("loss", loss)

        # return {"loss": loss, "predictions": out, "F1": f1_score, "labels": y}
        return {"loss": loss, "predictions": out, "F1": f1_s, "labels": y, "avg_pred": predict, "accuracy": acc}

    def validation_step(self, valid_batch, batch_idx):
        x_input_ids, y = valid_batch
        
        out = self(input_ids = x_input_ids)
        ## ke tiga parameter di input dan diolah oleh method / function forward

        loss = self.criterion(out, target = y.float())

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()
        f1_s = f1_score(true, pred, average='micro')

        avg_pred = sum(pred)/len(pred)
        predict = avg_pred.cpu().detach().numpy()

        acc = accuracy_score(pred, true)

        # pred = out
        # true = y

        # report = classification_report(true, pred, output_dict = True, zero_division = 0)
        # acc = self.accuracy(out, y)
        # f1_score = self.f1(out, y)
        
        # self.log("f1_score", f1_score, prog_bar = True)
        self.log("f1_score", f1_s, prog_bar = True)
        self.log("accuracy", acc, prog_bar = True)
        self.log("loss", loss)

        # return loss
        return {"val_loss": loss, "predictions": out, "F1": f1_s, "labels": y, "avg_pred": predict, "accuracy": acc}
    
    def test_step(self, test_batch, batch_idx):
        x_input_ids, y = test_batch
        
        out = self(input_ids = x_input_ids)
        ## ke tiga parameter di input dan diolah oleh method / function forward

        pred = out.argmax(1).cpu()
        true = y.argmax(1).cpu()
        f1_s = f1_score(true, pred, average='micro')

        acc = accuracy_score(pred, true)

        self.log("f1_score", f1_s, prog_bar = True)
        self.log("accuracy", acc, prog_bar = True)

        return {"predictions": pred, "labels": true, "F1": f1_s, "accuracy": acc}
    
    def predict_step(self, pred_batch, batch_idx):
        x_input_ids, y = pred_batch
        
        out = self(input_ids = x_input_ids)
        ## ke tiga parameter di input dan diolah oleh method / function forward
        pred = out
        true = y

        return {"predictions": pred, "labels": true}
    
    def create_figure(self, data, fig_dir, y_label):
        ## F1 Score
        c_fig, c_ax = plt.subplots()
        c_ax.set_xlabel('epoch')
        c_ax.set_ylabel(y_label)
        c_ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        c_ax.plot(data, marker="o", mfc='green', mec='yellow', ms='7')

        ## {:.2f} = 2 decimal places
        for x_epoch, y_sc in enumerate(data):
            y_sc_lbl = "{:.2f}".format(y_sc)

            c_ax.annotate(y_sc_lbl, 
                           (x_epoch, y_sc),
                           textcoords="offset points",
                           xytext=(0,9),
                           ha='center',
                           arrowprops=dict(arrowstyle="->", color='black'))
        
        c_fig.savefig(fig_dir)

    def training_epoch_end(self, outputs):

        scores = {
            "loss": [],
            "F1" : [],
            "accuracy": []
        }

        for output in outputs:
            scores["F1"].append(output['F1'])
            scores["loss"].append(output['loss'].detach().cpu().item())
            scores["accuracy"].append(output['accuracy'])

        self.train_score["F1"].append(mean(scores["F1"]))
        self.train_score["loss"].append(mean(scores["loss"]))
        self.train_score["accuracy"].append(mean(scores["accuracy"]))

        df_scores = pd.DataFrame.from_dict(self.train_score)
        df_scores.to_csv('./images/training_scores.csv')

        self.create_figure(self.train_score["F1"], "./images/train_f1_scores_micro.png", "f1-score micro")
        self.create_figure(self.train_score["loss"], "./images/train_loss_scores.png", "loss")
        self.create_figure(self.train_score["accuracy"], "./images/train_accuracy_scores.png", "accuracy")

    def validation_epoch_end(self, outputs):
        scores = {
            "loss": [],
            "F1" : [],
            "accuracy": []
        }

        for output in outputs:
            scores["F1"].append(output['F1'])
            scores["loss"].append(output['val_loss'].detach().cpu().item())
            scores["accuracy"].append(output['accuracy'])

        self.validation_score["F1"].append(mean(scores["F1"]))
        self.validation_score["loss"].append(mean(scores["loss"]))
        self.validation_score["accuracy"].append(mean(scores["accuracy"]))

        df_scores = pd.DataFrame.from_dict(self.validation_score)
        df_scores.to_csv('./images/validation_scores.csv')

        self.create_figure(self.validation_score["F1"], "./images/validation_f1_scores_micro.png", "f1-score micro")
        self.create_figure(self.validation_score["loss"], "./images/validation_loss_scores.png", "loss")
        self.create_figure(self.validation_score["accuracy"], "./images/validation_accuracy_scores.png", "accuracy")
    
    def test_epoch_end(self, outputs):
        scores = {
            "F1" : [],
            "accuracy": []
        }

        for output in outputs:
            scores["F1"].append(output['F1'])
            scores["accuracy"].append(output['accuracy'])

        print('F1 Score = ', mean(scores["F1"]), ' Accuracy = ', mean(scores["accuracy"]))