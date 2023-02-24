import random

from statistics import mean

import torch
import torch.nn as nn

import pytorch_lightning as pl

from transformers import BertModel
from sklearn.metrics import classification_report

from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, PrecisionRecallCurve

from sklearn.metrics import f1_score

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

        self.train_f1_score = []
        self.train_loss_score = []

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

        # pred = out
        # true = y

        # acc = self.accuracy(out, y)
        
        # precission, recall, _ = self.precission_recall(out, y)
        # report = classification_report(true, pred, output_dict = True, zero_division = 0)

        # self.log("accuracy", acc, prog_bar = True)
        self.log("f1_score", f1_s, prog_bar = True)
        self.log("loss", loss)

        # return {"loss": loss, "predictions": out, "F1": f1_score, "labels": y}
        return {"loss": loss, "F1": f1_s, "labels": y}

    def validation_step(self, valid_batch, batch_idx):
        x_input_ids, y = valid_batch
        
        out = self(input_ids = x_input_ids)
        ## ke tiga parameter di input dan diolah oleh method / function forward

        loss = self.criterion(out, target = y.float())

        # pred = out
        # true = y

        # report = classification_report(true, pred, output_dict = True, zero_division = 0)
        # acc = self.accuracy(out, y)
        # f1_score = self.f1(out, y)
        
        # self.log("f1_score", f1_score, prog_bar = True)
        # self.log("accuracy", acc, prog_bar = True)
        self.log("loss", loss)

        return loss
    
    def predict_step(self, pred_batch, batch_idx):
        x_input_ids, y = pred_batch
        
        out = self(input_ids = x_input_ids)
        ## ke tiga parameter di input dan diolah oleh method / function forward
        pred = out
        true = y

        return {"predictions": pred, "labels": true}

    def training_epoch_end(self, outputs):
        f1_scores = []
        loss_scores = []

        for output in outputs:
            f1_scores.append(output['F1'])
            loss_scores.append(output['loss'].detach().cpu().item())
            #.cpu().detach().numpy()

        self.train_f1_score.append(mean(f1_scores))
        self.train_loss_score.append(mean(loss_scores))
        ## Jumlah epoch
        # epochs = range(len(f1_scores))

        f1_fig, f1_ax = plt.subplots()
        f1_ax.set_xlabel('epoch')
        f1_ax.set_ylabel('f1 micro')
        f1_ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        f1_ax.plot(self.train_f1_score, marker="o", mfc='green', mec='yellow', ms='7')

        for x_epoch, y_f1_sc in enumerate(self.train_f1_score):
            f1_sc_lbl = "{:.2f}".format(y_f1_sc)

        ## {:.2f} = 2 decimal places

            f1_ax.annotate(f1_sc_lbl, 
                           (x_epoch, y_f1_sc),
                           textcoords="offset points",
                           xytext=(0,9),
                           ha='center',
                           arrowprops=dict(arrowstyle="->", color='black'))
        
        f1_fig.savefig("train_f1_scores.png")
        
        loss_fig, loss_ax = plt.subplots()
        
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        loss_ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        loss_ax.plot(self.train_loss_score, color='r', marker="o", mfc='yellow', mec='blue', ms='7')

        plt.title('F1-Score and Loss of Training')

        for x_epoch, y_loss_sc in enumerate(self.train_loss_score):
            y_loss_lbl = "{:.2f}".format(y_loss_sc)
            
            loss_ax.annotate(y_loss_lbl, 
                             (x_epoch, y_loss_sc),
                             textcoords="offset points",
                             xytext=(0,9),
                             ha='center',
                             arrowprops=dict(arrowstyle="->", color='black'))
            loss_ax.plot(x_epoch, y_loss_sc, color="black", linewidth=1)

        loss_fig.savefig("train_loss_scores.png")
        # plt.plot(f1_scores, 'b', label='F1 Score')
        # plt.plot(loss_scores, 'r', label='Loss')
        # plt.xlabel('Epoch(Number of Sentences)')
        # plt.ylabel('F1 Score')

        ## Put a legend to the right of the current axis
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ##plt.legend(bbox_to_anchor=(1.1, 1.05))
        ##plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
        ## Put a legend below current axis
        ##plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
        ##https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot

        # plt.title('Training F1 Score and Loss')
        # plt.show()
        # plt.savefig('f1_score.png') # menyimpan gambar

        # labels = []
        # predictions = []

        # for output in outputs:
        #     for out_lbl in output["labels"].detach().cpu():
        #         labels.append(out_lbl)
        #     for out_pred in output["predictions"].detach().cpu():
        #         predictions.append(out_pred)

        # labels = torch.stack(labels).int()
        # predictions = torch.stack(predictions)

        ## Hitung akurasi
        
        # # accuracy = Accuracy(task = "multiclass", num_classes = self.num_classes)
        # acc = self.accuracy(predictions, labels)
        # f1_score = self.f1(predictions, labels)
        ## Print Akurasinya
        # print("Overall Training Accuracy : ", acc , "| F1 Score : ", f1_score)

    # def on_predict_epoch_end(self, outputs):
    #     labels = []
    #     predictions = []

    #     for output in outputs:
    #         # print(output[0]["predictions"][0])
    #         # print(len(output))
    #         # break
    #         for out in output:
    #             for out_lbl in out["labels"].detach().cpu():
    #                 labels.append(out_lbl)
    #             for out_pred in out["predictions"].detach().cpu():
    #                 predictions.append(out_pred)

    #     labels = torch.stack(labels).int()
    #     predictions = torch.stack(predictions)
        
    #     acc = self.accuracy(predictions, labels)
    #     f1_score = self.f1(predictions, labels)
    #     print("Overall Testing Accuracy : ", acc , "| F1 Score : ", f1_score)