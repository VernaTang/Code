
import warnings
from datetime import datetime
import time 
import torch
import os
from transformers import BertModel,BertConfig,BertModel,BertTokenizer,get_cosine_schedule_with_warmup,BertForMaskedLM,set_seed
import pandas  as pd
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score

from models.bertmask import Bert_Model

from dataset.dataloader import load_csvdata,MyDataSet,ProcessData_meld_plus_temps,MASK_POS
import argparse
# hyperparameters

TRAIN_BATCH_SIZE=32
TEST_BATCH_SIZE=96
EVAL_PERIOD=20
MODEL_NAME="bert-large-uncased"
DATA_PATH="./dataset-process/MELD/"
NUM_WORKERS=10

# train_file="train_mix_sentence.csv"
# dev_file="dev_mix_sentence.csv"
test_file="test_mix_sentence.csv"


# env variables

#os.environ['TOKENIZERS_PARALLELISM']="false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
torch.cuda.set_device(0)
d_cnt = torch.cuda.device_count()
current_device = torch.cuda.current_device()
print("===============The device is {}, and number is {}, current device is {}================".format(device, d_cnt, current_device))


'''

'''

pd.options.display.max_columns = None
pd.options.display.max_rows = None


# get the data and label
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=16)
parser.add_argument('--ckpt', type=str, default=None)
#1706286217.6496248_3_16_4_f1score:0.9090909090909091.pth

args = parser.parse_args()
ckpt_name = args.ckpt
ckpt_arr = ckpt_name.split("_")
mapping_id = ckpt_arr[-2]
print("Mapping_id is {}".format(mapping_id))
if int(mapping_id) < 20:
    mapping_file = open("./Temps.txt")
    mapping_id = int(mapping_id)
    line = mapping_file.readlines()
    mapping_words = line[mapping_id - 1]
elif int(mapping_id) == 1001:
    mapping_words = '{"neutral": "neutral", "positive": "positive", "negative": "negative"}'
else:
    mapping_file = open("./M_Temps.txt")
    mapping_id = int(mapping_id) - 20
    line = mapping_file.readlines()
    mapping_words = line[mapping_id - 1]

print("mapping words is {}".format(mapping_words))

SEED = args.seed
set_seed(SEED)

# DATA_PATH+os.sep+filepath

tokenizer=BertTokenizer.from_pretrained(MODEL_NAME)

#Inputid_train,Labelid_train,typeids_train,inputnmask_train=ProcessData_meld_plus_temps(DATA_PATH+os.sep+train_file,tokenizer,mapping_words)
#Inputid_dev,Labelid_dev,typeids_dev,inputnmask_dev=ProcessData_meld_plus_temps(DATA_PATH+os.sep+dev_file,tokenizer,mapping_words)
Inputid_test,Labelid_test,typeids_test,inputnmask_test=ProcessData_meld_plus_temps(DATA_PATH+os.sep+test_file,tokenizer,mapping_words)


#train_dataset = Data.DataLoader(MyDataSet(Inputid_train,  inputnmask_train , typeids_train , Labelid_train), TRAIN_BATCH_SIZE, shuffle=True,num_workers=NUM_WORKERS)
#valid_dataset = Data.DataLoader(MyDataSet(Inputid_dev,  inputnmask_dev , typeids_dev , Labelid_dev), TRAIN_BATCH_SIZE,  shuffle=True,num_workers=NUM_WORKERS)
test_dataset = Data.DataLoader(MyDataSet(Inputid_test,  inputnmask_test , typeids_test , Labelid_test), TEST_BATCH_SIZE,  shuffle=True,num_workers=NUM_WORKERS)

#train_data_num=len(Inputid_train)
#dev_data_num=len(Inputid_dev)
test_data_num=len(Inputid_test)
print("==========================")
print("The length of test_dataset is {}".format(test_data_num))
print("==========================")

totaltime=0

print("==============Start testing=============")
model_path = os.path.join("./model_ckpt", ckpt_name)
ckpt = torch.load(model_path, map_location="cpu")
print("========Model has been loaded!!========")
config=BertConfig.from_pretrained(MODEL_NAME)
config.hidden_dropout_prob = 0.3
config.attention_probs_dropout_prob = 0.3

model=Bert_Model(bert_path=MODEL_NAME,config_file=config).to(device)
model.load_state_dict(ckpt['model'])
#print("========执行到这一步了========")
correct_test = 0

with torch.no_grad():
    for ids, att, tpe, y in test_dataset:
        # if epoch == 0 and idx == 1:
        #     print("=============testdata_1===============")
        #     print(ids, att, tpe, y)
        ids, att, tpe, y = ids.to(device), att.to(device), tpe.to(device), y.to(device)

        out_test = model(ids , att , tpe)
        ttruelabel = y[:, MASK_POS]
        tout_train_mask = out_test[:, MASK_POS, :]
        predicted_test = torch.max(tout_train_mask.data, 1)[1]
        correct_test += (predicted_test == ttruelabel).sum()
        correct_test = float(correct_test)
    acc_test = float(correct_test / test_data_num)
    list_true = ttruelabel.tolist()
    list_predict = predicted_test.tolist()
    f1score_test = f1_score(list_true, list_predict, average='weighted')


    out = ("acc_test {} ,f1score_test{}".format(acc_test, f1score_test))
    print(out)

