import torch
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from unimodals.common_models import GRU, MLP, Sequential, Identity  # noqa
from training_structures.Supervised_Learning import train, test  # noqa
from datasets.affect.get_data import get_dataloader  # noqa
from fusions.common_fusions import ConcatEarly  # noqa
#from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test # noqa
from memory_profiler import memory_usage # noqa
from models.eval_scripts.complexity import  all_in_one_train, all_in_one_test

if __name__ == '__main__':

    print(torch.cuda.is_available())

    print(torch.cuda.device_count())

    traindata, validdata, testdata = get_dataloader('C:/Users/29296/Documents/Tencent Files/2929629852/FileRecv/sarcasm.pkl', robust_test=False, max_pad=True,  data_type='sarcasm', max_seq_len=40)

    encoders = [Identity().cuda(),Identity().cuda(),Identity().cuda()]
    head = Sequential(GRU(752, 1128, dropout=True, has_padding=False, batch_first=True, last_only=True), MLP(1128, 512, 1)).cuda()

    fusion = ConcatEarly().cuda()

    allmodules = [encoders[0], encoders[1], encoders[2], head, fusion]

    def train_process():
        train(encoders, fusion, head, traindata, validdata, 4, task="regression", optimtype=torch.optim.AdamW,
            is_packed=False, lr=1e-3, save='sarcasm_temp.pt', weight_decay=0.01, objective=torch.nn.L1Loss())


    all_in_one_train(train_process, allmodules)

    print("Testing:")
    model = torch.load('sarcasm_temp.pt').cuda()
    test(model, testdata, 'affect', is_packed=False,
        criterion=torch.nn.L1Loss(), task="posneg-classification", no_robust=True)
