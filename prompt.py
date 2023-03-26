import os
from torch.utils.data import Dataset,DataLoader
import torch
from torch import nn
from transformers import AutoTokenizer,AutoModel,BertModel,BertTokenizer
from tqdm import tqdm
def get_data(path,max_len=None,mode='train'):
    with open(os.path.join(path,f'{mode}.txt'),'r',encoding='utf-8') as f:
        all_data = f.read().split('\n')
    with open(os.path.join(path,'index_2_label.txt'),'r',encoding='utf-8') as f:
        index_2_label = f.read().split('\n')
    all_text,all_label = [],[]
    for data in all_data:
        data = data.split('	')
        if len(data)!=2:
            continue
        all_text.append(data[0])
        all_label.append(index_2_label[int(data[1])])

    if max_len is not None:
        return all_text[:max_len],all_label[:max_len]
    return all_text,all_label
class TCdataset(Dataset):
    def __init__(self,all_text,all_label,tokenizers):
        self.all_text = all_text
        self.all_label = all_label
        self.tokenizers = tokenizers
        self.prompt_text1 = "文章内容是:"
        self.prompt_text2 = "。类别是:"
    def __getitem__(self, x):
        text = self.all_text[x]
        label = self.all_label[x]

        text_prompt = self.prompt_text1 + text + self.prompt_text2
        return text_prompt,label,len(text_prompt)+2 #给他两个[MASK],所以加2
    def process_data(self,data):
        batch_text,batch_label,batch_len = zip(*data)
        batch_max = max(batch_len)+2

        batch_text_idx = []
        batch_label_idx = []
        for text,label in zip(batch_text,batch_label):
            text = text+"[MASK]"*2
            text_idx = self.tokenizers.encode(text,add_special_tokens=True)#be+xxx+[MASK][MASK]+ed label[len_]


            label_idx = [-100]*(len(text_idx)-3) + self.tokenizers.encode(label,add_special_tokens=False)

            text_idx += [0]*(batch_max-len(text_idx))
            label_idx += [-100]*(batch_max-len(label_idx))

            assert(len(text_idx)==len(label_idx))

            batch_text_idx.append(text_idx)
            batch_label_idx.append(label_idx)


        return torch.tensor(batch_text_idx),torch.tensor(batch_label_idx)



    def __len__(self):
        return len(self.all_text)
class Bert_Model(nn.Module):
    def __init__(self,model_name):
        super().__init__()
        self.backbone = BertModel.from_pretrained(model_name)
        self.generater = nn.Linear(768,21128)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self,x,label=None):
        x,_ = self.backbone(x,return_dict=False,attention_mask = ( (x!=103) & (x!=0) ))

        x = self.generater(x)

        if label is not None:
            loss = self.loss_fn(x.reshape(-1,x.shape[-1]),label.reshape(-1))
            return loss
        else:
            return torch.argmax(x,dim=-1)

if __name__ == "__main__":
    train_text,train_label = get_data(os.path.join('data'))
    dev_text,dev_label = get_data(os.path.join('data'),mode='test')

    batch_size = 100
    epoch = 10
    lr = 1e-5
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    model_name = "./bert_base_chinese"#hfl/chinese-roberta-wwm-ext

    tokenizers = BertTokenizer.from_pretrained(model_name)

    train_dataset = TCdataset(train_text,train_label,tokenizers)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=train_dataset.process_data)
    dev_dataset = TCdataset(dev_text, dev_label, tokenizers)
    dev_dataloader = DataLoader(dev_dataset, batch_size=20, shuffle=False,
                                  collate_fn=dev_dataset.process_data)

    model = Bert_Model(model_name).to(device)
    opt = torch.optim.Adam(model.parameters(),lr = lr)
    print(f'train on {device}.....')
    for e in range(epoch):
        loss_sum = 0
        ba = 0
        for x,y in tqdm(train_dataloader):
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            loss = model(x,y)
            loss.backward()
            opt.step()

            loss_sum+=loss
            ba += 1
        print(f'e = {e} loss = {loss_sum / ba:.6f}')
        right = 0
        for x, y in tqdm(dev_dataloader):
            x = x.to(device)
            y = y.to(device)

            pre = model(x)

            for p,label in zip(pre,y):
                right += int((p[label!=-100]==label[label!=-100]).all())
        print(f'acc={right/len(dev_dataset):.5f}')



