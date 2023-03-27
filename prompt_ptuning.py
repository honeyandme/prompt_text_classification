import os
from torch.utils.data import Dataset,DataLoader
import torch
from torch import nn
from transformers import AutoTokenizer,AutoModel,BertModel,BertTokenizer
from tqdm import tqdm
import copy
def get_data(path,max_len=None,mode='train'):
    with open(os.path.join(path,f'{mode}.txt'),'r',encoding='utf-8') as f:
        all_data = f.read().split('\n')

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
    def __init__(self,all_text,all_label,tokenizers,mask_pos,prompt_len):
        self.all_text = all_text
        self.all_label = all_label
        self.tokenizers = tokenizers
        self.mask_pos = mask_pos
        self.prompt_len = prompt_len
        self.prompt_template = self.generate_template()
    def generate_template(self):
        temp = ""
        for i in range(prompt_len):
            if i in self.mask_pos:
                temp += "[MASK]"
            temp += f"[prompt_{i}]"
        return temp
    def __getitem__(self, x):
        text = self.all_text[x]
        label = self.all_label[x]


        text_prompt =self.prompt_template+text
        # text_prompt = self.prompt_text1 + text + self.prompt_text2
        return text_prompt,label,self.prompt_len+len(text)+2#给他两个[MASK],所以加2
    def process_data(self,data):
        batch_text,batch_label,batch_len = zip(*data)
        batch_max = max(batch_len)+2

        batch_text_idx = []
        batch_label_idx = []
        for text,label in zip(batch_text,batch_label):
            text_idx = self.tokenizers.encode(text,add_special_tokens=True)

            lt = self.tokenizers.encode(label, add_special_tokens=False)

            label_idx = [-100] * len(text_idx)
            label_idx[self.mask_pos[0]+2] = lt[0]
            label_idx[self.mask_pos[1] + 2] = lt[1]
            text_idx = text_idx + [0] * (batch_max - len(text_idx))
            label_idx = label_idx + [-100] * (batch_max - len(label_idx))

            assert len(text_idx) == len(label_idx)

            batch_text_idx.append(text_idx)
            batch_label_idx.append(label_idx)
            # label_idx = torch.tensor([-100]*len(text_idx))
            #
            # label_idx[text_idx==103]=self.tokenizers.encode(label,add_special_tokens=False,return_tensors='pt')[0]
            #
            #
            # text_idx = torch.cat((text_idx,torch.tensor([0]*(batch_max-len(text_idx)))))
            # label_idx = torch.cat((label_idx,torch.tensor([-100]*(batch_max-len(label_idx)))))
            #
            # assert text_idx.shape==label_idx.shape
            #
            # batch_text_idx.append(text_idx)
            # batch_label_idx.append(label_idx)
        return torch.tensor(batch_text_idx),torch.tensor(batch_label_idx),batch_label



    def __len__(self):
        return len(self.all_text)
class Promot_Embedding(nn.Module):
    def __init__(self,prompt_len=100,embedding_num=768):
        super().__init__()
        self.embedding = nn.Embedding(prompt_len,embedding_num)
        self.linear = nn.Sequential(
            nn.Linear(embedding_num,embedding_num),
            nn.GELU()
        )
    def forward(self,x):
        x = self.embedding(x)
        x = self.linear(x)
        return x
class Bert_Model(nn.Module):
    def __init__(self,model_name):
        super().__init__()
        self.bert1 = BertModel.from_pretrained(model_name)
        for name,param in self.bert1.named_parameters():
            param.requires_grad = False
        self.prompt_embedding = Promot_Embedding()
        self.bert_embedding = self.bert1.get_input_embeddings()

        self.generater = nn.Linear(768,21128)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self,x,label=None):
        x_mask = copy.deepcopy(x)
        copy_x = copy.deepcopy(x)
        prompt_position = x>=21128
        x[prompt_position] = 0#正常操作
        copy_x[copy_x<21128] = 0#使用Promot_Embedding

        x = self.bert_embedding(x)

        copy_x = self.prompt_embedding(copy_x[copy_x>0]-21127)
        x[prompt_position] = copy_x
        x,_ = self.bert1(inputs_embeds=x,return_dict=False,attention_mask = ( (x_mask!=103) & (x_mask!=0) ))

        x = self.generater(x)

        if label is not None:
            loss = self.loss_fn(x.reshape(-1,x.shape[-1]),label.reshape(-1))
            return loss
        else:
            return x
def add_prompt_len(tokenizer,prompt_len):#将prompt字符添加到bert_tokenizer中
    prompt_tokens = [f'[prompt_{i}]' for i in range(prompt_len)]
    tokenizer.add_special_tokens({"additional_special_tokens": prompt_tokens})

if __name__ == "__main__":
    with open(os.path.join('data','index_2_label.txt'),'r',encoding='utf-8') as f:
        index_2_label = f.read().split('\n')
    dict_ver = [[6568, 5307], [2791, 772], [5500, 4873], [3136, 5509], [4906, 2825], [4852, 833], [3198, 3124],
                [860, 5509], [3952, 2767], [2031, 727]]
    dict_ver0, dict_ver1 = zip(*dict_ver)
    dict_ver0,dict_ver1 = list(dict_ver0),list(dict_ver1)

    train_text,train_label = get_data(os.path.join('data'),max_len=5000)
    dev_text,dev_label = get_data(os.path.join('data'),max_len=600,mode='test')

    mask_pos = [3,4]

    batch_size = 40
    epoch = 10
    lr = 1e-3
    prompt_len = 10

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    model_name = "./bert_base_chinese"#hfl/chinese-roberta-wwm-ext

    tokenizers = BertTokenizer.from_pretrained(model_name)
    add_prompt_len(tokenizers, prompt_len)

    train_dataset = TCdataset(train_text,train_label,tokenizers,mask_pos,prompt_len)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=train_dataset.process_data)
    dev_dataset = TCdataset(dev_text, dev_label, tokenizers,mask_pos,prompt_len)
    dev_dataloader = DataLoader(dev_dataset, batch_size=20, shuffle=True,
                                  collate_fn=dev_dataset.process_data)

    model = Bert_Model(model_name).to(device)
    opt = torch.optim.Adam(model.parameters(),lr = lr)
    print(f'train on {device}.....')
    for e in range(epoch):
        loss_sum = 0
        ba = 0
        for x,y,cla in tqdm(train_dataloader):
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            loss = model(x,y)
            loss.backward()
            opt.step()

            loss_sum+=loss
            ba += 1
            # break
        print(f'e = {e} loss = {loss_sum / ba:.6f}')
        right = 0
        for x, y,cla in tqdm(dev_dataloader):
            x = x.to(device)
            y = y.to(device)

            pre = model(x)

            for p,label,cl in zip(pre,y,cla):
                p = p[label != -100]
                w1 = p[0][dict_ver0]
                w2 = p[1][dict_ver1]
                w0 = torch.argmax(w1+w2)

                right += int(index_2_label[w0]==cl)
        print(f'acc={right/len(dev_dataset):.5f}')