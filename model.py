# coding=utf8
# 你配不上自己的野心，也就辜负了先前的苦难
# 整段注释Control+/
import numpy as np
import torch
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block
from torch import nn


def random_indexes(size:int):
    forward_indexes=np.arange(size)
    #np.arange(size) 创建一个从 0 到 size-1 的连续整数数组。这个数组代表了原始的顺序。
    np.random.shuffle(forward_indexes)
    backward_indexes=np.argsort(forward_indexes)
    #使用 np.argsort 函数对打乱后的 forward_indexes 进行排序，得到 backward_indexes。这个数组记录了打乱顺序中的每个元素在原始顺序中的位置。
    return forward_indexes,backward_indexes

#根据提供的索引 indexes 从序列 sequences 中抽取对应的元素。
def take_indexes(sequences,indexes):
    return torch.gather(input=sequences,dim=0,index=repeat(indexes,'num_patch B->num_patch B Channel',Channel=sequences.shape[-1]))

class PatchShuffle(nn.Module):
    def __init__(self,ratio)->None:
        super(PatchShuffle, self).__init__()
        self.ratio=ratio

    def forward(self,patches:torch.Tensor):
        num_patch,B,Channel=patches.shape
        remain_num_patch=int(num_patch*(1-self.ratio))

        indexes=[random_indexes(size=num_patch) for _ in range(B)]
        forward_indexes=torch.as_tensor(data=np.stack([i[0] for i in indexes],axis=-1),dtype=torch.long).to(patches.device)
        backward_indexes=torch.as_tensor(np.stack([i[1] for i in indexes],axis=-1),dtype=torch.long).to(patches.device)
        #用于将 indexes 列表中的随机索引和逆序索引转换为 PyTorch 张量，并确保这些张量与输入的 patches 张量位于同一个设备上（例如 CPU 或 GPU）
        patches=take_indexes(sequences=patches,indexes=forward_indexes)
        patches=patches[:remain_num_patch]
        return patches,forward_indexes,backward_indexes

class MAE_encoder(nn.Module):
    def __init__(self,image_size=32,patch_size=2,emb_dim=192,num_layer=12,num_head=3,mask_ratio=0.75)->None:
        super(MAE_encoder, self).__init__()

        self.cls_token=torch.nn.Parameter(torch.zeros(1,1,emb_dim))
        self.position_embedding=torch.nn.Parameter(torch.zeros((image_size//patch_size)**2,1,emb_dim))
        self.shuffle=PatchShuffle(ratio=mask_ratio)
        self.patch_conv1=nn.Conv2d(in_channels=3,out_channels=emb_dim,kernel_size=patch_size,stride=patch_size)
        self.transformer=torch.nn.Sequential(*[Block(dim=emb_dim,num_heads=num_head) for _ in range(num_layer)])
        self.layer_norm=nn.LayerNorm(emb_dim)
        self.init_weight()

    def init_weight(self):
        #它将参数按照截断正态分布进行初始化。截断正态分布是一种正态分布，其中以一定的概率截断超出某个范围的值，
        #这样可以避免极端值的出现，有助于模型的稳定训练。
        trunc_normal_(tensor=self.cls_token,std=.02)
        trunc_normal_(tensor=self.position_embedding,std=.02)

    def forward(self,imgs):
        patches=self.patch_conv1(imgs)
        patches=rearrange(patches,'b c h w->(h w) b c')
        patches=patches+self.position_embedding
        patches,forward_indexes,backward_indexes=self.shuffle(patches)
        patches=torch.cat([self.cls_token.expand(-1,patches.shape[1],-1),patches],dim=0)
        #将 CLS 标记扩展并与 patches 连接，形成一个新的序列，其中第一个元素是 CLS 标记。
        patches=rearrange(patches,'num_patch B Channel->B num_patch Channel')
        features=self.layer_norm(self.transformer(patches))
        features=rearrange(features,'B num_patch Channel->num_patch B Channel')
        return features,backward_indexes

class MAE_Decoder(nn.Module):
    def __init__(self,img_size=32,emb_dim=192,patch_size=2,num_layer=4,num_head=3)->None:
        super(MAE_Decoder, self).__init__()

        self.mask_token=nn.Parameter(torch.zeros(1,1,emb_dim))
        self.pos_embedding=nn.Parameter(torch.zeros((img_size//patch_size)**2+1,1,emb_dim))
        #留一个位置给mask_token
        self.transformer=nn.Sequential(*[Block(dim=emb_dim,num_heads=num_head) for _ in range(num_layer)])
        self.head=nn.Linear(in_features=emb_dim,out_features=3*patch_size**2)
        self.patch2img=Rearrange('(h w) b (c p1 p2)->b c (h p1) (w p2) ',p1=patch_size,p2=patch_size,h=img_size//patch_size)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(tensor=self.mask_token,std=.02)
        trunc_normal_(tensor=self.pos_embedding,std=.02)

    def forward(self,features,backward_indexes):
        num_patches=features.shape[0]
        backward_indexes=torch.cat([torch.zeros(1,backward_indexes.shape[1]).to(backward_indexes),backward_indexes+1],dim=0)
        #在 backward_indexes 前面添加一个全零的行，并将所有索引加 1。
        # 这是因为在 features 中，第一个位置被用于遮蔽标记（mask token），所以原始的 backward_indexes 需要偏移。
        features=torch.cat([features,self.mask_token.expand(backward_indexes.shape[0]-features.shape[0],features.shape[1],-1)],dim=0)
        #将 self.mask_token 扩展并添加到 features 的末尾，以替换那些被遮蔽的 patches。
        # 扩展的数量由 backward_indexes 和 features 的形状差异决定。
        features=take_indexes(sequences=features,indexes=backward_indexes)
        features=features+self.pos_embedding
        features=rearrange(features,'num_patch B Channel->B num_patch Channel')
        features=self.transformer(features)
        features=rearrange(features,'B num_patch Channel->num_patch B Channel')
        features=features[1:]   # remove global feature
        patches=self.head(features)
        mask=torch.zeros_like(patches)
        mask[num_patches-1:]=1
        # 这行代码的目的是在一个全零的掩码中标记最后一个位置（即遮蔽标记的位置），以便在计算损失时忽略这个位置。
        # 这样做可以确保模型只对可见的（未被遮蔽的）patches 进行重建的优化，而忽略那些在训练过程中被随机遮蔽的部分。
        mask=take_indexes(sequences=mask,indexes=backward_indexes[1:]-1)
        img=self.patch2img(patches)
        mask=self.patch2img(mask)
        return img,mask

class MAE_ViT(nn.Module):
    def __init__(self,image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75)->None:
        super(MAE_ViT, self).__init__()
        self.encoder=MAE_encoder(image_size, patch_size, emb_dim, num_layer=encoder_layer,num_head=encoder_head,mask_ratio=mask_ratio)
        self.decoder=MAE_Decoder(image_size,emb_dim,patch_size,num_layer=decoder_layer,num_head=decoder_head)

    def forward(self,img):
        features,backward_indexes=self.encoder(img)
        predicted_img,mask=self.decoder(features,backward_indexes)
        return predicted_img,mask

class ViT_Classifier(nn.Module):
    def __init__(self,encoder:MAE_encoder,num_classes=10)->None:
        super(ViT_Classifier, self).__init__()
        self.cls_token=encoder.cls_token
        self.pos_embedding=encoder.position_embedding
        self.patch_proj=encoder.patch_conv1
        self.transformer=encoder.transformer
        self.layernorm=encoder.layer_norm
        self.head=nn.Linear(in_features=self.pos_embedding.shape[-1],out_features=num_classes)

    def forward(self,img):
        patches=self.patch_proj(img)
        patches=rearrange(patches,'b c h w->(h w) b c')
        patches=patches+self.pos_embedding
        patches=torch.cat([self.cls_token.expand(-1,patches.shape[1],-1),patches],dim=0)
        patches=rearrange(patches,'num_patch B Channel->B num_patch Channel')
        features=self.layernorm(self.transformer(patches))
        features=rearrange(features,'B num_patch Channel->num_patch B Channel')
        logits=self.head(features[0])
        return logits













