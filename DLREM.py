from torch import nn
import torch.nn.functional as F
import torch
torch.backends.cudnn.benchmark=True

gru_layer = 4

class DLREM(nn.Module):
    def __init__(self, num_users, num_items, num_rate_levels, emb_dim, batch_size, device):
        super(DLREM, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_rate_levels = num_rate_levels
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.device = device
        self.user_emb = nn.Embedding(self.num_users, self.emb_dim, padding_idx = 0)
        self.item_emb = nn.Embedding(self.num_items, self.emb_dim, padding_idx = 0)
        self.rate_emb = nn.Embedding(self.num_rate_levels, self.emb_dim, padding_idx = 0)
        self.user_model = _UserModel(self.emb_dim, self.user_emb, self.item_emb, self.rate_emb, self.batch_size, self.device)
        self.item_model = _ItemModel(self.emb_dim, self.user_emb, self.item_emb, self.rate_emb, self.batch_size, self.device)
        self.rate_pred = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2* self.emb_dim, self.emb_dim, bias = True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.emb_dim, self.emb_dim // 4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.emb_dim // 4, 1)
        )

    def forward(self, uids, iids, u_item_pad, u_user_pad, u_user_item_pad, i_user_pad, i_item_pad, i_item_user_pad):
        h = self.user_model(uids, u_item_pad, u_user_pad, u_user_item_pad)
        z = self.item_model(iids, i_user_pad, i_item_pad, i_item_user_pad)
        r_ij = self.rate_pred(torch.cat([h,z], dim = 1))
        return r_ij

class _MultiLayerPercep(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_MultiLayerPercep, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim // 2, output_dim, bias=True),
        )

    def forward(self, x):
        return self.mlp(x)

class _Aggregation(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_Aggregation, self).__init__()
        self.aggre = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.aggre(x)


class _UserModel(nn.Module):
    def __init__(self, emb_dim, user_emb, item_emb, rate_emb, batch_size, device):
        super(_UserModel, self).__init__()
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.rate_emb = rate_emb
        self.emb_dim = emb_dim 
        self.batch_size = batch_size 
        self.device = (torch.device('cpu') if device < 0 else torch.device(f'cuda:{device}'))
        self.w1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w2 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w3 = nn.Linear(self.emb_dim, self.emb_dim)
        self.g_v = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        self.user_items_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_items = _Aggregation(self.emb_dim, self.emb_dim)
        self.user_items_att_s1 = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_items_s1 = _Aggregation(self.emb_dim, self.emb_dim)
        self.user_users_att_s2 = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_neigbors_s2 = _Aggregation(self.emb_dim, self.emb_dim)
        self.gru = nn.GRU(self.emb_dim, self.emb_dim, gru_layer, batch_first=True)
        self.h0 = torch.randn(gru_layer, self.batch_size, self.emb_dim).to(self.device)
        self.combine_mlp = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2 * self.emb_dim, 2 * self.emb_dim, bias = True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2 * self.emb_dim, self.emb_dim, bias = True),
            nn.ReLU()
        )
        self.eps = 1e-10

    def forward(self, uids, u_item_pad, u_user_pad, u_user_item_pad):
        q_j = self.item_emb(u_item_pad[:,:,0])   
        mask_u = torch.where(u_item_pad[:,:,0] > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))   
        e_ij = self.rate_emb(u_item_pad[:, :, 1])  
        x_ij = self.g_v(torch.cat([q_j, e_ij], dim=2).view(-1, 2 * self.emb_dim)).view(q_j.size())  
        p_i = mask_u.unsqueeze(2).expand_as(q_j) * self.user_emb(uids).unsqueeze(1).expand_as(q_j) 
        alpha = self.user_items_att(torch.cat([self.w1(x_ij), self.w1(p_i)], dim = 2).view(-1, 2 * self.emb_dim)).view(mask_u.size())
        alpha = torch.exp(alpha) * mask_u
        alpha = alpha / (torch.sum(alpha, 1).unsqueeze(1).expand_as(alpha) + self.eps)  
        h_iL = self.aggre_items(torch.sum(alpha.unsqueeze(2).expand_as(x_ij) * x_ij, 1))     
        gru_ia, h_n = self.gru(x_ij, self.h0)
        h_iS = gru_ia[:,-1,:] 
        h_iI =  h_iL * h_iS 
        h_iI = F.dropout(h_iI, 0.5, training=self.training) 
        q_j_s = self.item_emb(u_user_item_pad[:,:,:,0])   
        mask_s = torch.where(u_user_item_pad[:,:,:,0] > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))  
        u_user_item_er = self.rate_emb(u_user_item_pad[:, :, :, 1])  
        x_ij_s = self.g_v(torch.cat([q_j_s, u_user_item_er], dim=3).view(-1, 2 * self.emb_dim)).view(q_j_s.size())  
        p_i_s = mask_s.unsqueeze(3).expand_as(q_j_s) * self.user_emb(u_user_pad).unsqueeze(2).expand_as(q_j_s)  
        alpha_s = self.user_items_att_s1(torch.cat([self.w2(x_ij_s), self.w2(p_i_s)], dim = 3).view(-1, 2 * self.emb_dim)).view(mask_s.size())    
        alpha_s = torch.exp(alpha_s) * mask_s
        alpha_s = alpha_s / (torch.sum(alpha_s, 2).unsqueeze(2).expand_as(alpha_s) + self.eps)
        h_oL_temp = torch.sum(alpha_s.unsqueeze(3).expand_as(x_ij_s) * x_ij_s, 2)   
        h_oL = self.aggre_items_s1(h_oL_temp.view(-1, self.emb_dim)).view(h_oL_temp.size())  
        h_oS = []
        for i in range(x_ij_s.shape[1]):
            output_s, h_n = self.gru(x_ij_s[:,i,:,:], self.h0)
            h_oS.append(output_s[:,-1,:])   
        h_oS = torch.stack(h_oS).permute(1, 0, 2) 
        h_oI =  h_oL * h_oS 
        h_oI = F.dropout(h_oI, p=0.5, training=self.training) 
        mask_su = torch.where(u_user_pad > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))
        beta = self.user_users_att_s2(torch.cat([self.w3(h_oI), self.w3(self.user_emb(u_user_pad))], dim = 2).view(-1, 2 * self.emb_dim)).view(u_user_pad.size())
        beta = torch.exp(beta) * mask_su
        beta = beta / (torch.sum(beta, 1).unsqueeze(1).expand_as(beta) + self.eps)
        h_iN = self.aggre_neigbors_s2(torch.sum(beta.unsqueeze(2).expand_as(h_oI) * h_oI, 1))     
        h_iN = F.dropout(h_iN, p=0.5, training=self.training) 
        h =  self.combine_mlp(torch.cat([h_iI, h_iN], dim = 1))  
        return h

class _ItemModel(nn.Module):
    def __init__(self, emb_dim, user_emb, item_emb, rate_emb, batch_size, device):
        super(_ItemModel, self).__init__()
        self.emb_dim = emb_dim
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.rate_emb = rate_emb
        self.batch_size = batch_size
        self.device = (torch.device('cpu') if device < 0 else torch.device(f'cuda:{device}'))
        self.w1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w2 = nn.Linear(self.emb_dim, self.emb_dim)
        self.w3 = nn.Linear(self.emb_dim, self.emb_dim)
        self.g_u = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        self.item_users_att_i = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_users_i = _Aggregation(self.emb_dim, self.emb_dim)
        self.i_friends_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_i_friends = _Aggregation(self.emb_dim, self.emb_dim)
        self.if_friends_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_if_friends = _Aggregation(self.emb_dim, self.emb_dim)
        self.gru = nn.GRU(self.emb_dim, self.emb_dim, gru_layer, batch_first=True)
        self.h0 = torch.randn(gru_layer, self.batch_size, self.emb_dim).to(self.device)
        self.combine_mlp = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2 * self.emb_dim, 2 * self.emb_dim, bias = True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2 * self.emb_dim, self.emb_dim, bias = True),
            nn.ReLU()
        )
        self.eps = 1e-10

    def forward(self, iids, i_user_pad, i_item_pad, i_item_user_pad):
        p_i = self.user_emb(i_user_pad[:,:,0])  
        mask_i = torch.where(i_user_pad[:,:,0] > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))
        e_ij = self.rate_emb(i_user_pad[:,:,1])  
        y_ji = self.g_u(torch.cat([p_i, e_ij], dim = 2).view(-1, 2 * self.emb_dim)).view(p_i.size())  
        q_j = mask_i.unsqueeze(2).expand_as(p_i) * self.item_emb(iids).unsqueeze(1).expand_as(p_i)  
        miu = self.item_users_att_i(torch.cat([self.w1(y_ji), self.w1(q_j)], dim = 2).view(-1, 2 * self.emb_dim)).view(mask_i.size())
        miu = torch.exp(miu) * mask_i
        miu = miu / (torch.sum(miu, 1).unsqueeze(1).expand_as(miu) + self.eps)
        z_jL = self.aggre_users_i(torch.sum(miu.unsqueeze(2).expand_as(y_ji) * self.w1(y_ji), 1))
        gru_jt, h_n = self.gru(y_ji, self.h0)
        z_jS = gru_jt[:,-1,:]       
        z_jA =  z_jL * z_jS
        z_jA = F.dropout(z_jA, p=0.5, training=self.training)
        p_i_s = self.user_emb(i_item_user_pad[:,:,:,0]) 
        mask_s = torch.where(i_item_user_pad[:,:,:,0] > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))  
        q_j_s = mask_s.unsqueeze(3).expand_as(p_i_s) * self.item_emb(i_item_pad).unsqueeze(2).expand_as(p_i_s)  
        i_item_user_er = self.rate_emb(i_item_user_pad[:, :, :, 1]) 
        y_ji_s = self.g_u(torch.cat([p_i_s, i_item_user_er], dim=3).view(-1, 2 * self.emb_dim)).view(p_i_s.size())  
        miu_s = self.i_friends_att(torch.cat([self.w2(y_ji_s), self.w2(q_j_s)], dim = 3).view(-1, 2 * self.emb_dim)).view(mask_s.size())    
        miu_s = torch.exp(miu_s) * mask_s
        miu_s = miu_s / (torch.sum(miu_s, 2).unsqueeze(2).expand_as(miu_s) + self.eps)
        z_kL_temp = torch.sum(miu_s.unsqueeze(3).expand_as(y_ji_s) * y_ji_s, 2)
        z_kL = self.aggre_i_friends(z_kL_temp.view(-1, self.emb_dim)).view(z_kL_temp.size())  
        z_kS = []
        for i in range(y_ji_s.shape[1]):
            output_s, h_n = self.gru(y_ji, self.h0)
            z_kS.append(output_s[:,-1,:])
        z_kS = torch.stack(z_kS).permute(1, 0, 2)
        z_kA = z_kL * z_kS
        z_kA = F.dropout(z_kA, p=0.5, training=self.training)
        mask_si = torch.where(i_item_pad > 0, torch.tensor([1.], device=self.device), torch.tensor([0.], device=self.device))
        kappa = self.if_friends_att(torch.cat([self.w3(z_kA), self.w3(self.item_emb(i_item_pad))], dim = 2).view(-1, 2 * self.emb_dim)).view(i_item_pad.size())
        kappa = torch.exp(kappa) * mask_si
        kappa = kappa / (torch.sum(kappa, 1).unsqueeze(1).expand_as(kappa) + self.eps)
        z_jN = self.aggre_if_friends(torch.sum(kappa.unsqueeze(2).expand_as(z_kA) * z_kA, 1))   
        z_jN = F.dropout(z_jN, p=0.5, training=self.training)
        z =  self.combine_mlp(torch.cat([z_jA, z_jN], dim = 1))
        return z