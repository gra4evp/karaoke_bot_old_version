# -*- coding: utf-8 -*-
import torch
import functools
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn


class MoCo(torch.nn.Module):
            
    def __init__(self, base_encoder, dim=2048, m=0.999):
        super(MoCo, self).__init__()
        self.m = m
        self.base_encoder = base_encoder(pretrained=True)
        self.momentum_encoder = base_encoder(pretrained=True)
        
        self.fc_layer = torch.nn.Linear(dim, 16)
        
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """
        Momentum update of the momentum encoder
        θ_k = m * θ_k + (1-m) * θ_q
        """
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + (1 - m) * param_b.data
    
    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)

        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k])
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long, device=logits.device)).cuda()
        return nn.CrossEntropyLoss()(logits, labels) 
        
    def forward(self, batch):
        batch = batch.permute(1, 0, 2, 3)

        x1 = batch[0].unsqueeze(1).clone()
        x2 = batch[1].unsqueeze(1).clone()
        
        tile = transforms.Lambda(lambda x: torch.cat([x, x, x], 1))
        x1 = tile(x1).float()
        x2 = tile(x2).float()

        q1 = self.base_encoder(x1)
        q2 = self.base_encoder(x2)
        
        with torch.no_grad():
            self._update_momentum_encoder(self.m)

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)
        
        q1 = self.fc_layer(q1.view(q1.size(0), -1))
        q2 = self.fc_layer(q2.view(q2.size(0), -1))
        
        k1 = self.fc_layer(k1.view(k1.size(0), -1))
        k2 = self.fc_layer(k2.view(k2.size(0), -1))

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


class MoCoInference(torch.nn.Module):
    def __init__(self, base_encoder, dim=2048):
        super(MoCoInference, self).__init__()
        self.dim = dim
        self.base_encoder = base_encoder(pretrained=False)
        self.tile = transforms.Lambda(lambda x: torch.cat([x, x, x], 1))
        self.fc_layer = torch.nn.Linear(self.dim, 16)

    def forward(self, samples):
        samples = samples.permute(1, 0, 2, 3)
        input_spectr = samples[0].unsqueeze(1).clone()
        input_spectr = self.tile(input_spectr).float()
        output_cnn = self.base_encoder(input_spectr)
        flatten_vec = output_cnn.view(output_cnn.size(0), -1)
        output_fc = self.fc_layer(flatten_vec)
        return output_fc


class MoCoV3(torch.nn.Module):
            
    def __init__(self, base_encoder, dim=16, mlp_dim=2048, m=0.999, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCoV3, self).__init__()
        self.m = m
        self.T = T
        
        # build encoders
        self.base_encoder = base_encoder(pretrained=True)
        self.momentum_encoder = base_encoder(pretrained=True)

        
        # build projector and predictor_mlps
        # ---------------------------------------------------------------------------------------------
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)
        # ----------------------------------------------------------------------------------------------
        
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """
        Momentum update of the momentum encoder
        θ_k = m * θ_k + (1-m) * θ_q
        """
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + (1 - m) * param_b.data 
    
    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        # k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long, device=logits.device)).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T) 
    
    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)
    
    def forward(self, batch):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            loss
        """
        # Выбираем вектора из батча по второй размерности
        # ------------------------------------------------------------------------------------------
        x1 = batch[:, 0, :, :].unsqueeze(1)
        x2 = batch[:, 1, :, :].unsqueeze(1)
        
        # Делаем спектрограммы трехканальными для resnet50
        tile = transforms.Lambda(lambda x: torch.cat([x, x, x], 1))
        x1 = tile(x1).float()
        x2 = tile(x2).float()
        # ------------------------------------------------------------------------------------------
        
        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():
            self._update_momentum_encoder(self.m)

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


class MoCoV3Inference(torch.nn.Module):
            
    def __init__(self, base_encoder, dim=16, mlp_dim=2048):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        """
        super(MoCoV3Inference, self).__init__()
        # build encoder
        self.base_encoder = base_encoder()
        
        
        # build projector and predictor_mlps
        # ---------------------------------------------------------------------------------------------
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc  # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)
        # ----------------------------------------------------------------------------------------------
    
    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)
    
    def forward(self, x):
        """
        Input:
            x views of images
        Output:
            logits
        """        
        # Делаем спектрограмму трехканальными для resnet50
        tile = transforms.Lambda(lambda x: torch.cat([x, x, x], 1))
        x = tile(x).float()
        
        with torch.no_grad():
            logits = self.predictor(self.base_encoder(x))
        return logits


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def make_base_encoder(model):
    @functools.wraps(model)
    def wrapper(*args, **kwargs):
        # Создаем экземпляр модели
        encoder = model(*args, **kwargs)
        
        # Изменяем количество входных каналов первого сверточного слоя
        first_conv_layer = None
        for name, module in encoder.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                first_conv_layer = module
                break
        
        if first_conv_layer is None:
            raise ValueError("Первый сверточный слой не найден в модели.")
        
        # Изменяем количество входных каналов и веса первого сверточного слоя
        with torch.no_grad():
            new_weights = first_conv_layer.weight[:, :1, :, :]  # Берем только первый канал
            first_conv_layer.weight = torch.nn.Parameter(new_weights)
            first_conv_layer.in_channels = 1
        
        print('Изменили веса и число входов')
        
        return encoder
    return wrapper


if __name__ == '__main__':
    # checkpoint_path = '/app/pgrachev/moco/experiments/exp2/checkpoints_with_momentum/model_ep0200.pt'
    
    # base_encoder = make_base_encoder(torchvision.models.resnet50)
    # model = MoCoInference(base_encoder=base_encoder)
    # checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # for key in list(checkpoint['model_state_dict'].keys()):
    #     if 'momentum_encoder' in key:
    #         del checkpoint['model_state_dict'][key]
    
    # if 'momentum' in checkpoint['model_state_dict']:
    #     del checkpoint['model_state_dict']['momentum']
    
    # base_encoder_state_dict = checkpoint['model_state_dict']
    # model.load_state_dict(base_encoder_state_dict)
    # model.eval()
#     model = MoCoV3(base_encoder=torchvision.models.resnet50)
#     for name, param in model.named_parameters():
#         print(name, param.shape)
    
    
    base_encoder = make_base_encoder(model=torchvision.models.resnet50)
    model = MoCoV3(base_encoder=base_encoder)
    # Выводим имя и количество параметров каждого слоя
    total_params = 0
    for name, module in model.named_modules():
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_params += num_params
        print(f"Layer: {name}, Parameters: {num_params}")

    # Выводим общее количество параметров
    print(f"Total trainable parameters: {total_params}")
    
    # Подсчитываем общее количество параметров
    print()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
