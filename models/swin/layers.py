from torch import nn
from torchvision.models.swin_transformer import SwinTransformerBlock, PatchMerging
import torch

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        patches_resolution = [img_size[0] //
                              patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        #self.proj = Conv2D(embed_dim, kernel_size=patch_size,
        #                   strides=patch_size, name='proj')
        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, bias = False)
        

    def forward(self, x):
        x = self.proj(x)
        
        return x
    
class PatchExpand(nn.Module):
    def __init__(self, dim: int, norm_layer = nn.LayerNorm):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        if norm_layer is not None:
            self.norm_layer = norm_layer(2*dim)
        else:
            self.norm_layer = None
        

    def forward(self, x):
        x = self.expand(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        x = x.permute((0,3,1,2))
        x = nn.functional.pixel_shuffle(x, 2)
        x = x.permute((0,2,3,1))
        
        return x

class PatchExpandx4(nn.Module):
    def __init__(self, dim: int, norm_layer = nn.LayerNorm):
        super().__init__()
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        if norm_layer is not None:
            self.norm_layer = norm_layer(16 * dim)
        else:
            self.norm_layer = None
        

    def forward(self, x):
        x = self.expand(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        x = x.permute((0,3,1,2))
        x = nn.functional.pixel_shuffle(x, 4)
        x = x.permute((0,2,3,1))
        
        return x

class SwinTransformerBlockSet(nn.Module):
    def __init__(
            self,
            n_blocks,
            dim,
            n_heads,
            window_size,
            shift_size
            ) -> None:
        super().__init__()
        
        self.msa_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim = dim, 
                num_heads = n_heads, 
                window_size= window_size,
                shift_size=[0 if i%2==0 else shift_size[0], 0 if i%2==0 else shift_size[1]] 
            )
            for i in range(n_blocks)
        ])

    def forward(self, x):
        for msa_block in self.msa_blocks:
            x = msa_block(x)
        return x

class SwinEncoder(nn.Module):
    def __init__(
            self, 
            input_depth, 
            base_dim, 
            window_size,
            shift_size,
            img_size,
            patch_size,
            n_heads,
            n_blocks) -> None:
        super(SwinEncoder, self).__init__()
        self.n_layers = len(n_blocks)
        self.embed = PatchEmbed(
            img_size=[img_size, img_size],
            patch_size=patch_size,
            in_chans=input_depth,
            embed_dim=base_dim
            )
        
        self.msa_sets = nn.ModuleList([
            SwinTransformerBlockSet(
                n_blocks = n_blocks[i],
                dim = base_dim * (2**i),
                n_heads = n_heads[i],
                window_size = window_size,
                shift_size = shift_size
            )
            for i in range(self.n_layers)
        ])
        self.merges = nn.ModuleList([
            PatchMerging((2**i)*base_dim) for i in range(self.n_layers-1)
        ])

    def forward(self, x):
        x = self.embed(x)
        x = x.permute((0,2,3,1))
        x_out = [self.msa_sets[0](x)]
        for i in range(1, self.n_layers):
            x_l = self.merges[i-1](x_out[-1])
            x_l = self.msa_sets[i](x_l)
            x_out.append(x_l)
        
        return x_out

class SwinDecoder(nn.Module):
    def __init__(self, 
                 base_dim, 
                 n_heads,
                 n_blocks,
                 window_size,
                 shift_size
                 ) -> None:
        super(SwinDecoder, self).__init__()

        self.n_layers = len(n_blocks)

        self.expands = nn.ModuleList([
            PatchExpand(dim = (2**(i+1))*base_dim) for i in range(self.n_layers-1)
        ])

        self.msa_sets = nn.ModuleList([
            SwinTransformerBlockSet(
                n_blocks = n_blocks[i],
                dim = base_dim * (2**i),
                n_heads = n_heads[i],
                window_size = window_size,
                shift_size = shift_size
            )
            for i in range(self.n_layers-1)
        ])

        self.linear_projs  = nn.ModuleList([
            nn.Conv2d( (2**(i+1))*base_dim,  (2**(i))*base_dim, kernel_size=1, bias = False)
            for i in range(self.n_layers-1)
        ])
        

        #self.patch_expand_last = PatchExpandx4(dim = base_dim)

    

    def forward(self, x):
        x_out = x[-1]
        for i in range(self.n_layers-2, -1, -1):
            x_out = self.expands[i](x_out)
            x_out = torch.cat([x_out, x[i]], dim = -1)
            x_out = self.linear_projs[i](x_out.permute((0,3,1,2))).permute((0,2,3,1))
            x_out = self.msa_sets[i](x_out)
        #x_out = self.patch_expand_last(x_out)
        
        return x_out

class SwinDecoderJF(nn.Module):
    def __init__(self, 
                 base_dim, 
                 n_heads,
                 n_blocks,
                 window_size,
                 shift_size
                 ) -> None:
        super(SwinDecoderJF, self).__init__()

        self.n_layers = len(n_blocks)

        self.expands = nn.ModuleList([
            PatchExpand(dim = (2**(i+1))*base_dim) for i in range(self.n_layers-1)
        ])

        self.msa_sets = nn.ModuleList([
            SwinTransformerBlockSet(
                n_blocks = n_blocks[i],
                dim = base_dim * (2**i),
                n_heads = n_heads[i],
                window_size = window_size,
                shift_size = shift_size
            )
            for i in range(self.n_layers-1)
        ])

        self.linear_projs  = nn.ModuleList([
            nn.Conv2d( (2**(i+1))*base_dim,  (2**(i))*base_dim, kernel_size=1, bias = False)
            for i in range(self.n_layers)
        ])

        self.linear_projs_skip  = nn.ModuleList([
            nn.Conv2d( (2**(i+1))*base_dim,  (2**(i))*base_dim, kernel_size=1, bias = False)
            for i in range(self.n_layers-1)
        ])

        #self.patch_expand_last = PatchExpandx4(dim = base_dim)

    def forward(self, x):
        x_out = x[-1]
        x_out = torch.cat((x[0][-1], x[1][-1]), dim = -1)
        x_out = self.linear_projs[self.n_layers-1](x_out.permute((0,3,1,2))).permute((0,2,3,1))
        for i in range(self.n_layers-2, -1, -1):
            x_out = self.expands[i](x_out)
            x_skip = torch.cat((x[0][i], x[1][i]), dim = -1)
            x_skip = self.linear_projs_skip[i](x_skip.permute((0,3,1,2))).permute((0,2,3,1))
            x_out = torch.cat([x_out, x_skip], dim = -1)
            x_out = self.linear_projs[i](x_out.permute((0,3,1,2))).permute((0,2,3,1))
            x_out = self.msa_sets[i](x_out)
        #x_out = self.patch_expand_last(x_out)
        
        return x_out

class SwinClassifier(nn.Module):
    def __init__(self, 
                 base_dim, 
                 n_heads,
                 n_blocks,
                 window_size,
                 shift_size,
                 n_classes) -> None:
        super(SwinClassifier, self).__init__()

        self.last_msa = SwinTransformerBlockSet(
                n_blocks = n_blocks[0],
                dim = base_dim,
                n_heads = n_heads[0],
                window_size = window_size,
                shift_size = shift_size
            )

        self.patch_expand_last = PatchExpandx4(dim = base_dim)

        self.last_proj = nn.Conv2d( base_dim,  n_classes, kernel_size=1, bias=False)
        self.last_act = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.last_msa(x)
        x = self.patch_expand_last(x)
        x = x.permute((0,3,1,2))
        x = self.last_proj(x)
        x = self.last_act(x)
        return x

class SwinRegressionClassifier(nn.Module):
    def __init__(self, 
                 base_dim, 
                 n_heads,
                 n_blocks,
                 window_size,
                 shift_size) -> None:
        super().__init__()

        self.last_msa = SwinTransformerBlockSet(
                n_blocks = n_blocks[0],
                dim = base_dim,
                n_heads = n_heads[0],
                window_size = window_size,
                shift_size = shift_size
            )

        self.patch_expand_last = PatchExpandx4(dim = base_dim)

        self.last_proj = nn.Conv2d(base_dim, 2, kernel_size=1, bias=False)
        self.last_act = nn.Sigmoid()

    def forward(self, x):
        x = self.last_msa(x)
        x = self.patch_expand_last(x)
        x = x.permute((0,3,1,2))
        x = self.last_proj(x)
        x = self.last_act(x)
        return x