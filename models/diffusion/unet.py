
import torch 
import torch.nn as nn 
from monai.networks.blocks import UnetOutBlock

from utils.conv_blocks import BasicBlock, UpBlock, DownBlock, UnetBasicBlock, UnetResBlock, save_add, BasicDown, BasicUp, SequentialEmb
from models.diffusion.time_embedder import TimeEmbedding
from utils.attention_blocks import Attention, zero_module
from einops import rearrange
from utils.adapter_block import AdapterBlock

class UNet(nn.Module):

    def __init__(self, 
            in_ch=1, 
            out_ch=1, 
            spatial_dims = 3,
            hid_chs =    [256, 256, 512,  1024],
            
            kernel_sizes=[ 3,  3,   3,   3],
            strides =    [ 1,  2,   2,   2], # WARNING, last stride is ignored (follows OpenAI)
            act_name=("SWISH", {}),
            norm_name = ("GROUP", {'num_groups':32, "affine": True}),
            time_embedder=TimeEmbbeding,
            time_embedder_kwargs={},
            cond_embedder=None,
            cond_embedder_kwargs={},
            deep_supervision=True, # True = all but last layer, 0/False=disable, 1=only first layer, ... 
            use_res_block=True,
            estimate_variance=False ,
            use_self_conditioning = False, 
            dropout=0.0, 
            learnable_interpolation=True,
            use_attention='none',
            num_res_blocks=2,
            vis_feat_dim=1024,      #ViT-H tokken dimension
            use_vis_adapter=True,   
        ):
        super().__init__()
        use_attention = use_attention if isinstance(use_attention, list) else [use_attention]*len(strides) 
        self.use_self_conditioning = use_self_conditioning
        self.use_res_block = use_res_block
        self.depth = len(strides)
        self.num_res_blocks = num_res_blocks
        self.use_vis_adapter = use_vis_adapter
        self.vis_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(vis_feat_dim, hid_chs[i]),
                nn.SiLU(),
                nn.Linear(hid_chs[i], hid_chs[i]),
            ) for i in range(self.depth)
        ])
        self.adapters = nn.ModuleList([
        AdapterBlock(dim=hid_chs[i], n_heads=8)
        for i in range(self.depth)
        ])

        # ------------- Time-Embedder-----------
        if time_embedder is not None:
            self.time_embedder=time_embedder(**time_embedder_kwargs)
            time_emb_dim = self.time_embedder.emb_dim
        else:
            self.time_embedder = None 
            time_emb_dim = None 

        # ------------- Condition-Embedder-----------
        if cond_embedder is not None:
            self.cond_embedder=cond_embedder(**cond_embedder_kwargs)
            cond_emb_dim = self.cond_embedder.emb_dim
        else:
            self.cond_embedder = None 
            cond_emb_dim = None 
            


        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock

        # ----------- In-Convolution ------------
        in_ch = in_ch*2 if self.use_self_conditioning else in_ch 
        self.in_conv = BasicBlock(spatial_dims, in_ch, hid_chs[0], kernel_size=kernel_sizes[0], stride=strides[0])
        
        
        # ----------- Encoder ------------
        in_blocks = [] 
        for i in range(1, self.depth):
            for k in range(num_res_blocks):
                seq_list = [] 
                seq_list.append(
                    ConvBlock(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i-1 if k==0 else i],
                        out_channels=hid_chs[i],
                        kernel_size=kernel_sizes[i],
                        stride=1,
                        norm_name=norm_name,
                        act_name=act_name,
                        dropout=dropout,
                        emb_channels=time_emb_dim
                    )
                )

                seq_list.append(
                    Attention(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i],
                        out_channels=hid_chs[i],
                        num_heads=8,
                        ch_per_head=hid_chs[i]//8,
                        depth=1,
                        norm_name=norm_name,
                        dropout=dropout,
                        emb_dim=time_emb_dim,
                        attention_type=use_attention[i]
                    )
                )
                in_blocks.append(SequentialEmb(*seq_list))

            if i < self.depth-1:
                in_blocks.append(
                    BasicDown(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i],
                        out_channels=hid_chs[i],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        learnable_interpolation=learnable_interpolation 
                    )
                )
 

        self.in_blocks = nn.ModuleList(in_blocks)
        
        # ----------- Middle ------------
        self.middle_block = SequentialEmb(
            ConvBlock(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[-1],
                out_channels=hid_chs[-1],
                kernel_size=kernel_sizes[-1],
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
                dropout=dropout,
                emb_channels=time_emb_dim
            ),
            Attention(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[-1],
                out_channels=hid_chs[-1],
                num_heads=8,
                ch_per_head=hid_chs[-1]//8,
                depth=1,
                norm_name=norm_name,
                dropout=dropout,
                emb_dim=time_emb_dim,
                attention_type=use_attention[-1]
            ),
            ConvBlock(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[-1],
                out_channels=hid_chs[-1],
                kernel_size=kernel_sizes[-1],
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
                dropout=dropout,
                emb_channels=time_emb_dim
            )
        )

 
     
        # ------------ Decoder ----------
        out_blocks = [] 
        for i in range(1, self.depth):
            for k in range(num_res_blocks+1):
                seq_list = [] 
                out_channels=hid_chs[i-1 if k==0 else i]
                seq_list.append(
                    ConvBlock(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i]+hid_chs[i-1 if k==0 else i],
                        out_channels=out_channels,
                        kernel_size=kernel_sizes[i],
                        stride=1,
                        norm_name=norm_name,
                        act_name=act_name,
                        dropout=dropout,
                        emb_channels=time_emb_dim
                    )
                )
            
                seq_list.append(
                    Attention(
                        spatial_dims=spatial_dims,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        num_heads=8,
                        ch_per_head=out_channels//8,
                        depth=1,
                        norm_name=norm_name,
                        dropout=dropout,
                        emb_dim=time_emb_dim,
                        attention_type=use_attention[i]
                    )
                )

                if (i >1) and k==0:
                    seq_list.append(
                        BasicUp(
                            spatial_dims=spatial_dims,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=strides[i],
                            stride=strides[i],
                            learnable_interpolation=learnable_interpolation 
                        )
                    )
        
                out_blocks.append(SequentialEmb(*seq_list))
        self.out_blocks = nn.ModuleList(out_blocks)
        
        
        # --------------- Out-Convolution ----------------
        out_ch_hor = out_ch*2 if estimate_variance else out_ch
        self.outc = zero_module(UnetOutBlock(spatial_dims, hid_chs[0], out_ch_hor, dropout=None))
        if isinstance(deep_supervision, bool):
            deep_supervision = self.depth-2 if deep_supervision else 0 
        self.outc_ver = nn.ModuleList([
            zero_module(UnetOutBlock(spatial_dims, hid_chs[i]+hid_chs[i-1], out_ch, dropout=None) )
            for i in range(2, deep_supervision+2)
        ])

        self.hid_chs = hid_chs
        self.ch2idx  = {c: i for i, c in enumerate(hid_chs)}
 

    def forward(self, x_t, t=None, condition=None, vis_tokens=None, beta=0.3, self_cond=None):
        # x_t [B, C, *]
        # t [B,]
        # condition [B,]
        # self_cond [B, C, *]
        

        # -------- Time Embedding (Gloabl) -----------
        if t is None:
            time_emb = None 
        else:
            time_emb = self.time_embedder(t) # [B, C]

        # -------- Condition Embedding (Gloabl) -----------
        if (condition is None) or (self.cond_embedder is None):
            cond_emb = None  
        else:
            cond_emb = self.cond_embedder(condition) # [B, C]
        
        emb = save_add(time_emb, cond_emb)
       
        # ---------- Self-conditioning-----------
        if self.use_self_conditioning:
            self_cond =  torch.zeros_like(x_t) if self_cond is None else x_t 
            x_t = torch.cat([x_t, self_cond], dim=1)  
    
        # --------- Encoder --------------
        x = [self.in_conv(x_t)]
        
        max_tokens = 4096
  
        for blk in self.in_blocks:
            h = blk(x[-1], emb)

            use_adapter = (
            self.use_vis_adapter
            and (vis_tokens is not None)
            and not isinstance(blk, BasicDown)      # Down-sampling 블록은 건너뜀
            and (h.shape[2] * h.shape[3] <= max_tokens)
            )

            if use_adapter:
              idx = self.ch2idx[h.shape[1]] 
              vis_proj = self.vis_mlps[idx](vis_tokens[idx].to(h.dtype))
              B, C, Ht, Wt = h.shape
              h_seq = rearrange(h, 'b c h w -> b (h w) c')
              h_seq = h_seq + beta * self.adapters[idx](h_seq, vis_proj)
              h     = rearrange(h_seq, 'b (h w) c -> b c h w', h=Ht, w=Wt)

              
            x.append(h)
                   

        # ---------- Middle --------------
        h = self.middle_block(x[-1], emb)
        
        # -------- Decoder -----------
        y_ver = []
        for i in range(len(self.out_blocks), 0, -1):
            h = torch.cat([h, x.pop()], dim=1)

            depth, j = i//(self.num_res_blocks+1), i%(self.num_res_blocks+1)-1
            y_ver.append(self.outc_ver[depth-1](h)) if (len(self.outc_ver)>=depth>0) and (j==0) else None 

            h = self.out_blocks[i-1](h, emb)

        # ---------Out-Convolution ------------
        y = self.outc(h)

        return y, y_ver[::-1]




if __name__=='__main__':
    model = UNet(in_ch=3, use_res_block=False, learnable_interpolation=False)
    input = torch.randn((1,3,16,32,32))
    time = torch.randn((1,))
    out_hor, out_ver = model(input, time)
    print(out_hor[0].shape)