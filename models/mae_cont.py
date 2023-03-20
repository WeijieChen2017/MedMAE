# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from .mae_utils import get_2d_sincos_pos_embed

import inspect

class MaskedAutoencoderViT_cont(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, modality_club=["MR", "PET"],
                 img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 loss_type="l2"):
        super().__init__()

        # --------------------------------------------------------------------------
        # parameter settings
        self.modality_club = modality_club
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.norm_pix_loss = norm_pix_loss
        self.loss_type = loss_type
        # --------------------------------------------------------------------------


        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) # qk_scale=None,
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        self.decoder_club = nn.ModuleDict()
        for modality in self.modality_club:
            self.decoder_club[modality] = self.build_decoder()
        # # --------------------------------------------------------------------------
        # # MAE decoder specifics
        # self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        # self.decoder_blocks = nn.ModuleList([
        #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     for i in range(decoder_depth)])

        # self.decoder_norm = norm_layer(decoder_embed_dim)
        # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def build_decoder(self):
        decoder = nn.ModuleList()
        decoder.decoder_embed = nn.Linear(self.embed_dim, self.decoder_embed_dim, bias=True)
        decoder.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        decoder.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        decoder.decoder_blocks = nn.ModuleList([
            Block(self.decoder_embed_dim, self.decoder_num_heads, self.mlp_ratio, qkv_bias=True, norm_layer=self.norm_layer) #  qk_scale=None,
            for i in range(self.decoder_depth)])
        decoder.decoder_norm = self.norm_layer(self.decoder_embed_dim)
        decoder.decoder_pred = nn.Linear(self.decoder_embed_dim, self.patch_size**2 * self.in_chans, bias=True) # decoder to patch
        return decoder

    # def build_decoder_dict(self, modality):
    #     decoder = dict()
    #     if modality == "MR":
    #         decoder["flag"] = "I am MR decoder"
    #     decoder["modality"] = modality
    #     decoder["decoder_embed"] = nn.Linear(self.embed_dim, self.decoder_embed_dim, bias=True)
    #     decoder["mask_token"] = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
    #     decoder["decoder_pos_embed"] = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
    #     decoder["decoder_blocks"] = nn.ModuleList([
    #         Block(self.decoder_embed_dim, self.decoder_num_heads, self.mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=self.norm_layer)
    #         for i in range(self.decoder_depth)])
    #     decoder["decoder_norm"] = self.norm_layer(self.decoder_embed_dim)
    #     decoder["decoder_pred"] = nn.Linear(self.decoder_embed_dim, self.patch_size**2 * self.in_chans, bias=True) # decoder to patch
    #     return decoder

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        for modality in self.modality_club:
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_club[modality].decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            self.decoder_club[modality].decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
            # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
            # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

            # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
            w = self.patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

            # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
            torch.nn.init.normal_(self.cls_token, std=.02)
            torch.nn.init.normal_(self.decoder_club[modality].mask_token, std=.02)

            # initialize nn.Linear and nn.LayerNorm
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    # def random_masking(self, x, mask_ratio):
    #     """
    #     Perform per-sample random masking by per-sample shuffling.
    #     Per-sample shuffling is done by argsort random noise.
    #     x: [N, L, D], sequence
    #     """
    #     N, L, D = x.shape  # batch, length, dim
    #     len_keep = int(L * (1 - mask_ratio))
        
    #     noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
    #     # sort noise for each sample
    #     ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    #     ids_restore = torch.argsort(ids_shuffle, dim=1)

    #     # keep the first subset
    #     ids_keep = ids_shuffle[:, :len_keep]
    #     x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    #     # generate the binary mask: 0 is keep, 1 is remove
    #     mask = torch.ones([N, L], device=x.device)
    #     mask[:, :len_keep] = 0
    #     # unshuffle to get the binary mask
    #     mask = torch.gather(mask, dim=1, index=ids_restore)

    #     return x_masked, mask, ids_restore

    # def forward_encoder(self, x, mask_ratio):
    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x #, mask, ids_restore

    # def forward_decoder(self, x, modality, ids_restore):
    def forward_decoder(self, x, modality):
        # embed tokens
        decoder = self.decoder_club[modality]
        # x = decoder.decoder_embed(x)

        # append mask tokens to sequence
        # mask_tokens = decoder.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        # x = x + decoder.decoder_pos_embed

        # apply Transformer blocks
        for blk in decoder.decoder_blocks:
            x = blk(x)
        x = decoder.decoder_norm(x)

        # predictor projection
        x = decoder.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    
    # def forward_loss(self, imgs, pred, mask):
    def forward_loss(self, trgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(trgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        if self.loss_type == "l2":
            loss = (pred - target) ** 2
        if self.loss_type == "l1":
            loss = (pred - target).abs()
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    # def forward(self, imgs, trgs, modality, mask_ratio=0.75):
    def forward(self, imgs, trgs, modality):
        # latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        latent = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, modality)  # [N, L, p*p*3]
        loss = self.forward_loss(trgs, pred)
        return loss, pred


def mae_vit_base_patch16_dec512d8b_cont(**kwargs):
    model = MaskedAutoencoderViT_cont(
        modality_club=["MR", "PET"],
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b_cont(modality_club, **kwargs):
    model = MaskedAutoencoderViT_cont(
        modality_club=modality_club, img_size=256,
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b_cont(**kwargs):
    model = MaskedAutoencoderViT_cont(
        modality_club=["MR", "PET"],
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16_cont = mae_vit_base_patch16_dec512d8b_cont  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16_cont = mae_vit_large_patch16_dec512d8b_cont  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14_cont = mae_vit_huge_patch14_dec512d8b_cont  # decoder: 512 dim, 8 blocks