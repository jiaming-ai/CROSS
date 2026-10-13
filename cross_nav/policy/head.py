"""Causal visual memory with regression or conditional rectified-flow decoding.

Coordinates are metric (forward, left); outputs describe a geometric path, not
velocity commands. No future observation or privileged geometry enters the head.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class TrajectoryHead(nn.Module):
    def __init__(self, token_dim=2048, width=192, layers=4, heads=6, horizon=8,
                 source="multi", compression=0, mode="regression", max_views=8, geometry="none", action_scale=4., goal_encoding="linear", trail=0):
        super().__init__()
        if source not in ("last", "multi", "goal") or mode not in ("regression", "flow"):
            raise ValueError("invalid source or policy mode")
        self.cfg = dict(token_dim=token_dim, width=width, layers=layers, heads=heads,
                        horizon=horizon, source=source, compression=compression, mode=mode, max_views=max_views, geometry=geometry, action_scale=action_scale,
                        goal_encoding=goal_encoding, trail=trail)
        if goal_encoding not in ("linear", "polar"): raise ValueError(goal_encoding)
        self.goal_encoding = goal_encoding
        self.source, self.mode, self.horizon = source, mode, horizon
        if geometry not in ("none","scalar","metric"):raise ValueError(geometry)
        self.geometry,self.action_scale=geometry,action_scale
        self.scale_embed=nn.Sequential(nn.Linear(1,width),nn.SiLU(),nn.Linear(width,width)) if geometry!="none" else None
        self.xyz_embed=nn.Sequential(nn.Linear(3,width),nn.SiLU(),nn.Linear(width,width)) if geometry=="metric" else None
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, width))
                                   for _ in range(4 if source == "multi" else 1)])
        self.mix = nn.Parameter(torch.zeros(len(self.proj)))
        self.xy = nn.Linear(2, width)
        self.view = nn.Embedding(max_views, width)
        self.goal = nn.Sequential(nn.Linear(2 if goal_encoding == "linear" else 4, width), nn.SiLU(), nn.Linear(width, width))
        self.query = nn.Parameter(torch.randn(1, horizon, width) * .02)
        self.action = nn.Linear(2, width)
        self.time = nn.Sequential(nn.Linear(32, width), nn.SiLU(), nn.Linear(width, width))
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            width, heads, width * 4, dropout=0., activation="gelu", batch_first=True, norm_first=True), layers)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2))
        self.latents = nn.Parameter(torch.randn(1, compression, width) * .02) if compression else None
        # odometry trail: past positions (forward, left, available) at fixed travel distances, one memory token each
        self.trail = trail
        self.trail_embed = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width)) if trail else None
        self.trail_index = nn.Embedding(trail, width) if trail else None
        self.resampler = nn.MultiheadAttention(width, heads, batch_first=True) if compression else None

    def memory(self, tokens, geometry=None, trail=None):
        """tokens: B,L,S,H,W,C (layers 4/11/17/23, or just last)."""
        B,L,S,H,W,C = tokens.shape
        if self.source == "goal":
            return tokens.new_zeros(B, 1, self.query.shape[-1], dtype=self.query.dtype)
        if self.source == "last":
            tokens = tokens[:, -1:]
        elif L != 4:
            raise ValueError("multi-layer policy requires four layers")
        z = sum(w * proj(tokens[:, i].float()) for i, (w, proj) in enumerate(zip(self.mix.softmax(0), self.proj)))
        yy,xx = torch.meshgrid(torch.linspace(-1,1,H,device=z.device), torch.linspace(-1,1,W,device=z.device), indexing="ij")
        z = z + self.xy(torch.stack((xx,yy),-1)) + self.view(torch.arange(S,device=z.device))[None,:,None,None]
        if self.geometry != "none":
            if geometry is None:raise ValueError("scale conditioning requires geometry")
            log_s=geometry['log_scale'].float()
            if not torch.isfinite(log_s).all():raise ValueError("nonfinite scale")
            z=z+self.scale_embed(log_s[:,None]/4.)[:,None,None,None]
            if self.xyz_embed is not None:
                xyz=geometry['xyz_unit'].float()*log_s.exp()[:,None,None,None,None]
                if xyz.shape[:-1]!=z.shape[:-1]:raise ValueError("geometry/token grid mismatch")
                z=z+self.xyz_embed(torch.asinh(xyz/self.action_scale))
        z = z.flatten(1,3)
        if self.trail:
            if trail is None:
                trail = z.new_zeros(B, self.trail, 3)      # no trail: all points marked unavailable
            t = torch.cat((torch.asinh(trail[..., :2].float() / self.action_scale) * trail[..., 2:3], trail[..., 2:3]), -1)
            z = torch.cat((z, self.trail_embed(t) + self.trail_index.weight[None]), 1)
        if self.latents is not None:
            q = self.latents.expand(B,-1,-1)
            z = q + self.resampler(q,z,z,need_weights=False)[0]
        return z

    def encode_goal(self, goal):
        if self.goal_encoding == "linear":
            return goal / 8.
        # polar: direction, log distance and the distance clipped to the path horizon (far goals stay in range)
        d = goal.float().norm(dim=-1, keepdim=True)
        return torch.cat((goal / d.clamp_min(1e-3), torch.log1p(d), d.clamp(max=self.action_scale) / self.action_scale), -1)

    def decode(self, memory, goal, x=None, t=None):
        q = self.query.expand(goal.shape[0],-1,-1) + self.goal(self.encode_goal(goal))[:,None]
        if self.mode == "flow":
            if x is None or t is None:
                raise ValueError("flow decoding requires state and time")
            freq = torch.exp(torch.linspace(0, math.log(1000), 16, device=q.device))
            angle = t[:,None] * freq[None]
            q = q + self.action(x) + self.time(torch.cat((angle.sin(),angle.cos()),-1))[:,None]
        return self.out(self.decoder(q,memory))

    def loss(self, tokens, goal, target, geometry=None, trail=None):
        mem = self.memory(tokens,geometry,trail)
        y = target / self.action_scale
        if self.mode == "regression":
            pred = self.decode(mem,goal)
            return F.smooth_l1_loss(pred,y) + .2 * F.smooth_l1_loss(pred[:,1:]-pred[:,:-1],y[:,1:]-y[:,:-1])
        z = torch.randn_like(y)
        t = torch.rand(y.shape[0],device=y.device)
        x = (1-t[:,None,None])*z + t[:,None,None]*y
        return F.mse_loss(self.decode(mem,goal,x,t),y-z)

    @torch.no_grad()
    def predict(self, tokens, goal, steps=10, noise=None, geometry=None, trail=None):
        mem = self.memory(tokens,geometry,trail)
        if self.mode == "regression":
            return self.decode(mem,goal) * self.action_scale
        if steps < 1:
            raise ValueError("steps must be positive")
        x = torch.randn(goal.shape[0],self.horizon,2,device=goal.device) if noise is None else noise.clone()
        for i in range(steps):
            t = torch.full((goal.shape[0],),i/steps,device=goal.device)
            x = x + self.decode(mem,goal,x,t) / steps
        return x * self.action_scale
