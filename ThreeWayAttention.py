import torch
from torch import nn, einsum
from einops import rearrange, repeat

class ThreeWayAttention(nn.Module):
    '''
    Multiheaded three way attention. pass in three sets of tokens, output three sets of tokens but now they've all attended to each other.
    Could generalize further with 4way etc but it just gets more and more expensive. Also some combinatorial choices start to appear.
    '''
    def __init__(self, input_channels, heads, inner_channels_per_head, dropout=0.):
        super(ThreeWayAttention, self).__init__()
        self.input_channels = input_channels
        inner_channels = heads * inner_channels_per_head
        self.heads = heads
        self.scale = inner_channels_per_head ** -1./3.

        self.from_A = nn.Linear(input_channels[0], inner_channels, bias=False)
        self.from_B = nn.Linear(input_channels[1], inner_channels, bias=False)
        self.from_C = nn.Linear(input_channels[2], inner_channels, bias=False)

        self.val_A = nn.Linear(input_channels[0], inner_channels, bias=False)
        self.val_B = nn.Linear(input_channels[1], inner_channels, bias=False)
        self.val_C = nn.Linear(input_channels[2], inner_channels, bias=False)


        self.to_A = nn.Sequential(nn.Linear(inner_channels, input_channels[0]), nn.Dropout(dropout))
        self.to_B = nn.Sequential(nn.Linear(inner_channels, input_channels[1]), nn.Dropout(dropout))
        self.to_C = nn.Sequential(nn.Linear(inner_channels, input_channels[2]), nn.Dropout(dropout))

    def forward(self, A, B, C, mask): 
        bs, nA, cA = A.shape 
        _, nB, cB = B.shape 
        _, nC, cC = C.shape 
        assert [cA, cB, cC] == self.input_channels
        assert list(mask.shape) == [bs, nA, nB, nC]

        # linear layer embed all tokens into the same scoring space
        # WE DONT SEPARATE Q AND K BECAUSE WE ASSUME ATTENDING IS SYMMETRIC. token 1 attends to token 2 means token 2 attends to token 1.
        aa = self.from_A(A)
        bb = self.from_B(B)
        cc = self.from_C(C)

        vaa = self.val_A(A)
        vbb = self.val_B(B)
        vcc = self.val_C(C)

        # group the batch and heads dimensions
        a, b, c, va, vb, vc = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (aa, bb, cc, vaa, vbb, vcc))

        # the core of it all. a third order score tensor (+1 more order to include batch dimension)
        sim = einsum('e a d, e b d, e c d -> e a b c', a, b, c) * self.scale

        # mask if we dont want everyone to attend to everything. True means attend. False means no attend.
        if mask is not None:
            mask = repeat(mask, 'b ... -> (b h) ...', h=self.heads)
            max_neg_value = -torch.finfo(sim.dtype).max
            attn_bias = torch.zeros_like(sim)
            attn_bias.masked_fill_(~mask, max_neg_value)
            sim = sim + attn_bias

        # score tensor gets contracted against two sets of tokens. softmax across both those tokens' worth of dimensions
        attn_a = rearrange(sim, 'e a b c -> e a (b c)').softmax(dim=2)
        attn_b = rearrange(sim, 'e a b c -> e b (c a)').softmax(dim=2)
        attn_c = rearrange(sim, 'e a b c -> e c (a b)').softmax(dim=2)

        # get (V)alues from standard attention. except the values are pairwise products of tokens
        ab = rearrange(einsum('e a d, e b d -> e a b d', va, vb), 'a b c d -> a (b c) d')
        bc = rearrange(einsum('e b d, e c d -> e b c d', vb, vc), 'a b c d -> a (b c) d')
        ca = rearrange(einsum('e c d, e a d -> e c a d', vc, va), 'a b c d -> a (b c) d')

        # contract tensor against Value tensors
        out_a0 = einsum('b i j, b j d -> b i d', attn_a, bc)
        out_b0 = einsum('b i j, b j d -> b i d', attn_b, ca)
        out_c0 = einsum('b i j, b j d -> b i d', attn_c, ab)

        # split heads out for out linear to recieve
        out_a1 = rearrange(out_a0, '(b h) n d -> b n (h d)', h=self.heads)
        out_b1 = rearrange(out_b0, '(b h) n d -> b n (h d)', h=self.heads)
        out_c1 = rearrange(out_c0, '(b h) n d -> b n (h d)', h=self.heads)

        out_A = self.to_A(out_a1)
        out_B = self.to_B(out_b1)
        out_C = self.to_C(out_c1)

        return out_A, out_B, out_C
    
def main():

    for _ in range(100):
        cA = torch.randint(10, (1,1)).item()+1
        cB = torch.randint(10, (1,1)).item()+1
        cC = torch.randint(10, (1,1)).item()+1
        input_channels = [cA, cB, cC]
        heads = torch.randint(10, (1,1)).item()+1
        dim_head = torch.randint(10, (1,1)).item()+1
        twa = ThreeWayAttention(input_channels, heads, dim_head)

        nA = torch.randint(10, (1,1)).item()+1
        nB = torch.randint(10, (1,1)).item()+1
        nC = torch.randint(10, (1,1)).item()+1
        batch_size = torch.randint(10, (1,1)).item()+1
        A = torch.randn(batch_size, nA, cA)
        B = torch.randn(batch_size, nB, cB)
        C = torch.randn(batch_size, nC, cC)
        mask = torch.ones(batch_size, nA, nB, nC, dtype=bool)
        Atwa, Btwa, Ctwa = twa(A, B, C, mask)

        assert A.shape == Atwa.shape
        assert B.shape == Btwa.shape
        assert C.shape == Ctwa.shape

        b = torch.randint(batch_size, (1,1)).item()+1
        Atwa1, Btwa1, Ctwa1 = twa(A[b:b+1,:,:], B[b:b+1,:,:], C[b:b+1,:,:], mask[b:b+1,:,:,:])
        assert ((Atwa1 - Atwa[b:b+1,:,:]).norm() < 1e-6)
        assert ((Btwa1 - Btwa[b:b+1,:,:]).norm() < 1e-6)
        assert ((Ctwa1 - Ctwa[b:b+1,:,:]).norm() < 1e-6)

if __name__ == "__main__":
    main()