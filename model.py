import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input tensor to help the model understand the order of the sequence.
    The Transformer architecture does not have an inherent notion of position, so this class encodes 
    positional information using sine and cosine functions of different frequencies.
    Args:
        d_model (int): Dimension of the model (the size of the input embedding).
        device (str): Device on which the positional encodings are calculated (e.g., 'cpu', 'cuda').
        max_len (int): Maximum length of the input sequence to encode positions for.
    """
    
    def __init__(self, d_model, device="cpu", max_len=500):
        super().__init__()
        
        # Create a positional encoding matrix of shape (max_len, d_model)
        self.pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).type(torch.float32)
        tmp = torch.arange(0, d_model, 2)
        den = 1 / torch.pow(torch.ones(int(d_model / 2)) * max_len, 2 * tmp / d_model)
        den = den.unsqueeze(0)
       
        # Populate the positional encoding matrix with sine and cosine functions
        self.pe[:, 0::2] = torch.sin(torch.matmul(pos, den))
        self.pe[:, 1::2] = torch.cos(torch.matmul(pos, den))
        self.pe = self.pe.to(device)
   
   
    def forward(self, x):
        """
        Adds the positional encoding to the input tensor.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            Tensor: Input tensor with positional encoding added.
        """
        return x + self.pe[:x.shape[1], :]

class PreLayer(nn.Module):
    """
    A pre-processing layer that applies a linear transformation to input data, mapping it to the model's
    dimensionality.
    Args:
        hid (int): Hidden layer size.
        d_model (int): Dimension of the model.
        drop_out (float): Dropout rate.
        in_dim (int): Dimension of the input data.
    """
    
    def __init__(self, hid, d_model, drop_out=0.0, in_dim=1):
        super().__init__()
        self.linear = nn.Linear(in_dim, d_model)
   
    def forward(self, x):
        """
        Transforms the input using a linear layer.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, in_dim).
        Returns:
            Tensor: Transformed tensor of shape (batch_size, seq_len, d_model).
        """
        
        out = self.linear(x)
        return out

class PostLayer(nn.Module):
    """
    A post-processing layer that applies a linear transformation to reduce the dimensionality to the desired
    output size.
    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hid (int): Hidden layer size.
        drop_out (float): Dropout rate.
    """
    
    def __init__(self, in_dim, out_dim, hid, drop_out=0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
   
    def forward(self, x):
        """
        Applies a linear transformation to the input data.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, in_dim).
        Returns:
            Tensor: Transformed tensor of shape (batch_size, seq_len, out_dim).
        """
        
        out = self.linear(x)
        return out

class SelfAttention(nn.Module):
    """
    Self-Attention mechanism to compute attention scores and apply them to the input sequence. This helps 
    the model focus on different parts of the input sequence when making predictions.
    Args:
        d_model (int): Dimension of the model.
        n_head (int): Number of attention heads (for multi-head attention).
        attn_type (str): Type of attention mechanism (e.g., 'full' or 'causal').
    """
    
    def __init__(self, d_model, n_head, attn_type):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.n_head = n_head
        self.attn_type = attn_type
        self.softmax = nn.Softmax(dim=-1)
        scale = torch.sqrt(torch.FloatTensor([d_model // n_head]))
        self.register_buffer('scale', scale)
  
    def forward(self, x, mask=None):
        """
        Applies the self-attention mechanism to the input sequence.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor, optional): Mask to avoid attending to certain positions (e.g., padding or future tokens).
        Returns:
            Tensor: Output of the attention mechanism with the same shape as the input.
        """
        
        B, N, D = x.shape
        q = self.query(x).view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        k = self.key(x).view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, N, self.n_head, D // self.n_head).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.softmax(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, D)
        return out

class FeedForward(nn.Module):
    """
    A two-layer feed-forward neural network applied independently to each position in the sequence.
    Args:
        d_model (int): Dimension of the model.
        ff_hidnum (int): Size of the hidden layer in the feed-forward network.
    """
    
    def __init__(self, d_model, ff_hidnum):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ff_hidnum)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ff_hidnum, d_model)
 
    def forward(self, x):
        """
        Applies the feed-forward network to the input.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            Tensor: Output tensor with the same shape as the input.
        """
        return self.linear2(self.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    """
    A single layer of the Transformer encoder. It consists of a self-attention mechanism followed by
    a feed-forward network, both with residual connections and layer normalization.
    Args:
        d_model (int): Dimension of the model.
        n_head (int): Number of attention heads.
        attn_type (str): Type of attention.
        ff_hidnum (int): Hidden layer size of the feed-forward network.
        drop_out (float): Dropout rate.
    """
    
    def __init__(self, d_model, n_head, attn_type, ff_hidnum, drop_out=0.1):
        super().__init__()
        self.attn = SelfAttention(d_model, n_head, attn_type)
        self.ff = FeedForward(d_model, ff_hidnum)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)
  
    def forward(self, x, mask=None):
        """
        Forward pass through a single encoder layer.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor, optional): Optional attention mask.
        Returns:
            Tensor: Output of the encoder layer with the same shape as the input.
        """
        
        x = self.norm1(x + self.dropout1(self.attn(x, mask)))
        x = self.norm2(x + self.dropout2(self.ff(x)))
        return x

class DecoderLayer(nn.Module):
    """
    A single layer of the Transformer decoder. It consists of two attention mechanisms (self-attention 
    and encoder-decoder attention) followed by a feed-forward network. Residual connections and 
    layer normalization are applied after each sub-layer.
    Args:
        d_model (int): Dimension of the model.
        n_head (int): Number of attention heads.
        attn_type (str): Type of attention.
        ff_hidnum (int): Hidden layer size of the feed-forward network.
        drop_out (float): Dropout rate.
    """
    
    def __init__(self, d_model, n_head, attn_type, ff_hidnum, drop_out=0.1):
        super().__init__()
        self.attn1 = SelfAttention(d_model, n_head, attn_type)
        self.attn2 = SelfAttention(d_model, n_head, attn_type)
        self.ff = FeedForward(d_model, ff_hidnum)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)
        self.dropout3 = nn.Dropout(drop_out)
   
    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        """
        Forward pass through a single decoder layer.
        Args:
            x (Tensor): Target sequence input.
            memory (Tensor): Output from the encoder.
            src_mask (Tensor): Mask for the source sequence.
            tgt_mask (Tensor): Mask for the target sequence.
        Returns:
            Tensor: Output of the decoder layer with the same shape as the input.
        """
        
        x = self.norm1(x + self.dropout1(self.attn1(x, tgt_mask)))
        x = self.norm2(x + self.dropout2(self.attn2(x, src_mask)))
        x = self.norm3(x + self.dropout3(self.ff(x)))
        return x

class Encoder(nn.Module):
    """
    The Transformer encoder, which is composed of multiple stacked encoder layers.
    Args:
        N (int): Number of encoder layers.
        d_model (int): Dimension of the model.
        n_head (int): Number of attention heads.
        attn_type (str): Type of attention.
        ff_hidnum (int): Hidden layer size of the feed-forward network.
        drop_out (float): Dropout rate.
    """
    
    def __init__(self, N, d_model, n_head, attn_type, ff_hidnum, drop_out=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_head, attn_type, ff_hidnum, drop_out) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        """
        Forward pass through the encoder.
        Args:
            x (Tensor): Input sequence tensor.
        Returns:
            Tensor: Final output of the encoder.
        """
        
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    """
    The Transformer decoder, which is composed of multiple stacked decoder layers.
    Args:
        N (int): Number of decoder layers.
        d_model (int): Dimension of the model.
        n_head (int): Number of attention heads.
        attn_type (str): Type of attention.
        ff_hidnum (int): Hidden layer size of the feed-forward network.
        drop_out (float): Dropout rate.
    """
    
    def __init__(self, N, d_model, n_head, attn_type, ff_hidnum, drop_out=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, attn_type, ff_hidnum, drop_out) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        """
        Forward pass through the decoder.
        Args:
            x (Tensor): Target sequence tensor.
            memory (Tensor): Encoder output (source sequence representation).
            src_mask (Tensor): Mask for the source sequence.
            tgt_mask (Tensor): Mask for the target sequence.
        Returns:
            Tensor: Final output of the decoder.
        """
        
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class Transformer(nn.Module):
    """
    Full Transformer model combining the encoder and decoder, with support for checkpointing to 
    save memory during training.
    Args:
        device (str): Device on which the model will be run ('cpu' or 'cuda').
        d_model (int): Dimension of the model.
        in_dim_enc (int): Input dimension for the encoder.
        in_dim_dec (int): Input dimension for the decoder.
        attn_type (str): Type of attention mechanism.
        N_enc (int): Number of encoder layers.
        N_dec (int): Number of decoder layers.
        h_enc (int): Number of attention heads in the encoder.
        h_dec (int): Number of attention heads in the decoder.
        ff_hidnum (int): Hidden layer size of the feed-forward network.
        hid_pre (int): Dimension for the pre-processing layers.
        hid_post (int): Dimension for the post-processing layers.
        dropout_pre (float): Dropout rate for the pre-processing layer.
        dropout_post (float): Dropout rate for the post-processing layer.
        dropout_model (float): Dropout rate for the encoder and decoder.
        use_checkpoint (bool): Whether to use gradient checkpointing to save memory.
    """
    
    def __init__(self, device, d_model, in_dim_enc, in_dim_dec, attn_type, N_enc, N_dec, h_enc, h_dec,
                 ff_hidnum, hid_pre, hid_post, dropout_pre, dropout_post, dropout_model, use_checkpoint=False):
        super().__init__()
        self.device = device
        self.use_checkpoint = use_checkpoint
        self.x_pre = PreLayer(hid_pre, d_model, dropout_pre, in_dim_enc)
        self.y_pre = PreLayer(hid_pre, d_model, dropout_pre, in_dim_dec)
        self.pos = PositionalEncoding(d_model, device=device)
        self.enc = Encoder(N_enc, d_model, h_enc, attn_type, ff_hidnum, dropout_model)
        self.dec = Decoder(N_dec, d_model, h_dec, attn_type, ff_hidnum, dropout_model)
        self.post = PostLayer(d_model, 1, hid_post, dropout_post)
   
    def forward(self, x, y):
        """
        Forward pass through the entire Transformer model.
        Args:
            x (Tensor): Input sequence for the encoder.
            y (Tensor): Target sequence for the decoder.
        Returns:
            Tensor: Final output of the model after the decoder.
        """
        
        x_emb = self.x_pre(x)
        y_emb = self.y_pre(y)
        x_emb_pos = self.pos(x_emb)
        y_emb_pos = self.pos(y_emb)
        
        if self.use_checkpoint and self.training:
            memory = checkpoint(self.enc, x_emb_pos, use_reentrant=True)
            out = checkpoint(self.dec, y_emb_pos, memory, use_reentrant=True)
        else:
            memory = self.enc(x_emb_pos)
            out = self.dec(y_emb_pos, memory)
        out = self.post(out)
        out = out.squeeze(-1)
        return out

def define_model(device, use_checkpoint=False):
    """
    Utility function to create and define the Transformer model with specified parameters.
    Args:
        device (str): Device on which the model will be run ('cpu' or 'cuda').
        use_checkpoint (bool): Whether to use gradient checkpointing to save memory during training.
    Returns:
        Transformer: The instantiated Transformer model.
    """
    
    model = Transformer(
        device=device,
        in_dim_enc=3,
        in_dim_dec=1,
        d_model=512,
        N_enc=4,
        N_dec=4,
        h_enc=8,
        h_dec=8,
        ff_hidnum=1024,
        hid_pre=16,
        hid_post=8,
        dropout_pre=0.0,
        dropout_post=0.0,
        dropout_model=0.0,
        attn_type='full',
        use_checkpoint=use_checkpoint
    ).to(device)
    
    return model
