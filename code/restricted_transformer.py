class Embedding(nn.Module):
  def __init__(self, emb_size, vocab_size):
    super().__init__()
    self.embed = nn.Embedding(vocab_size,emb_size)

  def forward(self,x):
    return self.embed(x)  #x(batch,seq_len, emb_size)

class Attention(nn.Module):
  def __init__(self,emb_size, saturated=False):
    super().__init__()

    self.q = nn.Linear(emb_size,emb_size,bias =False)
    self.k = nn.Linear(emb_size,emb_size,bias =False)
    self.v = nn.Linear(emb_size,emb_size,bias =False)
    self.emb_size = emb_size
    self.saturated = saturated


  def saturated_attention(self,logits):

    max_vals = logits.max(dim=-1,keepdim=True).values
    output =  (logits==max_vals).float()
    return output/output.sum(dim=-1,keepdim=True)


  def attention(self,q,k,v):

      att_logits = (q @ k.transpose(-2,-1)) / self.emb_size**0.5  # (batch,q_len, k_len)

      if self.saturated:
        att_weight = self.saturated_attention(att_logits)
        
      else:

        att_weight =  F.softmax(att_logits,dim=-1)
      att_output = att_weight @ v

      return att_output, att_weight


  def forward(self,x):
      query = self.q(x)  # (batch,q_len, emb_size)
      key = self.k(x)   # (batch,k_len, emb_size)
      value = self.v(x) # (batch,k_len, emb_size)
      att_output, att_weight = self.attention(query,key,value)

      return att_output,att_weight

class Classification_head(nn.Module):
  def __init__(self,emb_size,num_classes =1):
    super().__init__()

    self.out_layer = nn.Linear(emb_size,num_classes)

  def forward(self,x):
    return self.out_layer(x)

class Transformer(nn.Module):
  def __init__(self,emb_size,vocab_size,num_classes =1):
    super().__init__()

    self.embed = Embedding(emb_size,vocab_size)
    self.attention = Attention(emb_size)
    self.head = Classification_head(emb_size, num_classes)

  def forward(self,x):
    x = self.embed(x)
    x,att_weight = self.attention(x)
    x = x.mean(dim = -2)
    x = self.head(x)
    return x,att_weight
