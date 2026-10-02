import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import SAGEConv

# --------------------------------------------------
# 1. Load interactions
# --------------------------------------------------
df = pd.read_csv("interactions.csv")

genes = pd.unique(df[["gene_a", "gene_b"]].values.ravel())
gene2id = {g: i for i, g in enumerate(genes)}

src = torch.tensor(df.gene_a.map(gene2id), dtype=torch.long)
dst = torch.tensor(df.gene_b.map(gene2id), dtype=torch.long)
w = torch.tensor(df.weight.values, dtype=torch.float32)

# Undirected graph
g = dgl.graph(
    (torch.cat([src, dst]), torch.cat([dst, src])),
    num_nodes=len(genes)
)

g.edata["weight"] = torch.cat([w, w])

# --------------------------------------------------
# 2. Node features
# --------------------------------------------------
# Learnable embedding for every gene
dim = 64

# --------------------------------------------------
# 3. GNN
# --------------------------------------------------
class GNN(nn.Module):
    def __init__(self, n_genes, dim=64):
        super().__init__()
        self.emb = nn.Embedding(n_genes, dim)
        self.conv1 = SAGEConv(dim, 128, "mean")
        self.conv2 = SAGEConv(128, dim, "mean")

    def forward(self, g):
        ids = torch.arange(g.num_nodes(), device=g.device)
        h = self.emb(ids)
        h = F.relu(self.conv1(g, h))
        return self.conv2(g, h)


# --------------------------------------------------
# 4. Edge predictor
# --------------------------------------------------
class Predictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, h, src, dst):
        return self.mlp(
            torch.cat([h[src], h[dst]], dim=1)
        ).squeeze()


# --------------------------------------------------
# 5. Negative samples
# --------------------------------------------------
def negative_edges(num_nodes, src, dst):
    n = len(src)
    neg_src = torch.randint(0, num_nodes, (n,))
    neg_dst = torch.randint(0, num_nodes, (n,))
    return neg_src, neg_dst


# --------------------------------------------------
# 6. Train
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

g = g.to(device)
src = src.to(device)
dst = dst.to(device)

model = GNN(g.num_nodes(), dim).to(device)
predictor = Predictor(dim).to(device)

optimizer = torch.optim.Adam(
    list(model.parameters()) +
    list(predictor.parameters()),
    lr=1e-3
)

for epoch in range(100):

    model.train()

    h = model(g)

    # Positive interactions
    pos_score = predictor(h, src, dst)

    # Random non-interacting pairs
    neg_src, neg_dst = negative_edges(
        g.num_nodes(), src, dst
    )

    neg_src = neg_src.to(device)
    neg_dst = neg_dst.to(device)

    neg_score = predictor(
        h, neg_src, neg_dst
    )

    # Binary classification
    scores = torch.cat([pos_score, neg_score])

    labels = torch.cat([
        torch.ones_like(pos_score),
        torch.zeros_like(neg_score)
    ])

    loss = F.binary_cross_entropy_with_logits(
        scores,
        labels
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(
            f"epoch={epoch:03d} "
            f"loss={loss.item():.4f}"
        )


# --------------------------------------------------
# 7. Predict an interaction
# --------------------------------------------------
model.eval()

with torch.no_grad():
    h = model(g)

def predict_gene_pair(gene_a, gene_b):
    a = torch.tensor([gene2id[gene_a]], device=device)
    b = torch.tensor([gene2id[gene_b]], device=device)

    score = predictor(h, a, b)
    return torch.sigmoid(score).item()


print(
    "TP53-BRCA1:",
    predict_gene_pair("TP53", "BRCA1")
)