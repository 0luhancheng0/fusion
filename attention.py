import dgl.transforms as T
import lightning as L
from torchmetrics.functional.classification import multiclass_accuracy
from torch import Tensor
from jaxtyping import Float
import dgl
from torch import nn
import dgl.sparse as dglsp
from dataloading import load_ogbn_arxiv
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch
from torch.optim import AdamW
from transformer_lens import FactoredMatrix
from einops import rearrange





class SparseMHA(nn.Module):
    """Sparse Multi-head Attention Module"""

    def __init__(self, hidden_size=128, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_head = hidden_size // num_heads
        self.scaling = self.d_head**-0.5

        self.q_proj: Float[Tensor, "d_model d_model"] = nn.Linear(
            hidden_size, hidden_size
        )
        self.k_proj: Float[Tensor, "d_model d_model"] = nn.Linear(
            hidden_size, hidden_size
        )
        self.v_proj: Float[Tensor, "d_model d_model"] = nn.Linear(
            hidden_size, hidden_size
        )
        self.out_proj: Float[Tensor, "d_model d_model"] = nn.Linear(
            hidden_size, hidden_size
        )

    def forward(self, A, h):
        N = len(h)
        # [N, dh, nh]
        q = self.q_proj(h).reshape(N, self.d_head, self.num_heads)
        q *= self.scaling
        # [N, dh, nh]
        k = self.k_proj(h).reshape(N, self.d_head, self.num_heads)
        # [N, dh, nh]
        v = self.v_proj(h).reshape(N, self.d_head, self.num_heads)

        attn = dglsp.bsddmm(A, q, k.transpose(1, 0))  # (sparse) [N, N, nh]
        attn = attn.softmax()  # (sparse) [N, N, nh]
        out = dglsp.bspmm(attn, v)  # [N, dh, nh]

        return self.out_proj(out.reshape(N, -1))


class GTLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, hidden_size=128, num_heads=8):
        super().__init__()
        self.MHA = SparseMHA(hidden_size=hidden_size, num_heads=num_heads)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        # self.FFN1 = nn.Linear(hidden_size, hidden_size * 2)
        self.mlp = nn.Linear(hidden_size, hidden_size)

    def forward(self, A, residual):

        h1 = self.layernorm1(residual)
        h1 = self.MHA(A, h1)

        residual = residual + h1

        h2 = self.layernorm2(residual)
        h2 = F.gelu(self.mlp(residual))
        residual = residual + h2

        return residual


class GTModel(L.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        out_size,
        num_layers=8,
        num_heads=8,
        pos_enc_size=2,
    ):
        super().__init__()
        # from dgl.nn.pytorch.sparse_emb import NodeEmbedding
        # self.embedding = nn.ModuleList([NodeEmbedding(
        #     num_embeddings=num_embeddings,
        #     embedding_dim=hidden_size,
        #     name="node_embedding",
        # )])
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.layers = nn.ModuleList(
            [GTLayer(hidden_size, num_heads) for _ in range(num_layers)]
        )
        # self.pooler = dglnn.SumPooling()
        self.pos_linear = nn.Linear(pos_enc_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, out_size)
        self.final_ln = nn.LayerNorm(hidden_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, A, X, pos):
        # h = self.embedding(X)
        h = self.input_layer(X)
        h = h + self.pos_linear(pos)
        for layer in self.layers:
            h = layer(A, h)
        h = self.final_ln(h)
        return self.classifier(h)

    def _step(self, A, feats, pos, labels, masks, stage):
        logits = self(A, feats, pos)
        labels = labels[masks]
        logits = logits[masks]
        loss = self.loss_fn(logits, labels)
        acc = multiclass_accuracy(logits.argmax(dim=-1), labels, num_classes=40)
        self.log(f"{stage}_loss", loss)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, _):
        A, feats, pos, labels, masks = batch[0]
        return self._step(A, feats, pos, labels, masks, "train")

    def validation_step(self, batch, _):
        A, feats, pos, labels, masks = batch[0]
        return self._step(A, feats, pos, labels, masks, "val")

    @property
    def W_K(self):
        return torch.stack(
            [self.layers[i].MHA.k_proj.weight for i in range(len(self.layers))]
        )

    @property
    def W_Q(self):
        return torch.stack(
            [self.layers[i].MHA.q_proj.weight for i in range(len(self.layers))]
        )

    @property
    def W_V(self):
        return torch.stack(
            [self.layers[i].MHA.v_proj.weight for i in range(len(self.layers))]
        )

    @property
    def W_O(self):
        return torch.stack(
            [self.layers[i].MHA.out_proj.weight for i in range(len(self.layers))]
        )

    def QK(self):
        W_K_transpose = rearrange(
            self.W_K, "head_index d_model d_head -> head_index d_head d_model"
        )
        return FactoredMatrix(self.W_Q, W_K_transpose)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-2, weight_decay=1e-4)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer=optimizer, step_size=100, gamma=0.5
                ),
                "interval": "step",
            },
        }



k = 2
dataset = dgl.data.CoraGraphDataset()
graph = dataset[0]

model = GTModel(
    1433, 128, 7, num_layers=4, num_heads=4, pos_enc_size=k
)

graph.ndata['pos_embedding'] = dgl.lap_pe(graph, k=k)

train_dataloader = DataLoader(
    [
        (
            graph.adj(),
            graph.ndata["feat"],
            graph.ndata["pos_embedding"],
            graph.ndata["label"],
            graph.ndata["train_mask"],
        )
    ],
    batch_size=1,
    collate_fn=lambda x: x,
)

val_dataloader = DataLoader(
    [
        (
            graph.adj(),
            graph.ndata["feat"],
            graph.ndata["pos_embedding"],
            graph.ndata["label"],
            graph.ndata["val_mask"],
        )
    ],
    batch_size=1,
    collate_fn=lambda x: x,
)


trainer = L.Trainer(max_epochs=50)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


