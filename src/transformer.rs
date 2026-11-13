use tch::{nn, Device, Kind, Tensor};

const HEADS: i64 = 2;

fn attention_kind(device: Device, input_kind: Kind) -> Kind {
    if device.is_cuda() {
        Kind::BFloat16
    } else {
        input_kind
    }
}

pub(crate) fn attention_backend_for_shape(
    device: Device,
    batch: i64,
    tokens: i64,
    training: bool,
    full_queries: bool,
    dim: i64,
) -> String {
    let kind = attention_kind(device, Kind::Float);
    // Match the real projection's strided K/V views and selected Q gather,
    // including dtype conversion: backend constraints can depend on strides.
    let qkv = Tensor::zeros(
        [batch, tokens, 3, HEADS, dim / HEADS],
        (Kind::Float, device),
    )
    .set_requires_grad(training)
    .permute([2, 0, 3, 1, 4]);
    let indices = Tensor::from_slice(&[0i64, tokens - 1])
        .to_device(device)
        .view([1, 1, 2, 1])
        .expand([batch, HEADS, 2, dim / HEADS], true);
    let query = if full_queries {
        qkv.get(0)
    } else {
        qkv.get(0).gather(2, &indices, false)
    }
    .to_kind(kind);
    let key = qkv.get(1).to_kind(kind);
    let value = qkv.get(2).to_kind(kind);
    let backend = Tensor::internal_fused_sdp_choice(
        &query,
        &key,
        &value,
        None::<&Tensor>,
        0.0,
        false,
        None,
        false,
    );
    let name = match backend {
        0 => "math",
        1 => "flash_attention",
        2 => "memory_efficient",
        3 => "cudnn_attention",
        _ => "other",
    };
    let queries = if full_queries { tokens } else { 2 };
    format!(
        "{name}(id={backend},dtype={kind:?},batch={batch},queries={queries},training={training})"
    )
}

pub(crate) struct EncoderBlock {
    dim: i64,
    attention_norm: nn::LayerNorm,
    qkv: nn::Linear,
    attention_out: nn::Linear,
    ff_norm: nn::LayerNorm,
    ff_in: nn::Linear,
    ff_out: nn::Linear,
}

impl EncoderBlock {
    pub(crate) fn with_dim(path: &nn::Path<'_>, dim: i64) -> Self {
        Self {
            dim,
            attention_norm: nn::layer_norm(path / "attention_norm", vec![dim], Default::default()),
            qkv: nn::linear(path / "qkv", dim, 3 * dim, Default::default()),
            attention_out: nn::linear(path / "attention_out", dim, dim, Default::default()),
            ff_norm: nn::layer_norm(path / "ff_norm", vec![dim], Default::default()),
            ff_in: nn::linear(path / "ff_in", dim, 2 * dim, Default::default()),
            ff_out: nn::linear(path / "ff_out", 2 * dim, dim, Default::default()),
        }
    }

    pub(crate) fn forward(&self, tokens: &Tensor, readout_indices: Option<&Tensor>) -> Tensor {
        let batch = tokens.size()[0];
        let length = tokens.size()[1];
        let qkv = tokens
            .apply(&self.attention_norm)
            .apply(&self.qkv)
            .view([batch, length, 3, HEADS, self.dim / HEADS])
            .permute([2, 0, 3, 1, 4]);
        let (queries, readout_tokens) = match readout_indices {
            Some(indices) => (
                qkv.get(0).gather(
                    2,
                    &indices
                        .view([batch, 1, 2, 1])
                        .expand([batch, HEADS, 2, self.dim / HEADS], true),
                    false,
                ),
                tokens.gather(
                    1,
                    &indices.unsqueeze(-1).expand([batch, 2, self.dim], true),
                    false,
                ),
            ),
            None => (qkv.get(0), tokens.shallow_clone()),
        };
        let output_length = queries.size()[2];
        let context = Tensor::scaled_dot_product_attention(
            &queries.to_kind(attention_kind(tokens.device(), tokens.kind())),
            &qkv.get(1)
                .to_kind(attention_kind(tokens.device(), tokens.kind())),
            &qkv.get(2)
                .to_kind(attention_kind(tokens.device(), tokens.kind())),
            None::<&Tensor>,
            0.0,
            false,
            None,
            false,
        )
        .to_kind(tokens.kind())
        .transpose(1, 2)
        .contiguous()
        .view([batch, output_length, self.dim]);
        let tokens = readout_tokens + context.apply(&self.attention_out);
        let ff = tokens
            .apply(&self.ff_norm)
            .apply(&self.ff_in)
            .gelu("none")
            .apply(&self.ff_out);
        tokens + ff
    }
}
