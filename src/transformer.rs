use tch::{nn, Device, Kind, Tensor};

const DIM: i64 = 32;
const HEADS: i64 = 2;

fn attention_kind(device: Device, input_kind: Kind) -> Kind {
    if device.is_cuda() {
        Kind::BFloat16
    } else {
        input_kind
    }
}

/// Query the installed runtime's selection rather than calling every SDPA
/// invocation "FlashAttention". The real forward uses the same dtype/layout.
fn attention_backend(
    device: Device,
    batch: i64,
    cells: i64,
    training: bool,
    full_queries: bool,
) -> String {
    attention_backend_for_tokens(device, batch, cells + 1, training, full_queries)
}

pub(crate) fn attention_backend_for_tokens(
    device: Device,
    batch: i64,
    tokens: i64,
    training: bool,
    full_queries: bool,
) -> String {
    attention_backend_for_shape(device, batch, tokens, training, full_queries, DIM)
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

/// Pre-norm, bidirectional encoder blocks. Every grid cell participates;
/// the additional learned token provides a separate critic readout.
pub(crate) struct TransformerEncoder {
    cells: i64,
    channels: i64,
    embedding: nn::Sequential,
    positions: Tensor,
    critic_token: Tensor,
    blocks: Vec<EncoderBlock>,
    final_norm: nn::LayerNorm,
    projection: nn::Linear,
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

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind};

    fn assert_gradient_close(actual: &Tensor, expected: &Tensor, name: &str) {
        // Use max-norm relative error near cancellation-zero components.
        let error = (actual - expected).abs().max().double_value(&[]);
        let scale = expected.abs().max().double_value(&[]);
        assert!(
            error <= 1e-10 + 1e-8 * scale,
            "{name}: max_abs_error={error} reference_max={scale}"
        );
    }

    // Literal full-token encoder reference, independent of readout pruning.
    fn full_reference(model: &TransformerEncoder, observations: &Tensor) -> (Tensor, Tensor) {
        let batch = observations.size()[0];
        let length = model.cells + 1;
        let cells = observations
            .view([batch, model.channels, model.cells])
            .transpose(1, 2);
        let mut tokens = Tensor::cat(
            &[
                cells.apply(&model.embedding) + &model.positions,
                model.critic_token.expand([batch, 1, DIM], true),
            ],
            1,
        );
        for block in &model.blocks {
            let qkv = tokens
                .apply(&block.attention_norm)
                .apply(&block.qkv)
                .view([batch, length, 3, HEADS, DIM / HEADS])
                .permute([2, 0, 3, 1, 4]);
            let context = Tensor::scaled_dot_product_attention(
                &qkv.get(0),
                &qkv.get(1),
                &qkv.get(2),
                None::<&Tensor>,
                0.0,
                false,
                None,
                false,
            )
            .transpose(1, 2)
            .contiguous()
            .view([batch, length, DIM]);
            let attended = &tokens + context.apply(&block.attention_out);
            let ff = attended
                .apply(&block.ff_norm)
                .apply(&block.ff_in)
                .gelu("none")
                .apply(&block.ff_out);
            tokens = attended + ff;
        }
        let tokens = tokens.apply(&model.final_norm);
        let index = observations
            .narrow(1, 0, model.cells)
            .argmax(-1, false)
            .view([batch, 1, 1])
            .expand([batch, 1, DIM], true);
        let head = tokens.gather(1, &index, false).squeeze_dim(1);
        let critic = tokens.select(1, model.cells);
        (
            head.apply(&model.projection).relu(),
            critic.apply(&model.projection).relu(),
        )
    }

    #[test]
    fn selected_queries_match_full_encoder_outputs_and_gradients() {
        for (channels, depth, mlp_input) in
            [(3, 1, false), (7, 1, false), (7, 2, false), (31, 3, true)]
        {
            // Check algebraic pruning equivalence in Float64, independent of
            // different FP32 query tilings/reduction orders. Real FP32/BF16
            // training and gradients are exercised by CLI/CUDA tests.
            let mut store = nn::VarStore::new(Device::Cpu);
            store.set_kind(Kind::Double);
            let model = TransformerEncoder::new(&store.root(), 35, channels, 16, depth, mlp_input);
            let mut reference_store = nn::VarStore::new(Device::Cpu);
            reference_store.set_kind(Kind::Double);
            let reference = TransformerEncoder::new(
                &reference_store.root(),
                35,
                channels,
                16,
                depth,
                mlp_input,
            );
            reference_store.copy(&store).unwrap();
            let observations = Tensor::rand([3, channels * 35], (Kind::Double, Device::Cpu))
                .set_requires_grad(true);
            let reference_observations = observations.detach().set_requires_grad(true);
            let (head, critic) = model.forward(&observations);
            let (reference_head, reference_critic) =
                full_reference(&reference, &reference_observations);
            assert!(head.allclose(&reference_head, 1e-8, 1e-10, false));
            assert!(critic.allclose(&reference_critic, 1e-8, 1e-10, false));
            (head.square().sum(Kind::Double) + critic.square().sum(Kind::Double)).backward();
            (reference_head.square().sum(Kind::Double)
                + reference_critic.square().sum(Kind::Double))
            .backward();
            assert_gradient_close(
                &observations.grad(),
                &reference_observations.grad(),
                "input",
            );
            for (name, variable) in store.variables() {
                assert_gradient_close(
                    &variable.grad(),
                    &reference_store.variables()[&name].grad(),
                    &format!("channels={channels} depth={depth} {name}"),
                );
            }
        }
    }
}

impl TransformerEncoder {
    pub(crate) fn new(
        path: &nn::Path<'_>,
        cells: i64,
        channels: i64,
        hidden_size: i64,
        depth: usize,
        mlp_input: bool,
    ) -> Self {
        assert!(depth > 0);
        let init = nn::Init::Randn {
            mean: 0.0,
            stdev: 0.02,
        };
        // Keep the one-layer versions' parameter names and initialization order.
        let mut embedding = nn::seq().add(nn::linear(
            path / "token_embedding",
            channels,
            DIM,
            Default::default(),
        ));
        if mlp_input {
            embedding = embedding
                .add_fn(Tensor::relu)
                .add(nn::linear(
                    path / "token_embedding2",
                    DIM,
                    DIM,
                    Default::default(),
                ))
                .add_fn(Tensor::relu);
        }
        let positions = path.var("positions", &[1, cells, DIM], init);
        let critic_token = path.var("critic_token", &[1, 1, DIM], init);
        let blocks = (0..depth)
            .map(|index| {
                let name = if index == 0 {
                    "encoder".to_owned()
                } else {
                    format!("encoder{}", index + 1)
                };
                EncoderBlock::new(&(path / name))
            })
            .collect();
        Self {
            cells,
            channels,
            embedding,
            positions,
            critic_token,
            blocks,
            final_norm: nn::layer_norm(path / "final_norm", vec![DIM], Default::default()),
            projection: nn::linear(path / "projection", DIM, hidden_size, Default::default()),
        }
    }

    pub(crate) fn attention_backend(
        &self,
        device: Device,
        batch: i64,
        cells: i64,
        training: bool,
    ) -> String {
        (0..self.blocks.len())
            .map(|index| {
                format!(
                    "layer{}={}",
                    index + 1,
                    attention_backend(
                        device,
                        batch,
                        cells,
                        training,
                        index + 1 < self.blocks.len()
                    )
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    }

    pub(crate) fn forward(&self, observations: &Tensor) -> (Tensor, Tensor) {
        let batch = observations.size()[0];
        let cells = observations
            .view([batch, self.channels, self.cells])
            .transpose(1, 2);
        let head_indices = observations.narrow(1, 0, self.cells).argmax(-1, false);
        let embedded = cells.apply(&self.embedding) + &self.positions;
        let critic = self.critic_token.expand([batch, 1, DIM], true);
        let mut tokens = Tensor::cat(&[embedded, critic], 1);
        let readout_indices = Tensor::stack(
            &[
                head_indices.shallow_clone(),
                head_indices.full_like(self.cells),
            ],
            1,
        );
        for (index, block) in self.blocks.iter().enumerate() {
            // Intermediate outputs become the next layer's K/V, so only the
            // final layer may discard unused output rows.
            let indices = (index + 1 == self.blocks.len()).then_some(&readout_indices);
            tokens = block.forward(&tokens, indices);
        }
        let tokens = tokens.apply(&self.final_norm);
        let head = tokens.select(1, 0);
        let critic = tokens.select(1, 1);
        (
            head.apply(&self.projection).relu(),
            critic.apply(&self.projection).relu(),
        )
    }
}

impl EncoderBlock {
    pub(crate) fn new(path: &nn::Path<'_>) -> Self {
        Self::with_dim(path, DIM)
    }

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
