use tch::{nn, Device, Tensor};

use crate::{
    entity_features::{CONTEXT, FOOD, GLOBAL, SNAKE},
    transformer::{attention_backend_for_shape, EncoderBlock},
};

pub(crate) struct EntityTransformer {
    cells: i64,
    channels: i64,
    dim: i64,
    cell_input: nn::Sequential,
    snake_input: nn::Sequential,
    food_input: nn::Sequential,
    global_input: nn::Sequential,
    positions: Tensor,
    types: Tensor,
    blocks: Vec<EncoderBlock>,
    final_norm: nn::LayerNorm,
    projection: nn::Linear,
}

fn input_mlp(path: nn::Path<'_>, input: i64, width: i64, dim: i64) -> nn::Sequential {
    nn::seq()
        .add(nn::linear(&path / "fc1", input, width, Default::default()))
        .add_fn(Tensor::relu)
        .add(nn::linear(&path / "fc2", width, dim, Default::default()))
        .add_fn(Tensor::relu)
}

impl EntityTransformer {
    pub(crate) fn new(
        path: &nn::Path<'_>,
        cells: i64,
        channels: i64,
        hidden: i64,
        compact: bool,
    ) -> Self {
        let (dim, width, depth) = if compact { (16, 32, 2) } else { (32, 64, 3) };
        let init = nn::Init::Randn {
            mean: 0.0,
            stdev: 0.02,
        };
        Self {
            cells,
            channels,
            dim,
            cell_input: input_mlp(path / "cell_input", channels, width, dim),
            snake_input: input_mlp(path / "snake_input", SNAKE as i64, width, dim),
            food_input: input_mlp(path / "food_input", FOOD as i64, width, dim),
            global_input: input_mlp(path / "global_input", GLOBAL as i64, width, dim),
            positions: path.var("positions", &[1, cells, dim], init),
            types: path.var("token_types", &[4, dim], init),
            blocks: (0..depth)
                .map(|index| {
                    let name = if index == 0 {
                        "encoder".into()
                    } else {
                        format!("encoder{}", index + 1)
                    };
                    EncoderBlock::with_dim(&(path / name), dim)
                })
                .collect(),
            final_norm: nn::layer_norm(path / "final_norm", vec![dim], Default::default()),
            projection: nn::linear(path / "projection", dim, hidden, Default::default()),
        }
    }

    fn tokens(&self, observations: &Tensor) -> Tensor {
        let batch = observations.size()[0];
        let offset = self.channels * self.cells;
        assert_eq!(
            observations.size()[1],
            offset + CONTEXT as i64,
            "entity observation must include Cell, Snake, Food and Global features"
        );
        let cells = observations
            .narrow(1, 0, offset)
            .view([batch, self.channels, self.cells])
            .transpose(1, 2)
            .apply(&self.cell_input)
            + &self.positions
            + self.types.get(0);
        let snake = observations
            .narrow(1, offset, SNAKE as i64)
            .apply(&self.snake_input)
            + self.types.get(1);
        let food = observations
            .narrow(1, offset + SNAKE as i64, FOOD as i64)
            .apply(&self.food_input)
            + self.types.get(2);
        let global = observations
            .narrow(1, offset + (SNAKE + FOOD) as i64, GLOBAL as i64)
            .apply(&self.global_input)
            + self.types.get(3);
        Tensor::cat(
            &[
                cells,
                snake.unsqueeze(1),
                food.unsqueeze(1),
                global.unsqueeze(1),
            ],
            1,
        )
    }

    pub(crate) fn forward(&self, observations: &Tensor) -> (Tensor, Tensor) {
        let batch = observations.size()[0];
        let mut tokens = self.tokens(observations);
        let indices = Tensor::from_slice(&[self.cells, self.cells + 2])
            .to_device(observations.device())
            .view([1, 2])
            .expand([batch, 2], true);
        for (index, block) in self.blocks.iter().enumerate() {
            tokens = block.forward(
                &tokens,
                (index + 1 == self.blocks.len()).then_some(&indices),
            );
        }
        let tokens = tokens.apply(&self.final_norm);
        (
            tokens.select(1, 0).apply(&self.projection).relu(),
            tokens.select(1, 1).apply(&self.projection).relu(),
        )
    }

    pub(crate) fn attention_backend(&self, device: Device, batch: i64, training: bool) -> String {
        (0..self.blocks.len())
            .map(|index| {
                format!(
                    "layer{}={}",
                    index + 1,
                    attention_backend_for_shape(
                        device,
                        batch,
                        self.cells + 3,
                        training,
                        index + 1 < self.blocks.len(),
                        self.dim,
                    )
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, Device, Kind};

    #[test]
    fn snake_global_pruning_matches_full_token_encoder_outputs_and_gradients() {
        for compact in [false, true] {
            let mut store = nn::VarStore::new(Device::Cpu);
            let model = EntityTransformer::new(&store.root(), 12, 66, 16, compact);
            store.set_kind(Kind::Double);
            let input = Tensor::randn([2, 12 * 66 + 53], (Kind::Double, Device::Cpu))
                .set_requires_grad(true);
            let actual = model.forward(&input);
            let mut full = model.tokens(&input);
            for block in &model.blocks {
                full = block.forward(&full, None);
            }
            full = full.apply(&model.final_norm);
            let expected = (
                full.select(1, 12).apply(&model.projection).relu(),
                full.select(1, 14).apply(&model.projection).relu(),
            );
            assert!(actual.0.allclose(&expected.0, 1e-8, 1e-10, false));
            assert!(actual.1.allclose(&expected.1, 1e-8, 1e-10, false));
            let parameters = store.trainable_variables();
            let inputs = std::iter::once(&input)
                .chain(parameters.iter())
                .collect::<Vec<_>>();
            let loss = actual.0.square().sum(Kind::Double) + actual.1.square().sum(Kind::Double);
            let expected_loss =
                expected.0.square().sum(Kind::Double) + expected.1.square().sum(Kind::Double);
            let actual_grad = Tensor::run_backward(&[loss], &inputs, true, false);
            let expected_grad = Tensor::run_backward(&[expected_loss], &inputs, false, false);
            for (a, b) in actual_grad.iter().zip(&expected_grad) {
                let error = (a - b).abs().max().double_value(&[]);
                let scale = b.abs().max().double_value(&[]);
                assert!(
                    error <= 1e-10 + 1e-8 * scale,
                    "gradient error={error} scale={scale}"
                );
            }
        }
    }
}
