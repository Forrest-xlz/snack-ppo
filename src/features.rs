use crate::env::{Point, SnakeEnv};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FeatureEncoding {
    #[default]
    HeadBodyFood,
    HeadBodyFoodGeometryOrder,
    GridRich31,
    EntityBase,
    EntityRegions,
}

impl FeatureEncoding {
    pub fn channels(self) -> usize {
        match self {
            Self::HeadBodyFood => 3,
            Self::HeadBodyFoodGeometryOrder => 7,
            Self::GridRich31 => 31,
            Self::EntityBase => crate::entity_features::CELL_BASE,
            Self::EntityRegions => crate::entity_features::CELL_REGIONS,
        }
    }

    pub fn observation_size(self, width: usize, height: usize) -> usize {
        self.channels() * width * height
            + if matches!(self, Self::EntityBase | Self::EntityRegions) {
                crate::entity_features::CONTEXT
            } else {
                0
            }
    }

    pub fn is_prefix_of(self, student: Self) -> bool {
        self == student
            || (self == Self::HeadBodyFood
                && !matches!(student, Self::EntityBase | Self::EntityRegions))
    }

    pub fn encode_into(self, env: &SnakeEnv, output: &mut [f32]) {
        assert_eq!(
            output.len(),
            self.observation_size(env.width(), env.height())
        );
        match self {
            Self::EntityBase => crate::entity_features::encode_into(env, false, output),
            Self::EntityRegions => crate::entity_features::encode_into(env, true, output),
            _ => output.copy_from_slice(&self.encode(env)),
        }
    }

    pub fn encode(self, env: &SnakeEnv) -> Vec<f32> {
        match self {
            Self::EntityBase => crate::entity_features::encode(env, false),
            Self::EntityRegions => crate::entity_features::encode(env, true),
            Self::HeadBodyFood => env.observation(),
            Self::HeadBodyFoodGeometryOrder => {
                let width = env.width();
                let area = width * env.height();
                let mut features = env.observation();
                features.resize(self.channels() * area, 0.0);
                let x_scale = width.saturating_sub(1).max(1) as f32;
                let y_scale = env.height().saturating_sub(1).max(1) as f32;
                for cell in 0..area {
                    features[3 * area + cell] = (cell % width) as f32 / x_scale;
                    features[4 * area + cell] = (cell / width) as f32 / y_scale;
                    let occupied =
                        features[cell] + features[area + cell] + features[2 * area + cell];
                    features[5 * area + cell] = if occupied == 0.0 { 1.0 } else { 0.0 };
                }
                let distance_scale = area.saturating_sub(1).max(1) as f32;
                for (distance, point) in env.snake().iter().rev().enumerate() {
                    features[6 * area + point.y * width + point.x] =
                        distance as f32 / distance_scale;
                }
                features
            }
            Self::GridRich31 => {
                let width = env.width();
                let height = env.height();
                let area = width * height;
                let mut f = env.observation();
                f.resize(31 * area, 0.0);
                let sx = width.saturating_sub(1).max(1) as f32;
                let sy = height.saturating_sub(1).max(1) as f32;
                let head = env.head();
                for cell in 0..area {
                    let x = cell % width;
                    let y = cell / width;
                    let occupied = f[cell] + f[area + cell] + f[2 * area + cell];
                    f[3 * area + cell] = if occupied == 0.0 { 1.0 } else { 0.0 };
                    f[4 * area + cell] = x as f32 / sx;
                    f[5 * area + cell] = y as f32 / sy;
                    f[6 * area + cell] = (x as f32 - head.x as f32) / sx;
                    f[7 * area + cell] = (y as f32 - head.y as f32) / sy;
                    f[19 * area + cell] = x as f32 / sx;
                    f[20 * area + cell] = (width - 1 - x) as f32 / sx;
                    f[21 * area + cell] = y as f32 / sy;
                    f[22 * area + cell] = (height - 1 - y) as f32 / sy;
                }
                let snake = env.snake();
                let log_scale = (area as f32).ln_1p();
                for (index, &point) in snake.iter().enumerate() {
                    let cell = point.y * width + point.x;
                    let t = snake.len() - index;
                    f[8 * area + cell] = if index + 1 == snake.len() { 1.0 } else { 0.0 };
                    f[9 * area + cell] = t as f32 / area as f32;
                    f[10 * area + cell] = (t as f32).ln_1p() / log_scale;
                    if t <= 8 {
                        f[(10 + t) * area + cell] = 1.0;
                    }
                    if index > 0 {
                        fill_chain_direction(&mut f, area, 23, cell, point, snake[index - 1]);
                    }
                    if index + 1 < snake.len() {
                        fill_chain_direction(&mut f, area, 27, cell, point, snake[index + 1]);
                    }
                }
                f
            }
        }
    }
}

fn fill_chain_direction(
    features: &mut [f32],
    area: usize,
    base: usize,
    cell: usize,
    point: Point,
    target: Point,
) {
    let delta = (
        target.x as isize - point.x as isize,
        target.y as isize - point.y as isize,
    );
    let direction = match delta {
        (0, -1) => Some(0),
        (0, 1) => Some(1),
        (-1, 0) => Some(2),
        (1, 0) => Some(3),
        _ => None,
    };
    if let Some(direction) = direction {
        features[(base + direction) * area + cell] = 1.0;
    }
}
