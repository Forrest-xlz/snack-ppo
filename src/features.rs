use crate::env::SnakeEnv;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FeatureEncoding {
    #[default]
    HeadBodyFood,
    EntityRegions,
}

impl FeatureEncoding {
    pub fn channels(self) -> usize {
        match self {
            Self::HeadBodyFood => 3,
            Self::EntityRegions => crate::entity_features::CELL_REGIONS,
        }
    }

    pub fn observation_size(self, width: usize, height: usize) -> usize {
        self.channels() * width * height
            + if self == Self::EntityRegions {
                crate::entity_features::CONTEXT
            } else {
                0
            }
    }

    pub fn is_prefix_of(self, student: Self) -> bool {
        self == student
    }

    pub fn encode_into(self, env: &SnakeEnv, output: &mut [f32]) {
        assert_eq!(
            output.len(),
            self.observation_size(env.width(), env.height())
        );
        match self {
            Self::EntityRegions => crate::entity_features::encode_into(env, output),
            Self::HeadBodyFood => output.copy_from_slice(&env.observation()),
        }
    }

    pub fn encode(self, env: &SnakeEnv) -> Vec<f32> {
        match self {
            Self::EntityRegions => crate::entity_features::encode(env),
            Self::HeadBodyFood => env.observation(),
        }
    }
}
