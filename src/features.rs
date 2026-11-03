use crate::env::SnakeEnv;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FeatureEncoding {
    #[default]
    HeadBodyFood,
}

impl FeatureEncoding {
    pub fn channels(self) -> usize {
        match self {
            Self::HeadBodyFood => 3,
        }
    }

    pub fn observation_size(self, width: usize, height: usize) -> usize {
        self.channels() * width * height
    }

    pub fn encode(self, env: &SnakeEnv) -> Vec<f32> {
        match self {
            Self::HeadBodyFood => env.observation(),
        }
    }
}
