use serde::{Deserialize, Serialize};

use crate::env::SnakeEnv;

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
pub enum FeatureVersion {
    #[default]
    #[serde(rename = "board_v1")]
    BoardV1,
}

impl FeatureVersion {
    pub fn channels(self) -> usize {
        match self {
            Self::BoardV1 => 3,
        }
    }

    pub fn observation_size(self, width: usize, height: usize) -> usize {
        self.channels() * width * height
    }

    pub fn encode(self, env: &SnakeEnv) -> Vec<f32> {
        match self {
            Self::BoardV1 => env.observation(),
        }
    }
}
