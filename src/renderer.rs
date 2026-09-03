use anyhow::Result;
use macroquad::prelude::*;
use tch::{nn, no_grad, Device, Tensor};

use crate::{
    config::Config,
    env::SnakeEnv,
    model::ActorCritic,
    trainer::{load_checkpoint, resolve_device, resolve_inference_checkpoint},
};

pub fn run_inference(config: Config) -> Result<()> {
    let device = resolve_device(config.device)?;
    let checkpoint_path = resolve_inference_checkpoint(
        &config.inference.model_path,
        &config.training.checkpoint_dir,
        config.wandb.run_name.as_deref(),
    );
    if !checkpoint_path.exists() {
        anyhow::bail!("checkpoint does not exist: {}", checkpoint_path.display());
    }
    let mut store = nn::VarStore::new(device);
    let input_size = (3 * config.environment.width * config.environment.height) as i64;
    let model = ActorCritic::new(&store.root(), input_size, config.model.hidden_size);
    load_checkpoint(&mut store, &checkpoint_path)?;
    let window = Conf {
        window_title: "PPO Snake".to_owned(),
        window_width: (config.environment.width as u32 * config.inference.cell_size) as i32,
        window_height: (config.environment.height as u32 * config.inference.cell_size + 64) as i32,
        high_dpi: true,
        ..Default::default()
    };
    macroquad::Window::from_config(window, inference_loop(config, device, store, model));
    Ok(())
}

async fn inference_loop(config: Config, device: Device, _store: nn::VarStore, model: ActorCritic) {
    let mut env = SnakeEnv::new(
        config.environment.width,
        config.environment.height,
        config.seed,
    )
    .expect("validated environment dimensions");
    let mut best_score = 0u32;
    let mut accumulator = 0.0f32;
    let interval = 1.0 / config.inference.fps as f32;
    let mut terminal_pause = 0.0f32;

    loop {
        let frame_time = get_frame_time();
        accumulator += frame_time;
        if terminal_pause > 0.0 {
            terminal_pause -= frame_time;
            if terminal_pause <= 0.0 {
                env.reset();
            }
        } else if accumulator >= interval {
            accumulator %= interval;
            let observation = env.observation();
            let tensor = Tensor::from_slice(&observation)
                .view([1, observation.len() as i64])
                .to_device(device);
            let output =
                no_grad(|| model.actions_and_values(&tensor, config.inference.deterministic));
            let action = output.actions.int64_value(&[0]);
            let result = env.step(crate::env::Action::from_index(action).expect("model action"));
            best_score = best_score.max(result.score);
            if result.done {
                terminal_pause = 0.7;
            }
        }

        clear_background(Color::from_rgba(18, 22, 30, 255));
        draw_board(&env, config.inference.cell_size as f32);
        let text_y = config.environment.height as f32 * config.inference.cell_size as f32 + 42.0;
        draw_text(
            format!("Score: {}    Best: {}", env.score(), best_score),
            14.0,
            text_y,
            30.0,
            WHITE,
        );
        next_frame().await;
    }
}

fn draw_board(env: &SnakeEnv, cell: f32) {
    for y in 0..env.height() {
        for x in 0..env.width() {
            draw_rectangle_lines(x as f32 * cell, y as f32 * cell, cell, cell, 1.0, DARKGRAY);
        }
    }
    if let Some(food) = env.food() {
        draw_circle(
            food.x as f32 * cell + cell / 2.0,
            food.y as f32 * cell + cell / 2.0,
            cell * 0.34,
            RED,
        );
    }
    for (index, point) in env.snake().iter().enumerate().rev() {
        let color = if index == 0 { LIME } else { GREEN };
        draw_rectangle(
            point.x as f32 * cell + 2.0,
            point.y as f32 * cell + 2.0,
            cell - 4.0,
            cell - 4.0,
            color,
        );
    }
}
