use std::collections::VecDeque;

use ppo_snake::env::{Action, Direction, Point, SnakeEnv};

fn env_with(
    snake: &[(usize, usize)],
    direction: Direction,
    food: Option<(usize, usize)>,
) -> SnakeEnv {
    SnakeEnv::from_state(
        5,
        5,
        snake
            .iter()
            .copied()
            .map(Point::from)
            .collect::<VecDeque<_>>(),
        direction,
        food.map(Point::from),
        7,
    )
    .unwrap()
}

#[test]
fn moves_one_cell_and_ignores_opposite_action() {
    let mut env = env_with(&[(2, 2), (1, 2), (0, 2)], Direction::Right, Some((4, 4)));
    let result = env.step(Action::Left);
    assert!(!result.done);
    assert_eq!(result.reward, 0.0);
    assert_eq!(env.head(), Point::new(3, 2));
    assert_eq!(env.direction(), Direction::Right);
}

#[test]
fn eating_food_grows_snake_and_rewards_exactly_one() {
    let mut env = env_with(&[(2, 2), (1, 2), (0, 2)], Direction::Right, Some((3, 2)));
    let result = env.step(Action::Right);
    assert_eq!(result.reward, 1.0);
    assert_eq!(result.score, 1);
    assert_eq!(env.snake().len(), 4);
}

#[test]
fn wall_and_self_collision_end_with_zero_reward() {
    let mut wall = env_with(&[(4, 2), (3, 2), (2, 2)], Direction::Right, Some((0, 0)));
    let result = wall.step(Action::Right);
    assert!(result.done);
    assert_eq!(result.reward, 0.0);

    let mut body = env_with(
        &[(2, 2), (2, 3), (1, 3), (1, 2), (1, 1)],
        Direction::Up,
        Some((4, 4)),
    );
    let result = body.step(Action::Left);
    assert!(result.done);
    assert_eq!(result.reward, 0.0);
}

#[test]
fn observation_uses_separate_head_body_and_food_channels() {
    let env = env_with(&[(2, 2), (1, 2), (0, 2)], Direction::Right, Some((4, 4)));
    let observation = env.observation();
    assert_eq!(observation.len(), 75);
    assert_eq!(observation.iter().filter(|&&x| x == 1.0).count(), 4);
    assert_eq!(observation[2 * 5 + 2], 1.0);
    assert_eq!(observation[25 + 2 * 5 + 1], 1.0);
    assert_eq!(observation[50 + 4 * 5 + 4], 1.0);
}

#[test]
fn food_never_overlaps_snake_after_many_resets() {
    let mut env = SnakeEnv::new(5, 5, 123).unwrap();
    for _ in 0..100 {
        env.reset();
        let food = env.food().expect("non-full board has food");
        assert!(!env.snake().contains(&food));
    }
}
