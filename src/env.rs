use std::collections::VecDeque;

use anyhow::{bail, Result};
use rand::{seq::SliceRandom, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Point {
    pub x: usize,
    pub y: usize,
}

impl Point {
    pub const fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }
}

impl From<(usize, usize)> for Point {
    fn from((x, y): (usize, usize)) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Up,
    Down,
    Left,
    Right,
}

impl Action {
    pub fn from_index(value: i64) -> Result<Self> {
        match value {
            0 => Ok(Self::Up),
            1 => Ok(Self::Down),
            2 => Ok(Self::Left),
            3 => Ok(Self::Right),
            _ => bail!("action index must be in 0..4, got {value}"),
        }
    }
}

impl From<Action> for Direction {
    fn from(action: Action) -> Self {
        match action {
            Action::Up => Self::Up,
            Action::Down => Self::Down,
            Action::Left => Self::Left,
            Action::Right => Self::Right,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StepResult {
    pub reward: f32,
    pub done: bool,
    pub score: u32,
}

pub struct SnakeEnv {
    width: usize,
    height: usize,
    snake: VecDeque<Point>,
    direction: Direction,
    food: Option<Point>,
    score: u32,
    done: bool,
    rng: ChaCha8Rng,
}

impl SnakeEnv {
    pub fn new(width: usize, height: usize, seed: u64) -> Result<Self> {
        if width < 3 || height == 0 {
            bail!("board must be at least 3 cells wide and 1 cell high");
        }
        let mut env = Self {
            width,
            height,
            snake: VecDeque::new(),
            direction: Direction::Right,
            food: None,
            score: 0,
            done: false,
            rng: ChaCha8Rng::seed_from_u64(seed),
        };
        env.reset();
        Ok(env)
    }

    pub fn from_state(
        width: usize,
        height: usize,
        snake: VecDeque<Point>,
        direction: Direction,
        food: Option<Point>,
        seed: u64,
    ) -> Result<Self> {
        if width == 0 || height == 0 || snake.is_empty() {
            bail!("board and snake must not be empty");
        }
        if snake.iter().any(|p| p.x >= width || p.y >= height) {
            bail!("snake point is outside the board");
        }
        if food.is_some_and(|p| p.x >= width || p.y >= height || snake.contains(&p)) {
            bail!("food must be an unoccupied board cell");
        }
        Ok(Self {
            width,
            height,
            snake,
            direction,
            food,
            score: 0,
            done: false,
            rng: ChaCha8Rng::seed_from_u64(seed),
        })
    }

    pub fn reset(&mut self) {
        let head_x = self.width / 2;
        let y = self.height / 2;
        self.snake = VecDeque::from([
            Point::new(head_x, y),
            Point::new(head_x - 1, y),
            Point::new(head_x - 2, y),
        ]);
        self.direction = Direction::Right;
        self.score = 0;
        self.done = false;
        self.place_food();
    }

    pub fn step(&mut self, action: Action) -> StepResult {
        if self.done {
            return StepResult {
                reward: 0.0,
                done: true,
                score: self.score,
            };
        }
        let requested = Direction::from(action);
        if !is_opposite(self.direction, requested) {
            self.direction = requested;
        }
        let head = self.head();
        let next = match self.direction {
            Direction::Up if head.y > 0 => Some(Point::new(head.x, head.y - 1)),
            Direction::Down if head.y + 1 < self.height => Some(Point::new(head.x, head.y + 1)),
            Direction::Left if head.x > 0 => Some(Point::new(head.x - 1, head.y)),
            Direction::Right if head.x + 1 < self.width => Some(Point::new(head.x + 1, head.y)),
            _ => None,
        };
        let Some(next) = next else {
            self.done = true;
            return self.result(0.0);
        };
        let eating = self.food == Some(next);
        let occupied_len = self.snake.len() - usize::from(!eating);
        if self.snake.iter().take(occupied_len).any(|&p| p == next) {
            self.done = true;
            return self.result(0.0);
        }
        self.snake.push_front(next);
        if eating {
            self.score += 1;
            self.place_food();
            if self.food.is_none() {
                self.done = true;
            }
            self.result(1.0)
        } else {
            self.snake.pop_back();
            self.result(0.0)
        }
    }

    pub fn observation(&self) -> Vec<f32> {
        let plane = self.width * self.height;
        let mut result = vec![0.0; plane * 3];
        let head = self.head();
        result[head.y * self.width + head.x] = 1.0;
        for point in self.snake.iter().skip(1) {
            result[plane + point.y * self.width + point.x] = 1.0;
        }
        if let Some(food) = self.food {
            result[plane * 2 + food.y * self.width + food.x] = 1.0;
        }
        result
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn head(&self) -> Point {
        self.snake[0]
    }
    pub fn snake(&self) -> &VecDeque<Point> {
        &self.snake
    }
    pub fn direction(&self) -> Direction {
        self.direction
    }
    pub fn food(&self) -> Option<Point> {
        self.food
    }
    pub fn score(&self) -> u32 {
        self.score
    }
    pub fn is_done(&self) -> bool {
        self.done
    }

    fn place_food(&mut self) {
        let mut empty = Vec::with_capacity(self.width * self.height - self.snake.len());
        for y in 0..self.height {
            for x in 0..self.width {
                let point = Point::new(x, y);
                if !self.snake.contains(&point) {
                    empty.push(point);
                }
            }
        }
        self.food = empty.choose(&mut self.rng).copied();
    }

    fn result(&self, reward: f32) -> StepResult {
        StepResult {
            reward,
            done: self.done,
            score: self.score,
        }
    }
}

fn is_opposite(current: Direction, requested: Direction) -> bool {
    matches!(
        (current, requested),
        (Direction::Up, Direction::Down)
            | (Direction::Down, Direction::Up)
            | (Direction::Left, Direction::Right)
            | (Direction::Right, Direction::Left)
    )
}
