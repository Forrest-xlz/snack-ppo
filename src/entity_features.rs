//! Heterogeneous observations: channel-major cell planes, then three context
//! vectors. Topology describes a static tail-open graph, never a safety proof.
use crate::env::{Direction, Point, SnakeEnv};
use std::{cell::RefCell, collections::HashMap, rc::Rc};

pub(crate) const CELL_BASE: usize = 60;
pub(crate) const CELL_REGIONS: usize = 66;
pub(crate) const SNAKE: usize = 21;
pub(crate) const FOOD: usize = 24;
pub(crate) const GLOBAL: usize = 8;
pub(crate) const CONTEXT: usize = SNAKE + FOOD + GLOBAL;

thread_local! {
    static TABLES: RefCell<HashMap<(usize,usize),Rc<FeatureTables>>>=RefCell::new(HashMap::new());
    static SCRATCH: RefCell<FeatureScratch>=RefCell::new(FeatureScratch::default());
}

/// Deterministic expansions keyed by both board dimensions. Tables are local
/// to each encoder thread, so parallel environments do not contend on locks.
struct FeatureTables {
    extent: usize,
    absolute: Vec<[f32; 2]>,
    trig: Vec<[f32; 8]>,
    release: Vec<[f32; 27]>,
    length: Vec<[f32; 13]>,
    region_size: Vec<[f32; 2]>,
}

impl FeatureTables {
    fn new(w: usize, h: usize) -> Self {
        let area = w * h;
        let sx = w.saturating_sub(1).max(1) as f32;
        let sy = h.saturating_sub(1).max(1) as f32;
        let extent = w.max(h);
        let absolute = (0..area)
            .map(|c| [(c % w) as f32 / sx, (c / w) as f32 / sy])
            .collect();
        let trig = (-(extent as isize - 1)..extent as isize)
            .map(|offset| {
                let mut values = [0.0; 8];
                for (j, freq) in [1.0, 2.0, 4.0, 8.0].into_iter().enumerate() {
                    let angle = std::f32::consts::PI * freq * offset as f32 / extent as f32;
                    values[2 * j] = angle.sin();
                    values[2 * j + 1] = angle.cos();
                }
                values
            })
            .collect();
        let mut release_nodes = (0..8)
            .map(|p| (1usize << p) as f32)
            .filter(|&v| v < area as f32)
            .collect::<Vec<_>>();
        release_nodes.push(area as f32);
        let mut length_nodes = vec![3., 5., 10., 20., 40., 60., 90., 120.];
        length_nodes.retain(|&v| v < area as f32);
        length_nodes.push(area as f32);
        let mut release = vec![[0.0; 27]; area + 1];
        let mut length = vec![[0.0; 13]; area + 1];
        let mut region_size = vec![[0.0; 2]; area + 1];
        for n in 1..=area {
            release[n][0] = n as f32 / area as f32;
            release[n][1 + (n - 1).min(16)] = 1.0;
            two_hot(n as f32, &release_nodes, true, &mut release[n][18..]);
            length[n][0] = n as f32 / area as f32;
            length[n][1] = (n as f32).ln() / (area as f32).ln();
            two_hot(n as f32, &length_nodes, false, &mut length[n][2..11]);
            length[n][11] = (area - n) as f32 / area as f32;
            length[n][12] = ((area - n) as f32).ln_1p() / (area as f32).ln_1p();
            region_size[n] = [
                n as f32 / area as f32,
                (n as f32).ln_1p() / (area as f32).ln_1p(),
            ];
        }
        Self {
            extent,
            absolute,
            trig,
            release,
            length,
            region_size,
        }
    }

    fn ego_features(&self, p: Point, head: Point, direction: Direction) -> [f32; 18] {
        let (dx, dy) = ego(
            p.x as f32 - head.x as f32,
            p.y as f32 - head.y as f32,
            direction,
        );
        let mut values = [0.0; 18];
        let scale = self.extent.saturating_sub(1).max(1) as f32;
        values[0] = dx / scale;
        values[1] = dy / scale;
        let x = &self.trig[(dx as isize + self.extent as isize - 1) as usize];
        let y = &self.trig[(dy as isize + self.extent as isize - 1) as usize];
        for j in 0..4 {
            let xs = if dx == 0.0 && dx.is_sign_negative() {
                -x[2 * j]
            } else {
                x[2 * j]
            };
            let ys = if dy == 0.0 && dy.is_sign_negative() {
                -y[2 * j]
            } else {
                y[2 * j]
            };
            values[2 + j * 4..6 + j * 4].copy_from_slice(&[xs, x[2 * j + 1], ys, y[2 * j + 1]]);
        }
        values
    }
}

fn tables(w: usize, h: usize) -> Rc<FeatureTables> {
    TABLES.with(|cache| {
        Rc::clone(
            cache
                .borrow_mut()
                .entry((w, h))
                .or_insert_with(|| Rc::new(FeatureTables::new(w, h))),
        )
    })
}

pub(crate) fn encode(env: &SnakeEnv, with_cell_regions: bool) -> Vec<f32> {
    let channels = if with_cell_regions {
        CELL_REGIONS
    } else {
        CELL_BASE
    };
    let mut f = vec![0.0; channels * env.width() * env.height() + CONTEXT];
    encode_into(env, with_cell_regions, &mut f);
    f
}

pub(crate) fn encode_into(env: &SnakeEnv, with_cell_regions: bool, f: &mut [f32]) {
    let w = env.width();
    let h = env.height();
    let area = w * h;
    let channels = if with_cell_regions {
        CELL_REGIONS
    } else {
        CELL_BASE
    };
    assert_eq!(f.len(), channels * area + CONTEXT);
    f.fill(0.0);
    let tables = tables(w, h);
    let snake = env.snake();
    let length = snake.len();
    let head = env.head();
    let tail = *snake.back().unwrap();
    SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        let FeatureScratch { order, regions } = &mut *scratch;
        order.resize(area, usize::MAX);
        order.fill(usize::MAX);
        for (i, p) in snake.iter().enumerate() {
            order[p.y * w + p.x] = i;
        }
        regions.update(w, h, order, length, head, tail, &tables);
        for cell in 0..area {
            let p = Point::new(cell % w, cell / w);
            let i = order[cell];
            let category = if p == head {
                0
            } else if p == tail {
                4
            } else if i != usize::MAX {
                1
            } else if env.food() == Some(p) {
                2
            } else {
                3
            };
            f[category * area + cell] = 1.0;
            if i != usize::MAX {
                for (j, &v) in tables.release[length - i].iter().enumerate() {
                    f[(5 + j) * area + cell] = v;
                }
                for (base, other) in [
                    (52, i.checked_sub(1)),
                    (56, (i + 1 < length).then_some(i + 1)),
                ] {
                    if let Some(j) = other {
                        let target = snake[j];
                        let (x, y) = ego(
                            target.x as f32 - p.x as f32,
                            target.y as f32 - p.y as f32,
                            env.direction(),
                        );
                        let dir = match (x as i32, y as i32) {
                            (0, -1) => Some(0),
                            (0, 1) => Some(1),
                            (-1, 0) => Some(2),
                            (1, 0) => Some(3),
                            _ => None,
                        };
                        if let Some(dir) = dir {
                            f[(base + dir) * area + cell] = 1.0;
                        }
                    }
                }
            }
            f[32 * area + cell] = tables.absolute[cell][0];
            f[33 * area + cell] = tables.absolute[cell][1];
            for (j, v) in tables
                .ego_features(p, head, env.direction())
                .into_iter()
                .enumerate()
            {
                f[(34 + j) * area + cell] = v;
            }
            if with_cell_regions {
                if let Some(region) = regions.at(cell) {
                    for (j, &v) in region.features.iter().enumerate() {
                        f[(60 + j) * area + cell] = v;
                    }
                }
            }
        }
        let s = channels * area;
        f[s..s + 13].copy_from_slice(&tables.length[length]);
        let direction = match env.direction() {
            Direction::Up => 0,
            Direction::Down => 1,
            Direction::Left => 2,
            Direction::Right => 3,
        };
        f[s + 13 + direction] = 1.0;
        f[s + 17..s + 19].copy_from_slice(&tables.absolute[head.y * w + head.x]);
        let largest = regions
            .regions
            .iter()
            .filter(|r| r.head_adjacent)
            .map(|r| r.size)
            .max()
            .unwrap_or(0) as f32;
        f[s + 19] = largest / area as f32;
        f[s + 20] = (largest / length as f32).min(2.0);
        let food = s + SNAKE;
        if let Some(p) = env.food() {
            f[food] = 1.0;
            f[food + 1..food + 3].copy_from_slice(&tables.absolute[p.y * w + p.x]);
            f[food + 3..food + 21].copy_from_slice(&tables.ego_features(p, head, env.direction()));
            if let Some(region) = regions.at(p.y * w + p.x) {
                f[food + 21] = region.features[0];
                f[food + 22] = region.features[2];
                f[food + 23] = f32::from(region.head_adjacent);
            }
        }
        let g = food + FOOD;
        f[g] = tables.length[length][0];
        f[g + 1..g + 3].copy_from_slice(&tables.length[length][11..13]);
        if !regions.regions.is_empty() {
            f[g + 3 + (regions.regions.len() - 1).min(4)] = 1.0;
        }
    });
}

fn two_hot(value: f32, nodes: &[f32], logarithmic: bool, output: &mut [f32]) {
    if value <= nodes[0] {
        output[0] = 1.0;
        return;
    }
    for j in 1..nodes.len() {
        if value <= nodes[j] {
            let view = |v: f32| if logarithmic { v.ln() } else { v };
            let right = (view(value) - view(nodes[j - 1])) / (view(nodes[j]) - view(nodes[j - 1]));
            output[j - 1] = 1.0 - right;
            output[j] = right;
            return;
        }
    }
    output[nodes.len() - 1] = 1.0;
}

fn ego(dx: f32, dy: f32, direction: Direction) -> (f32, f32) {
    match direction {
        Direction::Up => (dx, dy),
        Direction::Down => (-dx, -dy),
        Direction::Left => (-dy, dx),
        Direction::Right => (dy, -dx),
    }
}

#[derive(Default)]
struct FeatureScratch {
    order: Vec<usize>,
    regions: Regions,
}
struct Region {
    size: usize,
    head_adjacent: bool,
    features: [f32; 6],
}
#[derive(Default)]
struct Regions {
    labels: Vec<usize>,
    regions: Vec<Region>,
    queue: Vec<usize>,
}

impl Regions {
    fn update(
        &mut self,
        w: usize,
        h: usize,
        order: &[usize],
        length: usize,
        head: Point,
        tail: Point,
        tables: &FeatureTables,
    ) {
        self.labels.resize(w * h, usize::MAX);
        self.labels.fill(usize::MAX);
        self.regions.clear();
        for cell in 0..w * h {
            let passable = order[cell] == usize::MAX || order[cell] + 1 == length;
            if !passable || self.labels[cell] != usize::MAX {
                continue;
            }
            let id = self.regions.len();
            self.labels[cell] = id;
            self.queue.clear();
            self.queue.push(cell);
            let mut cursor = 0;
            let mut size = 0;
            let mut head_adjacent = false;
            let mut tail_adjacent = false;
            while cursor < self.queue.len() {
                let c = self.queue[cursor];
                cursor += 1;
                size += 1;
                let p = Point::new(c % w, c / w);
                head_adjacent |= p.x.abs_diff(head.x) + p.y.abs_diff(head.y) == 1;
                tail_adjacent |= p.x.abs_diff(tail.x) + p.y.abs_diff(tail.y) == 1;
                for next in [
                    p.x.checked_sub(1).map(|x| p.y * w + x),
                    (p.x + 1 < w).then_some(c + 1),
                    p.y.checked_sub(1).map(|y| y * w + p.x),
                    (p.y + 1 < h).then_some(c + w),
                ]
                .into_iter()
                .flatten()
                {
                    if self.labels[next] == usize::MAX
                        && (order[next] == usize::MAX || order[next] + 1 == length)
                    {
                        self.labels[next] = id;
                        self.queue.push(next);
                    }
                }
            }
            self.regions.push(Region {
                size,
                head_adjacent,
                features: [
                    tables.region_size[size][0],
                    tables.region_size[size][1],
                    (size as f32 / length as f32).min(2.0),
                    f32::from(size >= length),
                    f32::from(tail_adjacent),
                    f32::from(head_adjacent),
                ],
            });
        }
    }

    fn at(&self, cell: usize) -> Option<&Region> {
        self.regions.get(self.labels[cell])
    }
}
