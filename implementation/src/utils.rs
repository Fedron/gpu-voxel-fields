pub fn position_to_index(position: glam::UVec3, size: glam::UVec3) -> usize {
    (position.x + position.y * size.x + position.z * size.x * size.y) as usize
}

pub fn get_sphere_positions(center: glam::IVec3, size: u32) -> Vec<glam::IVec3> {
    let radius = size as i32 / 2;
    let mut positions = Vec::new();
    let radius_squared = (radius * radius) as f32;
    for x in -radius..=radius {
        for y in -radius..=radius {
            for z in -radius..=radius {
                let offset = glam::IVec3::new(x, y, z);
                if offset.as_vec3().length_squared() <= radius_squared {
                    positions.push(center + offset);
                }
            }
        }
    }

    positions
}
/// Contains various statistics about the averages for a list it was created for.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct Statistics {
    pub mean: f32,
    pub median: f32,
    pub std_dev: f32,
    pub min_max: (f32, f32),
    pub confidence_interval: (f32, f32),
}

impl Statistics {
    pub fn calculate(data: &[f32]) -> Self {
        let mean = Self::calculate_mean(data);
        let std_dev = Self::calculate_std_dev(data, mean);
        let confidence_interval = Self::calculate_confidence_interval(data, mean, std_dev);

        Self {
            mean,
            median: Self::calculate_median(data),
            std_dev,
            min_max: Self::calculate_min_max(data),
            confidence_interval,
        }
    }

    fn calculate_mean(data: &[f32]) -> f32 {
        let sum: f32 = data.iter().sum();
        sum / data.len() as f32
    }

    fn calculate_median(data: &[f32]) -> f32 {
        if data.len() == 0 {
            return 0.0;
        }

        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let len = sorted.len();
        if len % 2 == 0 {
            (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
        } else {
            sorted[len / 2]
        }
    }

    fn calculate_std_dev(data: &[f32], mean: f32) -> f32 {
        let variance: f32 =
            data.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / data.len() as f32;
        variance.sqrt()
    }

    fn calculate_min_max(data: &[f32]) -> (f32, f32) {
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        (min, max)
    }

    fn calculate_confidence_interval(data: &[f32], mean: f32, std_dev: f32) -> (f32, f32) {
        let n = data.len() as f32;
        let margin_of_error = 1.96 * (std_dev / n.sqrt());
        (mean - margin_of_error, mean + margin_of_error)
    }
}
