/// LCM (Latent Consistency Model) Scheduler
///
/// Implements the LCMScheduler compatible with diffusers LCMScheduler config:
///   beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
///   num_train_timesteps=1000, original_inference_steps=50
pub struct LcmScheduler {
    /// Precomputed cumulative product of (1 - beta_t)
    pub alphas_cumprod: Vec<f32>,
    /// Timesteps selected for inference (descending order)
    pub timesteps: Vec<usize>,
    /// Number of inference steps configured
    pub num_inference_steps: usize,
    /// Scaling factor for timestep conditioning (LCM uses 10.0)
    pub timestep_scaling: f64,
    /// sigma_data for c_skip / c_out computation
    pub sigma_data: f64,
}

impl LcmScheduler {
    /// Create scheduler and compute beta/alpha schedule.
    pub fn new() -> Self {
        let num_train = 1000usize;
        let beta_start = 0.00085f64;
        let beta_end = 0.012f64;

        // "scaled_linear" schedule: linspace(sqrt(beta_start), sqrt(beta_end), T) ** 2
        let betas: Vec<f64> = (0..num_train)
            .map(|i| {
                let t = i as f64 / (num_train - 1) as f64;
                let b = beta_start.sqrt() + t * (beta_end.sqrt() - beta_start.sqrt());
                b * b
            })
            .collect();

        // alphas_cumprod = cumprod(1 - beta_t)
        let mut alphas_cumprod = Vec::with_capacity(num_train);
        let mut cumprod = 1.0f64;
        for &beta in &betas {
            cumprod *= 1.0 - beta;
            alphas_cumprod.push(cumprod as f32);
        }

        LcmScheduler {
            alphas_cumprod,
            timesteps: vec![],
            num_inference_steps: 0,
            timestep_scaling: 10.0,
            sigma_data: 0.5,
        }
    }

    /// Set inference timesteps.
    ///
    /// Mirrors diffusers LCMScheduler.set_timesteps():
    ///   lcm_origin_steps=50, num_train_timesteps=1000
    pub fn set_timesteps(&mut self, num_inference_steps: usize) {
        self.num_inference_steps = num_inference_steps;
        let lcm_origin_steps = 50usize;
        let num_train = 1000usize;
        let c = num_train / lcm_origin_steps; // = 20

        // lcm_origin_timesteps = [c-1, 2c-1, ..., lcm_origin_steps*c - 1]
        //                      = [19, 39, ..., 999]
        let lcm_origin_timesteps: Vec<usize> =
            (1..=lcm_origin_steps).map(|i| i * c - 1).collect();

        // Reverse then pick every (lcm_origin_steps / num_inference_steps) step
        let skipping = (lcm_origin_steps as f64 / num_inference_steps as f64).ceil() as usize;
        let reversed: Vec<usize> = lcm_origin_timesteps.iter().rev().cloned().collect();
        self.timesteps = reversed
            .iter()
            .step_by(skipping)
            .take(num_inference_steps)
            .cloned()
            .collect();
    }

    /// Compute c_skip and c_out for a given timestep (LCM boundary condition).
    fn scalings(&self, timestep: usize) -> (f64, f64) {
        let scaled_t = timestep as f64 * self.timestep_scaling;
        let sigma_data2 = self.sigma_data * self.sigma_data;
        let c_skip = sigma_data2 / (scaled_t * scaled_t + sigma_data2);
        let c_out = scaled_t / (scaled_t * scaled_t + sigma_data2).sqrt();
        (c_skip, c_out)
    }

    /// Get alpha_cumprod at timestep t (0 ≤ t < 1000).
    #[allow(dead_code)]
    pub fn alpha_cumprod(&self, t: usize) -> f32 {
        self.alphas_cumprod[t]
    }

    /// Perform one LCM denoising step.
    ///
    /// Returns (prev_sample, denoised_sample).
    ///
    /// # Arguments
    /// * `noise_pred`  - UNet output, predicted noise (epsilon), shape [n]
    /// * `timestep`    - Current timestep t
    /// * `sample`      - Current noisy latent x_t, shape [n]
    /// * `step_idx`    - Index of current step in self.timesteps
    /// * `rng`         - Random number generator (used for adding prev-step noise)
    pub fn step(
        &self,
        noise_pred: &[f32],
        timestep: usize,
        sample: &[f32],
        step_idx: usize,
        rng: &mut impl rand::Rng,
    ) -> (Vec<f32>, Vec<f32>) {
        use rand_distr::{Distribution, Normal};

        let alpha_t = self.alphas_cumprod[timestep] as f64;
        let beta_t = 1.0 - alpha_t;

        // Determine previous timestep
        let prev_timestep = if step_idx + 1 < self.timesteps.len() {
            self.timesteps[step_idx + 1]
        } else {
            0
        };
        let alpha_prev = if prev_timestep > 0 {
            self.alphas_cumprod[prev_timestep] as f64
        } else {
            // final_alpha_cumprod: set_alpha_to_one=true → 1.0
            1.0f64
        };
        let beta_prev = 1.0 - alpha_prev;

        // Predicted original sample (x0) from epsilon prediction:
        //   x0_pred = (x_t - sqrt(1 - alpha_t) * noise) / sqrt(alpha_t)
        let (c_skip, c_out) = self.scalings(timestep);

        let n = sample.len();
        let mut x0_pred = vec![0.0f64; n];
        for i in 0..n {
            x0_pred[i] = (sample[i] as f64 - beta_t.sqrt() * noise_pred[i] as f64)
                / alpha_t.sqrt();
            // Clip to [-1, 1] (clip_sample = false in config, so skip)
        }

        // LCM denoised output: denoised = c_out * x0_pred + c_skip * x_t
        let mut denoised = vec![0.0f32; n];
        for i in 0..n {
            denoised[i] = (c_out * x0_pred[i] + c_skip * sample[i] as f64) as f32;
        }

        // DDPM reverse step to x_{t-1}:
        //   x_{t-1} = sqrt(alpha_{t-1}) * denoised + sqrt(1 - alpha_{t-1}) * noise
        let normal = Normal::new(0.0f64, 1.0).unwrap();
        let mut prev_sample = vec![0.0f32; n];
        for i in 0..n {
            let noise_sample = normal.sample(rng);
            prev_sample[i] =
                (alpha_prev.sqrt() * denoised[i] as f64 + beta_prev.sqrt() * noise_sample) as f32;
        }

        (prev_sample, denoised)
    }
}

impl Default for LcmScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn set_timesteps_count_matches_steps() {
        let mut scheduler = LcmScheduler::new();
        scheduler.set_timesteps(10);
        assert_eq!(scheduler.num_inference_steps, 10);
        assert_eq!(scheduler.timesteps.len(), 10);
    }

    #[test]
    fn timesteps_are_descending() {
        let mut scheduler = LcmScheduler::new();
        scheduler.set_timesteps(8);
        for window in scheduler.timesteps.windows(2) {
            assert!(window[0] > window[1]);
        }
    }

    #[test]
    fn step_preserves_latent_shape() {
        let mut scheduler = LcmScheduler::new();
        scheduler.set_timesteps(4);
        let latent = vec![0.5f32; 16];
        let noise_pred = vec![0.1f32; 16];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let (prev, _) = scheduler.step(&noise_pred, 999, &latent, 0, &mut rng);
        assert_eq!(prev.len(), latent.len());
    }
}
