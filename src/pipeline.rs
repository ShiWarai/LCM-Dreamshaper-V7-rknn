use anyhow::{Context, Result};
use hf_hub::api::sync::Api;
use image::{ImageBuffer, Rgb};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use tokenizers::Tokenizer;

use crate::models::{RknnModel, UNetModel};
use crate::scheduler::LcmScheduler;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

pub const LATENT_H: usize = 64;
pub const LATENT_W: usize = 64;
pub const LATENT_C: usize = 4;
pub const LATENT_SIZE: usize = LATENT_C * LATENT_H * LATENT_W;

pub const IMG_H: usize = 512;
pub const IMG_W: usize = 512;

const MAX_SEQ_LEN: usize = 77;
const TEXT_EMB_DIM: usize = 768;
const GUIDANCE_EMB_DIM: usize = 256;

/// VAE scaling factor from vae_decoder/config.json
const VAE_SCALE: f32 = 0.18215;

// ─────────────────────────────────────────────────────────────────────────────
// Generate request / result
// ─────────────────────────────────────────────────────────────────────────────

pub struct GenerateRequest {
    pub prompt: String,
    pub steps: usize,
    pub guidance_scale: f32,
    /// None → random seed chosen at runtime
    pub seed: Option<u64>,
}

pub struct GenerateResult {
    /// Raw PNG bytes
    pub png_bytes: Vec<u8>,
    /// Actual seed used (useful when caller requested random seed)
    pub seed: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Local models directory (env LCM_MODELS_DIR or ./models when cwd is project root)
// ─────────────────────────────────────────────────────────────────────────────

fn local_models_dir() -> Option<std::path::PathBuf> {
    if let Ok(dir) = std::env::var("LCM_MODELS_DIR") {
        let p = std::path::PathBuf::from(dir);
        if p.is_dir() {
            return Some(p);
        }
    }
    std::env::current_dir().ok().map(|cwd| cwd.join("models")).filter(|p| p.is_dir())
}

// ─────────────────────────────────────────────────────────────────────────────
// Model download helpers
// ─────────────────────────────────────────────────────────────────────────────

pub fn download_models() -> Result<(
    std::path::PathBuf, // text_encoder
    std::path::PathBuf, // unet
    std::path::PathBuf, // vae_decoder
    std::path::PathBuf, // tokenizer
)> {
    let local_dir = local_models_dir();
    let api = Api::new().context("Failed to create HF Hub API")?;
    let lcm_repo = api.model("whaoyang/LCM-Dreamshaper-V7-ONNX-rk3588-512x512-2.3.0".to_string());

    let (text_encoder, vae_decoder) = if let Some(base) = local_dir.as_ref() {
        let whaoyang = base.join("LCM-Dreamshaper-V7-ONNX-rk3588-512x512-2.3.0");
        let te = whaoyang.join("text_encoder").join("model.rknn");
        let vae = whaoyang.join("vae_decoder").join("model.rknn");
        if te.is_file() && vae.is_file() {
            eprintln!("📂 Using local models from {}", base.display());
            eprintln!("  text_encoder: {}", te.display());
            eprintln!("  vae_decoder: {}", vae.display());
            (te, vae)
        } else {
            eprintln!("📥 Downloading text_encoder & vae_decoder from HuggingFace...");
            let te = lcm_repo.get("text_encoder/model.rknn").context("Failed to download text_encoder/model.rknn")?;
            let vae = lcm_repo.get("vae_decoder/model.rknn").context("Failed to download vae_decoder/model.rknn")?;
            eprintln!("  text_encoder: {}", te.display());
            eprintln!("  vae_decoder: {}", vae.display());
            (te, vae)
        }
    } else {
        eprintln!("📥 Downloading models from HuggingFace (cached after first run)...");
        let te = lcm_repo.get("text_encoder/model.rknn").context("Failed to download text_encoder/model.rknn")?;
        let vae = lcm_repo.get("vae_decoder/model.rknn").context("Failed to download vae_decoder/model.rknn")?;
        eprintln!("  text_encoder: {}", te.display());
        eprintln!("  vae_decoder: {}", vae.display());
        (te, vae)
    };

    let unet = unet_model_path(&local_dir)?;
    eprintln!("  unet: {}", unet.display());

    let tokenizer = tokenizer_path(&local_dir)
        .unwrap_or_else(|| download_clip_tokenizer().expect("Failed to download tokenizer"));
    eprintln!("  tokenizer: {}", tokenizer.display());

    Ok((text_encoder, unet, vae_decoder, tokenizer))
}

/// Return path to the UNet RKNN model compatible with librknnrt.so 2.3.2.
/// Prefers: local models dir (kautism 2.3.2) → ~/.cache/lcm-rs → HF kautism → HF whaoyang fallback.
fn unet_model_path(local_dir: &Option<std::path::PathBuf>) -> Result<std::path::PathBuf> {
    // 1. Local project models: LCM_Dreamshaper_v7-RKNN-2.3.2 (UNet for 2.3.2)
    if let Some(base) = local_dir {
        let kautism = base.join("LCM_Dreamshaper_v7-RKNN-2.3.2").join("unet_v232.rknn");
        if kautism.is_file() {
            eprintln!("  (using local UNet 2.3.2: {})", kautism.display());
            return Ok(kautism);
        }
    }

    // 2. Cache: locally recompiled UNet (README § "Building the UNet").
    let cache_path = dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("lcm-rs")
        .join("unet_v232.rknn");
    if cache_path.exists() {
        eprintln!("  (using cached UNet 2.3.2: {})", cache_path.display());
        return Ok(cache_path);
    }

    let api = Api::new().context("Failed to create HF Hub API")?;

    // 3. HF: kautism/LCM_Dreamshaper_v7-RKNN-2.3.2
    let user_repo = api.model("kautism/LCM_Dreamshaper_v7-RKNN-2.3.2".to_string());
    if let Ok(path) = user_repo.get("unet_v232.rknn") {
        eprintln!("  (using UNet from HF: kautism/LCM_Dreamshaper_v7-RKNN-2.3.2/unet_v232.rknn)");
        return Ok(path);
    }

    // 4. Fallback: whaoyang 2.3.0 model (may SIGSEGV on librknnrt 2.3.2).
    eprintln!("  ⚠️  No 2.3.2 UNet found; falling back to 2.3.0 model (may crash on librknnrt 2.3.2).");
    let lcm_repo = api.model("whaoyang/LCM-Dreamshaper-V7-ONNX-rk3588-512x512-2.3.0".to_string());
    lcm_repo.get("unet/model.rknn").context("Failed to download unet/model.rknn from fallback repo")
}

/// Tokenizer: from local whaoyang repo if present, else download CLIP.
fn tokenizer_path(local_dir: &Option<std::path::PathBuf>) -> Option<std::path::PathBuf> {
    let base = local_dir.as_ref()?;
    let whaoyang = base.join("LCM-Dreamshaper-V7-ONNX-rk3588-512x512-2.3.0");
    let tok = whaoyang.join("tokenizer").join("tokenizer.json");
    if tok.is_file() {
        Some(tok)
    } else {
        None
    }
}

/// Download CLIP tokenizer.json and cache it under ~/.cache/lcm-rs/.
fn download_clip_tokenizer() -> Result<std::path::PathBuf> {
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("lcm-rs");
    std::fs::create_dir_all(&cache_dir).context("Failed to create cache dir")?;

    let tokenizer_path = cache_dir.join("clip_tokenizer.json");
    if tokenizer_path.exists() {
        return Ok(tokenizer_path);
    }

    eprintln!("  Downloading CLIP tokenizer.json...");
    let url = "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/tokenizer.json";
    let bytes = reqwest::blocking::get(url)
        .context("Failed to fetch tokenizer.json")?
        .error_for_status()
        .context("Bad HTTP status for tokenizer.json")?
        .bytes()
        .context("Failed to read tokenizer.json bytes")?;

    std::fs::write(&tokenizer_path, &bytes).context("Failed to write tokenizer.json")?;
    Ok(tokenizer_path)
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline — model paths + tokenizer (RKNN contexts loaded one at a time)
// ─────────────────────────────────────────────────────────────────────────────

pub struct Pipeline {
    text_encoder_path: std::path::PathBuf,
    unet_path: std::path::PathBuf,
    vae_decoder_path: std::path::PathBuf,
    tokenizer: Tokenizer,
}

// RKNN contexts live in process memory without cross-thread sharing.
// We guard Pipeline with a Mutex so only one thread calls into RKNN at a time.
unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}

impl Pipeline {
    pub fn load() -> Result<Self> {
        let (te_path, unet_path, vae_path, tok_path) = download_models()?;

        eprintln!("🔧 Models ready (RK3588 NPU: loading one RKNN context at a time)");

        let tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            text_encoder_path: te_path,
            unet_path,
            vae_decoder_path: vae_path,
            tokenizer,
        })
    }

    pub fn print_info(&self) -> Result<()> {
        eprintln!("\n── text_encoder ──");
        let te = RknnModel::load(&self.text_encoder_path).context("Load text_encoder")?;
        te.print_info()?;
        eprintln!("\n── unet ──");
        let unet = UNetModel::load(&self.unet_path).context("Load unet")?;
        unet.print_info()?;
        eprintln!("\n── vae_decoder ──");
        let vae = RknnModel::load(&self.vae_decoder_path).context("Load vae_decoder")?;
        vae.print_info()?;
        Ok(())
    }

    pub fn generate(&self, req: GenerateRequest) -> Result<GenerateResult> {
        // Tokenize
        let input_ids = tokenize(&self.tokenizer, &req.prompt)?;
        let n_tokens = input_ids.iter().position(|&x| x == 49407).unwrap_or(MAX_SEQ_LEN);
        eprintln!("🔤 Prompt: \"{}\" ({} tokens)", req.prompt, n_tokens.saturating_sub(1));

        // Text encoding (unload before UNet — RK3588 NPU SRAM fits one large model)
        eprintln!("📝 Loading text encoder...");
        let text_emb = {
            let text_encoder =
                RknnModel::load(&self.text_encoder_path).context("Load text_encoder")?;
            text_encoder
                .run_with_int32_inputs(&[(0, &input_ids)], &[])
                .context("Text encoder failed")?
        };
        let expected_emb = MAX_SEQ_LEN * TEXT_EMB_DIM;
        eprintln!("  text encoder output: {} f32 elements (expected {})", text_emb.len(), expected_emb);
        if text_emb.len() < expected_emb {
            anyhow::bail!(
                "Text encoder output too small: got {} elements, need {} ({}×{})",
                text_emb.len(), expected_emb, MAX_SEQ_LEN, TEXT_EMB_DIM
            );
        }
        let text_emb_flat: Vec<f32> = text_emb[..expected_emb].to_vec();

        // Scheduler
        let mut scheduler = LcmScheduler::new();
        scheduler.set_timesteps(req.steps);
        eprintln!("📅 Timesteps ({} steps): {:?}", req.steps, scheduler.timesteps);

        // Random latent
        let seed = req.seed.unwrap_or_else(rand::random::<u64>);
        eprintln!("🎲 Seed: {}", seed);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0f32, 1.0f32).unwrap();
        let mut latent_nchw: Vec<f32> = (0..LATENT_SIZE).map(|_| normal.sample(&mut rng)).collect();

        // Guidance embedding
        let ts_cond = guidance_scale_embedding(req.guidance_scale, GUIDANCE_EMB_DIM);

        // Denoise loop
        eprintln!("🎨 Loading UNet, denoising ({} steps)...", req.steps);
        let timesteps = scheduler.timesteps.clone();
        {
            let unet = UNetModel::load(&self.unet_path).context("Load unet")?;
            for (step_idx, &timestep) in timesteps.iter().enumerate() {
                let latent_nhwc = nchw_to_nhwc(&latent_nchw, LATENT_C, LATENT_H, LATENT_W);
                eprint!("  step {}/{} (t={}) running UNet ...", step_idx + 1, req.steps, timestep);
                let noise_pred = unet
                    .run(&latent_nhwc, timestep as i64, &text_emb_flat, &ts_cond)
                    .with_context(|| format!("UNet step {step_idx} failed"))?;
                let (prev_latent, _denoised) =
                    scheduler.step(&noise_pred, timestep, &latent_nchw, step_idx, &mut rng);
                latent_nchw = prev_latent;
                eprintln!(" done");
            }
        }

        // VAE decode
        eprintln!("🖼️  Loading VAE decoder...");
        let scaled_nchw: Vec<f32> = latent_nchw.iter().map(|&v| v / VAE_SCALE).collect();
        let scaled_nhwc = nchw_to_nhwc(&scaled_nchw, LATENT_C, LATENT_H, LATENT_W);
        let pixels = {
            let vae_decoder =
                RknnModel::load(&self.vae_decoder_path).context("Load vae_decoder")?;
            vae_decoder
                .run_f32(&[(0, &scaled_nhwc)])
                .context("VAE decoder failed")?
        };

        // Encode to PNG
        let img = vae_output_to_image(&pixels, IMG_H, IMG_W);
        let mut png_bytes: Vec<u8> = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut png_bytes), image::ImageFormat::Png)
            .context("Failed to encode PNG")?;

        eprintln!("✅ Generation complete ({} bytes PNG)", png_bytes.len());
        Ok(GenerateResult { png_bytes, seed })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn tokenize(tokenizer: &Tokenizer, prompt: &str) -> Result<Vec<i32>> {
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let ids = encoding.get_ids();
    let mut input_ids = vec![0i32; MAX_SEQ_LEN];
    input_ids[0] = 49406; // BOS
    let content_len = ids.len().min(MAX_SEQ_LEN - 2);
    for i in 0..content_len {
        input_ids[1 + i] = ids[i] as i32;
    }
    let eos_pos = 1 + content_len;
    if eos_pos < MAX_SEQ_LEN {
        input_ids[eos_pos] = 49407; // EOS
    }
    Ok(input_ids)
}

fn guidance_scale_embedding(guidance_scale: f32, embedding_dim: usize) -> Vec<f32> {
    let w = guidance_scale - 1.0;
    let half_dim = embedding_dim / 2;
    let log_10000 = (10000.0f32).ln();
    let mut emb = vec![0.0f32; embedding_dim];
    for i in 0..half_dim {
        let freq = (-(log_10000 * i as f32 / ((half_dim as f32) - 1.0))).exp() * w;
        emb[i] = freq.sin();
        emb[half_dim + i] = freq.cos();
    }
    emb
}

/// Convert NCHW [1, C, H, W] → NHWC [1, H, W, C].
pub fn nchw_to_nhwc(data: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for ci in 0..c {
        for hi in 0..h {
            for wi in 0..w {
                let nchw = ci * h * w + hi * w + wi;
                let nhwc = hi * w * c + wi * c + ci;
                out[nhwc] = data[nchw];
            }
        }
    }
    out
}

/// Convert VAE output (NCHW [1, 3, H, W], range [-1, 1]) to RGB image.
pub fn vae_output_to_image(data: &[f32], h: usize, w: usize) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let mut img = ImageBuffer::new(w as u32, h as u32);
    let channel_size = h * w;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let r = ((data[i].clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8;
            let g = ((data[channel_size + i].clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8;
            let b = ((data[2 * channel_size + i].clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8;
            img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nchw_to_nhwc_layout() {
        // C=2, H=2, W=2 — NCHW [c,h,w] → NHWC [h,w,c]
        let nchw = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let nhwc = nchw_to_nhwc(&nchw, 2, 2, 2);
        assert_eq!(nhwc, vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]);
    }

    #[test]
    fn guidance_embedding_dim() {
        let emb = guidance_scale_embedding(7.5, 256);
        assert_eq!(emb.len(), 256);
    }

    #[test]
    fn vae_output_to_image_dimensions() {
        let h = 4;
        let w = 4;
        let data = vec![0.0f32; 3 * h * w];
        let img = vae_output_to_image(&data, h, w);
        assert_eq!(img.width(), w as u32);
        assert_eq!(img.height(), h as u32);
    }
}
