use anyhow::Result;
use clap::{Parser, Subcommand};
use lcm_dreamshaper_v7_rknn::pipeline::{GenerateRequest, Pipeline};
use lcm_dreamshaper_v7_rknn::serve;

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "dreamshaper-cli",
    about = "LCM Dreamshaper V7 RKNN image generator — CLI and OpenAI-compatible server",
    version
)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,

    // ── generate defaults (used when no subcommand is given) ─────────────────

    /// Text prompt to generate
    #[arg(short, long, default_value = "a beautiful sunset over the ocean, photorealistic",
          global = true)]
    prompt: String,

    /// Number of LCM inference steps (higher = better quality, slower)
    #[arg(short = 's', long, default_value_t = 15, global = true)]
    steps: usize,

    /// Guidance scale
    #[arg(short = 'g', long, default_value_t = 8.5, global = true)]
    guidance_scale: f32,

    /// Random seed (omit or 0 for a random seed)
    #[arg(long, global = true)]
    seed: Option<u64>,

    /// Output image path
    #[arg(short, long, default_value = "output.png", global = true)]
    output: String,

    /// Print model tensor info and exit
    #[arg(long, global = true)]
    info_only: bool,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Generate an image from a text prompt (default when no subcommand given)
    Generate {
        #[arg(short, long)]
        prompt: Option<String>,
        #[arg(short = 's', long)]
        steps: Option<usize>,
        #[arg(short = 'g', long)]
        guidance_scale: Option<f32>,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(short, long)]
        output: Option<String>,
        #[arg(long)]
        info_only: bool,
    },
    /// Start an OpenAI-compatible HTTP server for image generation
    Serve {
        /// Host to bind
        #[arg(long, default_value = "0.0.0.0")]
        host: String,
        /// Port to bind
        #[arg(long, default_value_t = 8765)]
        port: u16,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let result = match &args.command {
        Some(Command::Serve { host, port }) => serve::serve(host, *port).await,
        Some(Command::Generate {
            prompt,
            steps,
            guidance_scale,
            seed,
            output,
            info_only,
        }) => {
            let prompt = prompt.clone().unwrap_or_else(|| args.prompt.clone());
            let steps = steps.unwrap_or(args.steps);
            let guidance_scale = guidance_scale.unwrap_or(args.guidance_scale);
            let seed = seed.or(args.seed);
            let output = output.clone().unwrap_or_else(|| args.output.clone());
            let info_only = *info_only || args.info_only;
            tokio::task::block_in_place(|| {
                run_generate(prompt, steps, guidance_scale, seed, output, info_only)
            })
        }
        None => {
            let seed = args.seed.filter(|&s| s != 0);
            tokio::task::block_in_place(|| {
                run_generate(
                    args.prompt.clone(),
                    args.steps,
                    args.guidance_scale,
                    seed,
                    args.output.clone(),
                    args.info_only,
                )
            })
        }
    };

    if let Err(e) = result {
        eprintln!("❌ Error: {:#}", e);
        std::process::exit(1);
    }
}

fn run_generate(
    prompt: String,
    steps: usize,
    guidance_scale: f32,
    seed: Option<u64>,
    output: String,
    info_only: bool,
) -> Result<()> {
    let pipeline = Pipeline::load()?;

    if info_only {
        pipeline.print_info()?;
        return Ok(());
    }

    let result = pipeline.generate(GenerateRequest {
        prompt,
        steps,
        guidance_scale,
        seed,
    })?;

    std::fs::write(&output, &result.png_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to write {}: {}", output, e))?;
    eprintln!("✅ Saved to: {} (seed={})", output, result.seed);
    Ok(())
}
