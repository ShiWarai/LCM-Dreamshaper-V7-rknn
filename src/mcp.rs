use std::sync::{Arc, Mutex};

use rmcp::{
    ServerHandler,
    ErrorData as McpError,
    handler::server::{
        router::tool::ToolRouter,
        tool::ToolCallContext,
        wrapper::Parameters,
    },
    model::{
        CallToolRequestParam, CallToolResult, Content, ListToolsResult,
        PaginatedRequestParam, ServerCapabilities, ServerInfo,
    },
    service::{RequestContext, RoleServer},
    schemars, tool, tool_router,
};

use crate::pipeline::{GenerateRequest, Pipeline};

// ─────────────────────────────────────────────────────────────────────────────
// Tool input schema
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct GenerateImageParams {
    /// Text description of the image to generate
    pub prompt: String,
    /// Number of LCM denoising steps (1–50, default: 15)
    #[serde(default = "default_steps")]
    pub steps: usize,
    /// Classifier-free guidance scale (default: 8.5)
    #[serde(default = "default_guidance_scale")]
    pub guidance_scale: f32,
    /// Optional fixed seed for reproducibility; omit for random
    pub seed: Option<u64>,
}

fn default_steps() -> usize { 15 }
fn default_guidance_scale() -> f32 { 8.5 }

// ─────────────────────────────────────────────────────────────────────────────
// MCP handler
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct DreamshaperMcp {
    pipeline: Arc<Mutex<Pipeline>>,
    /// Dynamically updated from the HTTP `Host` header so the returned image
    /// URL always reflects the interface the client actually connected through.
    pub base_url: Arc<Mutex<String>>,
    #[allow(dead_code)]
    tool_router: ToolRouter<Self>,
}

impl DreamshaperMcp {
    pub fn new(pipeline: Arc<Mutex<Pipeline>>, base_url: Arc<Mutex<String>>) -> Self {
        Self {
            pipeline,
            base_url,
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl DreamshaperMcp {
    /// Generate an image from a text prompt using LCM Dreamshaper V7 on the RK3588 NPU.
    #[tool(description = "Generate an image from a text prompt using LCM Dreamshaper V7 on Rockchip RK3588 NPU. Returns a URL to the generated 512×512 PNG image and the seed used.")]
    async fn generate_image(
        &self,
        Parameters(params): Parameters<GenerateImageParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let pipeline = self.pipeline.clone();
        let base_url = self.base_url.lock()
            .map_err(|e| rmcp::ErrorData::internal_error(format!("base_url lock: {e}"), None))?
            .clone();
        let req = GenerateRequest {
            prompt: params.prompt,
            steps: params.steps,
            guidance_scale: params.guidance_scale,
            seed: params.seed,
        };

        // Run blocking RKNN inference in a dedicated thread so the async executor
        // is not starved. The Mutex serialises concurrent requests — RKNN only
        // handles one inference at a time.
        let result = tokio::task::spawn_blocking(move || {
            pipeline
                .lock()
                .map_err(|e| rmcp::ErrorData::internal_error(format!("Lock poisoned: {e}"), None))?
                .generate(req)
                .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))
        })
        .await
        .map_err(|e| rmcp::ErrorData::internal_error(format!("Task join: {e}"), None))??;

        // Save to /tmp/dreamshaper/ and return a URL — avoids sending 500KB
        // of base64 through the MCP response which overflows AI context windows.
        let dir = std::path::Path::new("/tmp/dreamshaper");
        std::fs::create_dir_all(dir)
            .map_err(|e| rmcp::ErrorData::internal_error(format!("mkdir: {e}"), None))?;

        // Evict files older than 1 hour (best-effort, ignore errors)
        if let Ok(entries) = std::fs::read_dir(dir) {
            let cutoff = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata()
                    && meta.modified().map(|t| t < cutoff).unwrap_or(false)
                {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }

        let filename = format!("{}.png", result.seed);
        let filepath = dir.join(&filename);
        std::fs::write(&filepath, &result.png_bytes)
            .map_err(|e| rmcp::ErrorData::internal_error(format!("write: {e}"), None))?;

        let url = format!("{}/images/{}", base_url, filename);
        Ok(CallToolResult::success(vec![
            Content::text(format!("image_url={url}\nseed={}", result.seed)),
        ]))
    }
}

impl ServerHandler for DreamshaperMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "LCM Dreamshaper V7 image generation server running on Rockchip RK3588 NPU. \
                 Call generate_image with a text prompt to produce a 512×512 PNG image."
                    .into(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParam>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult {
            tools: self.tool_router.list_all(),
            next_cursor: None,
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParam,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        self.tool_router.call(ToolCallContext::new(self, request, context)).await
    }
}
