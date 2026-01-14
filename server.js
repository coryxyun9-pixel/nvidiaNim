// ============================================
// FILE: server.js
// PURPOSE: Main proxy server that translates
//          OpenAI requests to NVIDIA format
// ============================================

const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();
const PORT = process.env.PORT || 3000;

// NVIDIA API settings
const NIM_API_BASE =
  process.env.NIM_API_BASE || "https://integrate.api.nvidia.com/v1";
const NIM_API_KEY = process.env.NIM_API_KEY;

// Feature toggles (YOU CAN CHANGE THESE)
const SHOW_REASONING = false; // Change to true to see AI's thinking
const ENABLE_THINKING_MODE = false; // Keep false for your models

// Model mapping (YOU CAN CUSTOMIZE THIS)
const MODEL_MAPPING = {
  "gpt-4": "meta/llama-3.3-70b-instruct",
  "gpt-4-turbo": "meta/llama-3.3-70b-instruct",
  "gpt-4o": "deepseek-ai/deepseek-r1",
  "claude-3.5-sonnet": "deepseek-ai/deepseek-v3.2",
  "gpt-3.5-turbo": "meta/llama-4-maverick-17b-128e-instruct",
  "claude-3-sonnet": "meta/llama-4-maverick-17b-128e-instruct",
  "o1-preview": "deepseek-ai/deepseek-r1",
  "o1-mini": "meta/llama-3.3-70b-instruct",
  "gemini-pro": "meta/llama-4-scout-17b-16e-instruct",
  "gpt-4o-mini": "meta/llama-4-scout-17b-16e-instruct",
};

const FALLBACK_MODELS = {
  large: "meta/llama-3.3-70b-instruct",
  medium: "meta/llama-4-maverick-17b-128e-instruct",
  small: "meta/llama-4-scout-17b-16e-instruct",
};

// Middleware setup
app.use(
  cors({
    origin: "*",
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);

app.use(express.json({ limit: "50mb" }));

app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
  next();
});

// Helper function: Smart model selection
async function selectModel(requestedModel) {
  let nimModel = MODEL_MAPPING[requestedModel];

  if (nimModel) {
    return nimModel;
  }

  try {
    const testResponse = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      {
        model: requestedModel,
        messages: [{ role: "user", content: "test" }],
        max_tokens: 1,
      },
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          "Content-Type": "application/json",
        },
        validateStatus: (status) => status < 500,
        timeout: 5000,
      }
    );

    if (testResponse.status >= 200 && testResponse.status < 300) {
      console.log(`✓ Model ${requestedModel} available directly`);
      return requestedModel;
    }
  } catch (error) {
    console.log(`✗ Model ${requestedModel} not available directly`);
  }

  const modelLower = requestedModel.toLowerCase();

  if (
    modelLower.includes("gpt-4") ||
    modelLower.includes("claude-opus") ||
    modelLower.includes("405b") ||
    modelLower.includes("ultra")
  ) {
    return FALLBACK_MODELS.large;
  } else if (
    modelLower.includes("claude") ||
    modelLower.includes("gemini") ||
    modelLower.includes("70b") ||
    modelLower.includes("sonnet")
  ) {
    return FALLBACK_MODELS.medium;
  } else {
    return FALLBACK_MODELS.small;
  }
}

// Helper function: Format response content
function formatResponseContent(message, showReasoning) {
  let fullContent = message.content || "";

  if (showReasoning && message.reasoning_content) {
    fullContent = `<think>\n${message.reasoning_content}\n</think>\n\n${fullContent}`;
  }

  return fullContent;
}

// ENDPOINT 1: Root (welcome page)
app.get("/", (req, res) => {
  res.json({
    service: "OpenAI to NVIDIA NIM Proxy",
    version: "2.0",
    status: "online",
    endpoints: {
      health: "/health",
      models: "/v1/models",
      chat: "/v1/chat/completions",
    },
    documentation: "See README.md for usage instructions",
  });
});

// ENDPOINT 2: Health check
app.get("/health", (req, res) => {
  const hasApiKey = !!NIM_API_KEY;

  res.json({
    status: "ok",
    timestamp: new Date().toISOString(),
    service: "OpenAI to NVIDIA NIM Proxy",
    version: "2.0",
    config: {
      reasoning_display: SHOW_REASONING,
      thinking_mode: ENABLE_THINKING_MODE,
      api_key_configured: hasApiKey,
      available_models: Object.keys(MODEL_MAPPING).length,
    },
  });
});

// ENDPOINT 3: List models
app.get("/v1/models", (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map((model) => ({
    id: model,
    object: "model",
    created: 1677649963,
    owned_by: "nvidia-nim-proxy",
    permission: [],
    root: model,
    parent: null,
  }));

  res.json({
    object: "list",
    data: models,
  });
});

// ENDPOINT 4: Chat completions (THE MAIN ENDPOINT)
app.post("/v1/chat/completions", async (req, res) => {
  try {
    if (!NIM_API_KEY) {
      return res.status(500).json({
        error: {
          message:
            "NVIDIA API key not configured. Please set NIM_API_KEY environment variable.",
          type: "configuration_error",
          code: 500,
        },
      });
    }

    const {
      model,
      messages,
      temperature = 0.7,
      max_tokens = 6000,
      top_p = 1,
      frequency_penalty = 0,
      presence_penalty = 0,
      stream = false,
    } = req.body;

    if (!model) {
      return res.status(400).json({
        error: {
          message: "Model parameter is required",
          type: "invalid_request_error",
          code: 400,
        },
      });
    }

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({
        error: {
          message: "Messages array is required and must not be empty",
          type: "invalid_request_error",
          code: 400,
        },
      });
    }

    const nimModel = await selectModel(model);
    console.log(`📝 Routing ${model} → ${nimModel}`);

    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: Math.max(0, Math.min(2, temperature)),
      max_tokens: Math.min(max_tokens, 4096),
      top_p: Math.max(0, Math.min(1, top_p)),
      stream: stream,
    };

    if (ENABLE_THINKING_MODE) {
      nimRequest.extra_body = {
        chat_template_kwargs: { thinking: true },
      };
    }

    console.log(`🚀 Sending request to NVIDIA NIM...`);

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          "Content-Type": "application/json",
          Accept: stream ? "text/event-stream" : "application/json",
        },
        responseType: stream ? "stream" : "json",
        timeout: 300000,
      }
    );

    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
      res.setHeader("Cache-Control", "no-cache");
      res.setHeader("Connection", "keep-alive");
      res.setHeader("X-Accel-Buffering", "no");

      let buffer = "";
      let reasoningStarted = false;

      response.data.on("data", (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        lines.forEach((line) => {
          if (line.startsWith("data: ")) {
            if (line.includes("[DONE]")) {
              res.write(line + "\n\n");
              return;
            }

            try {
              const data = JSON.parse(line.slice(6));

              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;

                if (SHOW_REASONING) {
                  let combinedContent = "";

                  if (reasoning && !reasoningStarted) {
                    combinedContent = "<think>\n" + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }

                  if (content && reasoningStarted) {
                    combinedContent += "\n</think>\n\n" + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }

                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  if (content) {
                    data.choices[0].delta.content = content;
                  } else {
                    data.choices[0].delta.content = "";
                  }
                  delete data.choices[0].delta.reasoning_content;
                }

                if (data.model) {
                  data.model = model;
                }
              }

              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + "\n");
            }
          }
        });
      });

      response.data.on("end", () => {
        console.log("✓ Stream completed");
        res.end();
      });

      response.data.on("error", (err) => {
        console.error("✗ Stream error:", err.message);
        res.end();
      });
    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map((choice) => ({
          index: choice.index,
          message: {
            role: choice.message.role,
            content: formatResponseContent(choice.message, SHOW_REASONING),
          },
          finish_reason: choice.finish_reason,
        })),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0,
        },
      };

      console.log("✓ Response completed");
      res.json(openaiResponse);
    }
  } catch (error) {
    console.error("✗ Proxy error:", error.message);

    const statusCode = error.response?.status || 500;
    const errorMessage =
      error.response?.data?.error?.message ||
      error.message ||
      "Internal server error";

    res.status(statusCode).json({
      error: {
        message: errorMessage,
        type: "proxy_error",
        code: statusCode,
        details:
          process.env.NODE_ENV === "development" ? error.stack : undefined,
      },
    });
  }
});

// Catch-all for unknown endpoints
app.all("*", (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.method} ${req.path} not found`,
      type: "invalid_request_error",
      code: 404,
      supported_endpoints: [
        "GET /",
        "GET /health",
        "GET /v1/models",
        "POST /v1/chat/completions",
      ],
    },
  });
});

// Start the server
app.listen(PORT, () => {
  console.log("========================================");
  console.log("🚀 OpenAI to NVIDIA NIM Proxy");
  console.log("========================================");
  console.log(`📡 Server running on port ${PORT}`);
  console.log(`🔧 API Base: ${NIM_API_BASE}`);
  console.log(`🔑 API Key: ${NIM_API_KEY ? "✓ Configured" : "✗ Missing"}`);
  console.log(
    `🧠 Reasoning Display: ${SHOW_REASONING ? "ENABLED" : "DISABLED"}`
  );
  console.log(
    `💭 Thinking Mode: ${ENABLE_THINKING_MODE ? "ENABLED" : "DISABLED"}`
  );
  console.log(`📋 Available Models: ${Object.keys(MODEL_MAPPING).length}`);
  console.log("========================================");
  console.log(`🌐 Health check: http://localhost:${PORT}/health`);
  console.log("========================================");
});




