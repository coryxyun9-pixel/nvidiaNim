// ============================================
// FILE: server.js
// UPDATED: Jan 2026 for DeepSeek-V3.2 Reasoning
// ============================================

const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();
const PORT = process.env.PORT || 3000;

const NIM_API_BASE = process.env.NIM_API_BASE || "https://integrate.api.nvidia.com/v1";
const NIM_API_KEY = process.env.NIM_API_KEY;

// --- CRITICAL FIXES START HERE ---
const SHOW_REASONING = true;       // Set to true to see the <think> blocks
const ENABLE_THINKING_MODE = true;  // MUST be true for V3.2 to follow complex prompts
// ---------------------------------

const MODEL_MAPPING = {
  "gpt-4": "meta/llama-3.3-70b-instruct",
  "gpt-4o": "deepseek-ai/deepseek-v3.2", // Replaced R1 with V3.2 for general speed+smarts
  "claude-3.5-sonnet": "deepseek-ai/deepseek-v3.2",
  "o1-preview": "deepseek-ai/deepseek-r1",
  "gpt-4o-mini": "meta/llama-4-scout-17b-16e-instruct",
  "gemini-pro": "meta/llama-4-scout-17b-16e-instruct",
};

const FALLBACK_MODELS = {
  large: "deepseek-ai/deepseek-v3.2",
  medium: "meta/llama-3.3-70b-instruct",
  small: "meta/llama-4-scout-17b-16e-instruct",
};

app.use(cors({ origin: "*", methods: ["GET", "POST", "OPTIONS"], allowedHeaders: ["Content-Type", "Authorization"] }));
app.use(express.json({ limit: "50mb" }));

// Helper: Smart model selection
async function selectModel(requestedModel) {
  return MODEL_MAPPING[requestedModel] || FALLBACK_MODELS.large;
}

// Helper: Format response content for non-streaming
function formatResponseContent(message, showReasoning) {
  let fullContent = message.content || "";
  // NIM returns thoughts in reasoning_content field
  if (showReasoning && message.reasoning_content) {
    fullContent = `<think>\n${message.reasoning_content}\n</think>\n\n${fullContent}`;
  }
  return fullContent;
}

app.post("/v1/chat/completions", async (req, res) => {
  try {
    if (!NIM_API_KEY) return res.status(500).json({ error: "NIM_API_KEY missing" });

    const { model, messages, temperature = 0.6, max_tokens = 16384, stream = false } = req.body;
    const nimModel = await selectModel(model);

    // Dynamic Token Management: Reasoning needs space!
    // We increase the cap to 16k because V3.2 "Thinking" uses tokens for the thought trace.
    const safeMaxTokens = nimModel.includes("deepseek") ? 16384 : Math.min(max_tokens, 4096);

    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature,
      max_tokens: safeMaxTokens,
      stream: stream,
    };

    // ENABLE THE BRAIN: This tells V3.2 to use its reasoning engine
    if (ENABLE_THINKING_MODE && nimModel.includes("deepseek")) {
      nimRequest.extra_body = {
        chat_template_kwargs: { thinking: true }
      };
    }

    console.log(`📝 Routing ${model} → ${nimModel} (Thinking: ${ENABLE_THINKING_MODE})`);

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: { Authorization: `Bearer ${NIM_API_KEY}`, "Content-Type": "application/json" },
      responseType: stream ? "stream" : "json",
      timeout: 300000,
    });

    if (stream) {
      handleStreaming(response, res, model);
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
        usage: response.data.usage,
      };
      res.json(openaiResponse);
    }
  } catch (error) {
    console.error("✗ Proxy error:", error.message);
    res.status(500).json({ error: error.message });
  }
});

// Refactored streaming to handle <think> blocks correctly
function handleStreaming(response, res, originalModelName) {
  res.setHeader("Content-Type", "text/event-stream");
  let reasoningStarted = false;

  response.data.on("data", (chunk) => {
    const lines = chunk.toString().split("\n");
    lines.forEach((line) => {
      if (!line.startsWith("data: ") || line.includes("[DONE]")) {
        if (line.includes("[DONE]")) res.write(line + "\n\n");
        return;
      }

      try {
        const data = JSON.parse(line.slice(6));
        const delta = data.choices?.[0]?.delta;
        if (!delta) return;

        let combinedContent = "";
        const reasoning = delta.reasoning_content;
        const content = delta.content;

        if (SHOW_REASONING) {
          if (reasoning) {
            if (!reasoningStarted) {
              combinedContent = "<think>\n" + reasoning;
              reasoningStarted = true;
            } else {
              combinedContent = reasoning;
            }
          } else if (content) {
            if (reasoningStarted) {
              combinedContent = "\n</think>\n\n" + content;
              reasoningStarted = false;
            } else {
              combinedContent = content;
            }
          }
        } else {
          combinedContent = content || "";
        }

        if (combinedContent) {
          data.choices[0].delta.content = combinedContent;
          delete data.choices[0].delta.reasoning_content;
          data.model = originalModelName;
          res.write(`data: ${JSON.stringify(data)}\n\n`);
        }
      } catch (e) { /* ignore parse errors */ }
    });
  });

  response.data.on("end", () => res.end());
}

app.listen(PORT, () => console.log(`🚀 Proxy running on http://localhost:${PORT}`));
