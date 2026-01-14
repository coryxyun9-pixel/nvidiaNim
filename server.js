const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();
const PORT = process.env.PORT || 3000;

const NIM_API_BASE = "https://integrate.api.nvidia.com/v1";
const NIM_API_KEY = process.env.NIM_API_KEY;

// --- CONFIGURATION ---
const SHOW_REASONING = true;       
const ENABLE_THINKING_MODE = true;  

const MODEL_MAPPING = {
  "gpt-4": "meta/llama-3.3-70b-instruct",
  "gpt-4-turbo": "meta/llama-3.3-70b-instruct",
  "gpt-4o": "deepseek-ai/deepseek-v3.2", // RESTORED
  "claude-3.5-sonnet": "deepseek-ai/deepseek-v3.2", // RESTORED
  "gpt-3.5-turbo": "meta/llama-3.3-70b-instruct",
  "claude-3-sonnet": "meta/llama-3.3-70b-instruct",
  "o1-preview": "deepseek-ai/deepseek-r1",
  "o1-mini": "meta/llama-3.3-70b-instruct",
  "gemini-pro": "meta/llama-3.3-70b-instruct",
  "gpt-4o-mini": "meta/llama-3.3-70b-instruct",
};

const FALLBACK_MODELS = {
  large: "deepseek-ai/deepseek-v3.2",
  medium: "meta/llama-3.3-70b-instruct",
  small: "meta/llama-3.1-8b-instruct",
};

app.use(cors({ origin: "*", methods: ["GET", "POST", "OPTIONS"], allowedHeaders: ["Content-Type", "Authorization"] }));
app.use(express.json({ limit: "50mb" }));

async function selectModel(requestedModel) {
  return MODEL_MAPPING[requestedModel] || FALLBACK_MODELS.large;
}

function formatResponseContent(message, showReasoning) {
  let fullContent = message.content || "";
  // Check both DeepSeek specific and generic reasoning fields
  const reasoning = message.reasoning_content || message.reasoning;
  if (showReasoning && reasoning) {
    fullContent = `<think>\n${reasoning}\n</think>\n\n${fullContent}`;
  }
  return fullContent;
}

app.get("/", (req, res) => res.status(200).send("Proxy Active 🚀"));

app.post("/v1/chat/completions", async (req, res) => {
  try {
    if (!NIM_API_KEY) return res.status(500).json({ error: "NIM_API_KEY missing" });

    const { model, messages, temperature = 0.6, max_tokens, stream = false } = req.body;
    const nimModel = await selectModel(model);

    // LOGGING: Check your Vercel logs to see what model is actually used
    console.log(`[REQUEST] Using NIM Model: ${nimModel}`);

    const isDeepSeek = nimModel.includes("deepseek");
    
    // FORCED TOKEN WINDOW: V3.2 reasoning will FAIL if this is low.
    // If you send 4k, the model "gives up" on complex thought.
    const safeMaxTokens = isDeepSeek ? Math.max(max_tokens || 0, 16384) : (max_tokens || 4096);

    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature,
      max_tokens: safeMaxTokens,
      stream: stream,
    };

    // THE INTEL TOGGLE: This is what stops it from being "stupid"
    if (ENABLE_THINKING_MODE && isDeepSeek) {
      nimRequest.extra_body = {
        // vLLM / NIM standard for 2026
        chat_template_kwargs: { thinking: true }
      };
    }

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
    console.error("NIM Error Details:", error.response?.data || error.message);
    res.status(500).json({ error: error.message, details: error.response?.data });
  }
});

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
        // NVIDIA NIM sometimes swaps between 'reasoning_content' and 'reasoning'
        const reasoning = delta.reasoning_content || delta.reasoning;
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
      } catch (e) { }
    });
  });

  response.data.on("end", () => res.end());
}

module.exports = app;

if (require.main === module) {
  app.listen(PORT, () => console.log(`🚀 Server on port ${PORT}`));
}
