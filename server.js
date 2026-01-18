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
  "gpt-4o": "deepseek-ai/deepseek-v3.2",
  "claude-3.5-sonnet": "deepseek-ai/deepseek-v3.2",
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

// === LOGGING UTILITIES ===
const log = (level, message, data = {}) => {
  const timestamp = new Date().toISOString();
  console.log(JSON.stringify({ timestamp, level, message, ...data }));
};

const logInfo = (msg, data) => log('INFO', msg, data);
const logError = (msg, data) => log('ERROR', msg, data);
const logWarn = (msg, data) => log('WARN', msg, data);
const logDebug = (msg, data) => log('DEBUG', msg, data);

// === TIMER CLASS ===
class Timer {
  constructor(name) {
    this.name = name;
    this.start = Date.now();
    this.checkpoints = [];
  }

  checkpoint(label) {
    const elapsed = Date.now() - this.start;
    this.checkpoints.push({ label, elapsed });
    logDebug(`Checkpoint: ${label}`, { operation: this.name, elapsed: `${elapsed}ms` });
    return elapsed;
  }

  end() {
    const total = Date.now() - this.start;
    logInfo(`Operation completed: ${this.name}`, { totalDuration: `${total}ms`, checkpoints: this.checkpoints });
    return total;
  }
}

// === TOKEN ESTIMATOR ===
const estimateTokens = (text) => {
  if (!text) return 0;
  return Math.ceil(text.length / 4);
};

async function selectModel(requestedModel) {
  return MODEL_MAPPING[requestedModel] || FALLBACK_MODELS.large;
}

function formatResponseContent(message, showReasoning) {
  let fullContent = message.content || "";
  const reasoning = message.reasoning_content || message.reasoning;
  if (showReasoning && reasoning) {
    fullContent = `<think>\n${reasoning}\n</think>\n\n${fullContent}`;
  }
  return fullContent;
}

app.get("/", (req, res) => res.status(200).send("Proxy Active 🚀"));

app.post("/v1/chat/completions", async (req, res) => {
  const timer = new Timer('chat_completion');
  const requestId = Math.random().toString(36).substring(7);

  logInfo('=== NEW REQUEST RECEIVED ===', {
    requestId,
    method: req.method,
    userAgent: req.headers['user-agent']
  });

  try {
    if (!NIM_API_KEY) {
      logError('NIM_API_KEY missing', { requestId });
      return res.status(500).json({ error: "NIM_API_KEY missing" });
    }

    timer.checkpoint('body_received');

    const { model, messages, temperature = 0.6, max_tokens, stream = false } = req.body;
    const nimModel = await selectModel(model);

    // Analyze request
    const systemMessage = messages?.find(m => m.role === 'system');
    const userMessages = messages?.filter(m => m.role === 'user') || [];
    const assistantMessages = messages?.filter(m => m.role === 'assistant') || [];
    
    const systemTokens = estimateTokens(systemMessage?.content);
    const totalMessageTokens = messages?.reduce((sum, msg) => sum + estimateTokens(msg.content), 0) || 0;

    logInfo('Request analyzed', {
      requestId,
      requestedModel: model,
      nimModel: nimModel,
      totalMessages: messages?.length || 0,
      userMessages: userMessages.length,
      assistantMessages: assistantMessages.length,
      systemPromptTokens: systemTokens,
      totalInputTokens: totalMessageTokens,
      maxTokens: max_tokens || 'default',
      temperature,
      streaming: stream
    });

    // Warnings
    if (systemTokens > 10000) {
      logWarn('Very large system prompt detected', {
        requestId,
        systemTokens,
        warning: 'This may cause slow responses'
      });
    }

    if (totalMessageTokens > 30000) {
      logWarn('Very large total context', {
        requestId,
        totalMessageTokens
      });
    }

    const isDeepSeek = nimModel.includes("deepseek");
    const safeMaxTokens = isDeepSeek ? Math.max(max_tokens || 0, 16384) : (max_tokens || 4096);

    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature,
      max_tokens: safeMaxTokens,
      stream: stream,
    };

    if (ENABLE_THINKING_MODE && isDeepSeek) {
      nimRequest.extra_body = {
        chat_template_kwargs: { thinking: true }
      };
      logDebug('Thinking mode enabled', { requestId });
    }

    timer.checkpoint('before_nim_call');

    logInfo('Calling NVIDIA NIM API', {
      requestId,
      endpoint: NIM_API_BASE,
      model: nimModel,
      maxTokens: safeMaxTokens
    });

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: { Authorization: `Bearer ${NIM_API_KEY}`, "Content-Type": "application/json" },
      responseType: stream ? "stream" : "json",
      timeout: 300000,
    });

    timer.checkpoint('nim_responded');

    logInfo('NIM API responded', {
      requestId,
      status: response.status,
      streaming: stream,
      elapsed: `${Date.now() - timer.start}ms`
    });

    if (stream) {
      logInfo('Starting streaming response', { requestId });
      handleStreaming(response, res, model, requestId, timer);
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

      logInfo('Response sent', {
        requestId,
        inputTokens: response.data.usage?.prompt_tokens || 0,
        outputTokens: response.data.usage?.completion_tokens || 0,
        totalTokens: response.data.usage?.total_tokens || 0,
        finishReason: response.data.choices?.[0]?.finish_reason
      });

      timer.end();
      res.json(openaiResponse);
    }
  } catch (error) {
    const errorDuration = timer.checkpoint('error_occurred');
    
    logError('Request failed', {
      requestId,
      errorName: error.name,
      errorMessage: error.message,
      errorCode: error.code,
      nimError: error.response?.data,
      duration: `${errorDuration}ms`,
      stack: error.stack
    });

    res.status(500).json({ 
      error: error.message, 
      details: error.response?.data,
      requestId 
    });
  }
});

function handleStreaming(response, res, originalModelName, requestId, timer) {
  res.setHeader("Content-Type", "text/event-stream");
  let reasoningStarted = false;
  let chunksReceived = 0;
  let bytesReceived = 0;

  response.data.on("data", (chunk) => {
    chunksReceived++;
    bytesReceived += chunk.length;

    // Log every 20 chunks
    if (chunksReceived % 20 === 0) {
      logDebug('Streaming progress', {
        requestId,
        chunksReceived,
        bytesReceived,
        elapsed: `${Date.now() - timer.start}ms`
      });
    }

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
      } catch (e) {
        logError('Stream parsing error', { requestId, error: e.message });
      }
    });
  });

  response.data.on("end", () => {
    const streamDuration = timer.end();
    logInfo('Stream completed', {
      requestId,
      totalChunks: chunksReceived,
      totalBytes: bytesReceived,
      duration: `${streamDuration}ms`
    });
    res.end();
  });

  response.data.on("error", (err) => {
    logError('Stream error', {
      requestId,
      error: err.message,
      chunksBeforeError: chunksReceived
    });
    res.end();
  });
}

module.exports = app;

if (require.main === module) {
  app.listen(PORT, () => console.log(`🚀 Server on port ${PORT}`));
}
