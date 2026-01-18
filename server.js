const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();
const PORT = process.env.PORT || 3000;

const NIM_API_BASE = "https://integrate.api.nvidia.com/v1";
const NIM_API_KEY = process.env.NIM_API_KEY;

// === DEEPSEEK V3.2 OPTIMIZED CONFIGURATION ===
const SHOW_REASONING = true;       
const ENABLE_THINKING_MODE = true;

// Optimized token limits for DeepSeek V3.2
const DEEPSEEK_CONFIG = {
  // Lower max_tokens for faster responses while maintaining quality
  optimalMaxTokens: 4096,      // Reduced for faster responses
  minMaxTokens: 2048,          // Minimum for coherent responses
  maxMaxTokens: 8192,          // Reduced maximum to prevent timeouts
  
  // Temperature settings
  optimalTemperature: 0.7,     // Best for creative RP
  minTemperature: 0.3,         // Never use 0 - causes DeepSeek to overthink
  
  // Timeout settings
  requestTimeout: 180000,      // 3 minutes (increased for safety)
  retryAttempts: 1,            // Retry once on timeout
  retryDelay: 2000,            // 2 second delay before retry
};

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

// === OPTIMIZED TOKEN CALCULATION FOR DEEPSEEK ===
function calculateOptimalTokens(requestedTokens, totalInputTokens, isDeepSeek) {
  if (!isDeepSeek) {
    return requestedTokens || 4096;
  }

  // DeepSeek V3.2 optimization logic
  let optimalTokens = requestedTokens || DEEPSEEK_CONFIG.optimalMaxTokens;

  // If input is very large, reduce output to prevent timeouts
  if (totalInputTokens > 20000) {
    optimalTokens = Math.min(optimalTokens, 4096);
    logWarn('Large input detected, reducing max_tokens', { 
      totalInputTokens, 
      adjustedMaxTokens: optimalTokens 
    });
  } else if (totalInputTokens > 10000) {
    optimalTokens = Math.min(optimalTokens, 6144);
  }

  // Enforce bounds
  optimalTokens = Math.max(
    DEEPSEEK_CONFIG.minMaxTokens,
    Math.min(optimalTokens, DEEPSEEK_CONFIG.maxMaxTokens)
  );

  return optimalTokens;
}

function formatResponseContent(message, showReasoning) {
  let fullContent = message.content || "";
  const reasoning = message.reasoning_content || message.reasoning;
  if (showReasoning && reasoning) {
    fullContent = `<think>\n${reasoning}\n</think>\n\n${fullContent}`;
  }
  return fullContent;
}

app.get("/", (req, res) => res.status(200).send("Proxy Active 🚀 | DeepSeek V3.2 Optimized"));

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

    const { model, messages, temperature, max_tokens, stream = false } = req.body;
    const nimModel = await selectModel(model);
    const isDeepSeek = nimModel.includes("deepseek");

    // Analyze request
    const systemMessage = messages?.find(m => m.role === 'system');
    const userMessages = messages?.filter(m => m.role === 'user') || [];
    const assistantMessages = messages?.filter(m => m.role === 'assistant') || [];
    
    const systemTokens = estimateTokens(systemMessage?.content);
    const totalInputTokens = messages?.reduce((sum, msg) => sum + estimateTokens(msg.content), 0) || 0;

    // Calculate optimal tokens for DeepSeek
    const optimalMaxTokens = calculateOptimalTokens(max_tokens, totalInputTokens, isDeepSeek);
    
    // Optimize temperature for DeepSeek
    let optimalTemperature = temperature ?? DEEPSEEK_CONFIG.optimalTemperature;
    
    // CRITICAL: DeepSeek gets stuck with temperature 0
    if (isDeepSeek && optimalTemperature < DEEPSEEK_CONFIG.minTemperature) {
      logWarn('Temperature too low for DeepSeek, adjusting', {
        requestId,
        requestedTemp: optimalTemperature,
        adjustedTemp: DEEPSEEK_CONFIG.minTemperature
      });
      optimalTemperature = DEEPSEEK_CONFIG.minTemperature;
    }

    logInfo('Request analyzed', {
      requestId,
      requestedModel: model,
      nimModel: nimModel,
      isDeepSeek,
      totalMessages: messages?.length || 0,
      userMessages: userMessages.length,
      assistantMessages: assistantMessages.length,
      systemPromptTokens: systemTokens,
      totalInputTokens,
      requestedMaxTokens: max_tokens || 'none',
      optimalMaxTokens,
      temperature: optimalTemperature,
      streaming: stream,
      thinkingMode: ENABLE_THINKING_MODE && isDeepSeek
    });

    // Performance warnings
    if (systemTokens > 8000) {
      logWarn('Very large system prompt', {
        requestId,
        systemTokens,
        recommendation: 'Consider reducing system prompt size for faster responses'
      });
    }

    if (totalInputTokens > 25000) {
      logWarn('Very large total context', {
        requestId,
        totalInputTokens,
        warning: 'Expect slower response times (30-60s)'
      });
    }

    // Estimate response time for DeepSeek
    if (isDeepSeek) {
      let estimatedTime = 10; // Base time
      estimatedTime += Math.floor(totalInputTokens / 1000) * 2; // +2s per 1k input tokens
      estimatedTime += Math.floor(optimalMaxTokens / 1000) * 3; // +3s per 1k output tokens
      
      if (ENABLE_THINKING_MODE) {
        estimatedTime += 10; // Thinking mode adds ~10s
      }

      logInfo('DeepSeek performance estimate', {
        requestId,
        estimatedResponseTime: `${estimatedTime}s`,
        thinkingModeEnabled: ENABLE_THINKING_MODE
      });
    }

    // Build request
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: optimalTemperature,
      max_tokens: optimalMaxTokens,
      stream: stream,
    };

    // Enable thinking mode for DeepSeek
    if (ENABLE_THINKING_MODE && isDeepSeek) {
      nimRequest.extra_body = {
        chat_template_kwargs: { thinking: true }
      };
      logDebug('DeepSeek thinking mode enabled', { requestId });
    }

    timer.checkpoint('before_nim_call');

    logInfo('Calling NVIDIA NIM API', {
      requestId,
      endpoint: NIM_API_BASE,
      model: nimModel,
      maxTokens: optimalMaxTokens,
      temperature: optimalTemperature,
      thinkingMode: ENABLE_THINKING_MODE && isDeepSeek
    });

    // Retry logic for DeepSeek timeouts
    let response = null;
    let lastError = null;
    
    for (let attempt = 0; attempt <= DEEPSEEK_CONFIG.retryAttempts; attempt++) {
      try {
        if (attempt > 0) {
          logWarn('Retrying request', { 
            requestId, 
            attempt, 
            maxAttempts: DEEPSEEK_CONFIG.retryAttempts + 1,
            delay: `${DEEPSEEK_CONFIG.retryDelay}ms`
          });
          await new Promise(resolve => setTimeout(resolve, DEEPSEEK_CONFIG.retryDelay));
        }

        response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
          headers: { 
            Authorization: `Bearer ${NIM_API_KEY}`, 
            "Content-Type": "application/json" 
          },
          responseType: stream ? "stream" : "json",
          timeout: DEEPSEEK_CONFIG.requestTimeout,
        });

        // Success - break out of retry loop
        break;
        
      } catch (error) {
        lastError = error;
        
        // Only retry on timeout errors
        if (error.code !== 'ECONNABORTED' || attempt === DEEPSEEK_CONFIG.retryAttempts) {
          throw error;
        }
        
        logWarn('Request timed out, will retry', {
          requestId,
          attempt: attempt + 1,
          timeout: `${DEEPSEEK_CONFIG.requestTimeout}ms`
        });
      }
    }

    const nimResponseTime = timer.checkpoint('nim_responded');

    logInfo('NIM API responded', {
      requestId,
      status: response.status,
      streaming: stream,
      responseTime: `${nimResponseTime}ms`,
      performanceRating: nimResponseTime < 10000 ? 'excellent' : 
                         nimResponseTime < 20000 ? 'good' : 
                         nimResponseTime < 40000 ? 'acceptable' : 'slow'
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

      const usage = response.data.usage;
      logInfo('Response sent', {
        requestId,
        inputTokens: usage?.prompt_tokens || 0,
        outputTokens: usage?.completion_tokens || 0,
        totalTokens: usage?.total_tokens || 0,
        finishReason: response.data.choices?.[0]?.finish_reason,
        tokensPerSecond: usage?.completion_tokens ? 
          Math.round((usage.completion_tokens / nimResponseTime) * 1000) : 0
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
      isTimeout: error.code === 'ECONNABORTED',
      nimError: error.response?.data,
      duration: `${errorDuration}ms`,
      stack: error.stack?.split('\n').slice(0, 3).join('\n')
    });

    // Special handling for timeouts
    if (error.code === 'ECONNABORTED') {
      logWarn('Request timeout - consider reducing context or max_tokens', {
        requestId,
        timeout: `${DEEPSEEK_CONFIG.requestTimeout}ms`
      });
    }

    res.status(500).json({ 
      error: error.message, 
      details: error.response?.data,
      requestId,
      suggestion: error.code === 'ECONNABORTED' ? 
        'Request timed out. Try reducing system prompt size or max_tokens.' : null
    });
  }
});

function handleStreaming(response, res, originalModelName, requestId, timer) {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  
  let reasoningStarted = false;
  let chunksReceived = 0;
  let bytesReceived = 0;
  let firstChunkTime = null;
  let tokensGenerated = 0;

  response.data.on("data", (chunk) => {
    if (!firstChunkTime) {
      firstChunkTime = Date.now();
      const timeToFirstToken = firstChunkTime - timer.start;
      logInfo('First token received', {
        requestId,
        timeToFirstToken: `${timeToFirstToken}ms`
      });
    }

    chunksReceived++;
    bytesReceived += chunk.length;

    // Log every 30 chunks
    if (chunksReceived % 30 === 0) {
      const elapsed = Date.now() - timer.start;
      const tokensPerSecond = tokensGenerated > 0 ? 
        Math.round((tokensGenerated / elapsed) * 1000) : 0;
      
      logDebug('Streaming progress', {
        requestId,
        chunksReceived,
        bytesReceived,
        tokensGenerated,
        tokensPerSecond,
        elapsed: `${elapsed}ms`
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

        // Count tokens (approximate)
        if (delta.content || delta.reasoning_content || delta.reasoning) {
          tokensGenerated++;
        }

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
          delete data.choices[0].delta.reasoning;
          data.model = originalModelName;
          res.write(`data: ${JSON.stringify(data)}\n\n`);
        }
      } catch (e) {
        logError('Stream parsing error', { 
          requestId, 
          error: e.message,
          line: line.substring(0, 100)
        });
      }
    });
  });

  response.data.on("end", () => {
    const streamDuration = timer.end();
    const avgTokensPerSecond = tokensGenerated > 0 ? 
      Math.round((tokensGenerated / streamDuration) * 1000) : 0;
    
    logInfo('Stream completed successfully', {
      requestId,
      totalChunks: chunksReceived,
      totalBytes: bytesReceived,
      estimatedTokens: tokensGenerated,
      duration: `${streamDuration}ms`,
      avgTokensPerSecond,
      performanceRating: avgTokensPerSecond > 30 ? 'excellent' :
                         avgTokensPerSecond > 15 ? 'good' :
                         avgTokensPerSecond > 5 ? 'acceptable' : 'slow'
    });
    res.end();
  });

  response.data.on("error", (err) => {
    logError('Stream error', {
      requestId,
      error: err.message,
      chunksBeforeError: chunksReceived,
      bytesBeforeError: bytesReceived
    });
    if (!res.headersSent) {
      res.status(500).end();
    } else {
      res.end();
    }
  });
}

module.exports = app;

if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`🚀 Server on port ${PORT}`);
    console.log(`📊 DeepSeek V3.2 Optimizations:`);
    console.log(`   - Optimal max_tokens: ${DEEPSEEK_CONFIG.optimalMaxTokens}`);
    console.log(`   - Request timeout: ${DEEPSEEK_CONFIG.requestTimeout}ms`);
    console.log(`   - Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
  });
}
