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


}

// Fetch with timeout wrapper
const fetchWithTimeout = async (url, options, timeoutMs = 9000) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    logWarn('Fetch timeout triggered', {
      url,
      timeout: `${timeoutMs}ms`
    });
    controller.abort();
  }, timeoutMs);

  try {
    const fetchStart = Date.now();
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    const fetchDuration = Date.now() - fetchStart;
    
    clearTimeout(timeoutId);
    
    logInfo('Fetch completed', {
      url,
      status: response.status,
      statusText: response.statusText,
      duration: `${fetchDuration}ms`
    });
    
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    
    if (error.name === 'AbortError') {
      logError('Fetch aborted - timeout exceeded', {
        url,
        timeout: `${timeoutMs}ms`
      });
      throw new Error(`Request timeout after ${timeoutMs}ms`);
    }
    
    logError('Fetch failed', {
      url,
      errorName: error.name,
      errorMessage: error.message
    });
    throw error;
  }
};

// Token counter (approximate)
const estimateTokens = (text) => {
  if (!text) return 0;
  // Rough estimate: ~4 characters per token
  return Math.ceil(text.length / 4);
};

// Main handler
export default async function handler(req, res) {
  const timer = new Timer('chat_completion_request');
  const requestId = Math.random().toString(36).substring(7);

  logInfo('=== NEW REQUEST RECEIVED ===', {
    requestId,
    method: req.method,
    url: req.url,
    userAgent: req.headers['user-agent'],
    contentType: req.headers['content-type']
  });

  // Only accept POST requests
  if (req.method !== 'POST') {
    logWarn('Invalid method', { requestId, method: req.method });
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Parse request body
    timer.checkpoint('start_body_parse');
    const body = await req.json();
    timer.checkpoint('body_parsed');

    // Analyze request
    const systemMessage = body.messages?.find(m => m.role === 'system');
    const userMessages = body.messages?.filter(m => m.role === 'user') || [];
    const assistantMessages = body.messages?.filter(m => m.role === 'assistant') || [];
    
    const systemTokens = estimateTokens(systemMessage?.content);
    const totalMessageTokens = body.messages?.reduce((sum, msg) => 
      sum + estimateTokens(msg.content), 0
    ) || 0;

    logInfo('Request analyzed', {
      requestId,
      model: body.model,
      totalMessages: body.messages?.length || 0,
      userMessages: userMessages.length,
      assistantMessages: assistantMessages.length,
      systemPromptTokens: systemTokens,
      totalInputTokens: totalMessageTokens,
      maxTokens: body.max_tokens || 'default',
      temperature: body.temperature || 'default',
      streaming: body.stream || false,
      topP: body.top_p || 'default'
    });

    // Detailed message breakdown
    if (body.messages && body.messages.length > 0) {
      logDebug('Message details', {
        requestId,
        messages: body.messages.map((msg, idx) => ({
          index: idx,
          role: msg.role,
          contentLength: msg.content?.length || 0,
          estimatedTokens: estimateTokens(msg.content),
          contentPreview: msg.content?.substring(0, 100) + '...'
        }))
      });
    }

    // Check for potential issues
    if (systemTokens > 10000) {
      logWarn('Very large system prompt detected', {
        requestId,
        systemTokens,
        warning: 'This may cause slow responses or timeouts'
      });
    }

    if (totalMessageTokens > 30000) {
      logWarn('Very large total context detected', {
        requestId,
        totalMessageTokens,
        warning: 'This may cause slow responses or timeouts'
      });
    }

    // Prepare NIM API request
    const nimEndpoint = process.env.NIM_ENDPOINT || process.env.NVIDIA_NIM_ENDPOINT;
    const nimApiKey = process.env.NIM_API_KEY || process.env.NVIDIA_API_KEY;

    if (!nimEndpoint) {
      logError('NIM endpoint not configured', { requestId });
      return res.status(500).json({ 
        error: 'Server misconfiguration - NIM endpoint not set' 
      });
    }

    logInfo('Calling NVIDIA NIM API', {
      requestId,
      endpoint: nimEndpoint,
      model: body.model,
      streaming: body.stream || false
    });

    timer.checkpoint('before_nim_call');

    // Call NIM API with timeout
    const nimResponse = await fetchWithTimeout(
      `${nimEndpoint}/v1/chat/completions`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(nimApiKey ? { 'Authorization': `Bearer ${nimApiKey}` } : {})
        },
        body: JSON.stringify(body)
      },
      9000 // 9 second timeout (before Vercel's 10s limit)
    );

    timer.checkpoint('nim_responded');

    // Check response status
    if (!nimResponse.ok) {
      const errorText = await nimResponse.text();
      
      logError('NIM API error', {
        requestId,
        status: nimResponse.status,
        statusText: nimResponse.statusText,
        errorBody: errorText.substring(0, 500)
      });

      return res.status(nimResponse.status).json({
        error: `NIM API error: ${nimResponse.statusText}`,
        details: errorText,
        requestId
      });
    }

    // Handle streaming response
    if (body.stream) {
      logInfo('Starting streaming response', { requestId });
      
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const reader = nimResponse.body.getReader();
      const decoder = new TextDecoder();
      let chunksReceived = 0;
      let bytesReceived = 0;

      try {
        while (true) {
          const { done, value } = await reader.read();
          
          if (done) {
            const streamDuration = timer.checkpoint('stream_complete');
            logInfo('Stream completed successfully', {
              requestId,
              chunksReceived,
              bytesReceived,
              streamDuration: `${streamDuration}ms`
            });
            break;
          }

          chunksReceived++;
          bytesReceived += value.length;
          
          const chunk = decoder.decode(value, { stream: true });
          res.write(chunk);

          // Log every 10 chunks
          if (chunksReceived % 10 === 0) {
            logDebug('Streaming progress', {
              requestId,
              chunksReceived,
              bytesReceived,
              elapsed: `${Date.now() - timer.start}ms`
            });
          }
        }
      } catch (streamError) {
        logError('Stream error', {
          requestId,
          errorMessage: streamError.message,
          chunksBeforeError: chunksReceived
        });
        throw streamError;
      } finally {
        res.end();
        timer.end();
      }

    } else {
      // Handle non-streaming response
      logInfo('Parsing non-streaming response', { requestId });
      timer.checkpoint('before_json_parse');
      
      const data = await nimResponse.json();
      timer.checkpoint('json_parsed');

      const outputTokens = data.usage?.completion_tokens || 0;
      const totalTokens = data.usage?.total_tokens || 0;

      logInfo('Response received', {
        requestId,
        model: data.model,
        finishReason: data.choices?.[0]?.finish_reason,
        inputTokens: data.usage?.prompt_tokens || 0,
        outputTokens,
        totalTokens,
        responseLength: data.choices?.[0]?.message?.content?.length || 0
      });

      timer.end();

      return res.status(200).json(data);
    }

  } catch (error) {
    const errorDuration = timer.checkpoint('error_occurred');
    
    logError('Request failed', {
      requestId,
      errorName: error.name,
      errorMessage: error.message,
      errorStack: error.stack,
      duration: `${errorDuration}ms`
    });

    // Check if headers already sent (streaming error)
    if (res.headersSent) {
      logWarn('Headers already sent, cannot send error response', { requestId });
      res.end();
      return;
    }

    return res.status(500).json({
      error: 'Internal server error',
      message: error.message,
      requestId,
      duration: `${errorDuration}ms`
    });
  }
}

// Export config for Vercel
export const config = {
  api: {
    bodyParser: true,
    responseLimit: false,
  },
};

module.exports = app;

if (require.main === module) {
  app.listen(PORT, () => console.log(`🚀 Server on port ${PORT}`));
}

