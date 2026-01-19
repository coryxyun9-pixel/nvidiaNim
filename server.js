const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();
const PORT = process.env.PORT || 3000;
const NIM_API_BASE = "https://integrate.api.nvidia.com/v1";
const NIM_API_KEY = process.env.NIM_API_KEY;

// === RESTORED CONFIGURATION ===
const SHOW_REASONING = true;
const ENABLE_THINKING_MODE = true;

const DEEPSEEK_CONFIG = {
    optimalMaxTokens: 4096,
    minMaxTokens: 2048,
    maxMaxTokens: 8192,
    optimalTemperature: 0.7,
    minTemperature: 0.3,
    requestTimeout: 180000, 
    retryAttempts: 0,
    retryDelay: 2000,
    // FIXED: DeepSeek specific stop sequences to stop the "looping/rambling"
    stop: ["<｜end of sentence｜>", "<|endoftext|>", "\nUser:", "\nAssistant:", "###"]
};

const MODEL_MAPPING = {
    "gpt-3": "meta/llama-3.3-70b-instruct",
    "gpt-4o": "deepseek-ai/deepseek-v3.2",
    "claude-3.5-sonnet": "deepseek-ai/deepseek-v3.2",
    "o1-preview": "deepseek-ai/deepseek-r1",
    "gpt-4": "meta/llama-3.1-405b-instruct",
};

// === RESTORED UTILITIES ===
const log = (level, message, data = {}) => {
    const timestamp = new Date().toISOString();
    console.log(JSON.stringify({ timestamp, level, message, ...data }));
};
const logInfo = (msg, data) => log('INFO', msg, data);
const logError = (msg, data) => log('ERROR', msg, data);
const logWarn = (msg, data) => log('WARN', msg, data);
const logDebug = (msg, data) => log('DEBUG', msg, data);

class Timer {
    constructor(name) {
        this.name = name;
        this.start = Date.now();
        this.checkpoints = [];
    }
    checkpoint(label) {
        const elapsed = Date.now() - this.start;
        this.checkpoints.push({ label, elapsed });
        return elapsed;
    }
    end() {
        return Date.now() - this.start;
    }
}

const estimateTokens = (text) => text ? Math.ceil(text.length / 4) : 0;

function calculateOptimalTokens(requestedTokens, totalInputTokens, isDeepSeek) {
    if (!isDeepSeek) return requestedTokens || 4096;
    let optimalTokens = requestedTokens || DEEPSEEK_CONFIG.optimalMaxTokens;
    if (totalInputTokens > 20000) optimalTokens = Math.min(optimalTokens, 4096);
    return Math.max(DEEPSEEK_CONFIG.minMaxTokens, Math.min(optimalTokens, DEEPSEEK_CONFIG.maxMaxTokens));
}

function formatResponseContent(message, showReasoning) {
    let fullContent = message.content || "";
    const reasoning = message.reasoning_content || message.reasoning;
    if (showReasoning && reasoning) {
        fullContent = `<think>\n${reasoning}\n</think>\n\n${fullContent}`;
    }
    return fullContent;
}

// === MAIN ROUTE ===
app.use(cors());
app.use(express.json({ limit: "50mb" }));

app.post("/v1/chat/completions", async (req, res) => {
    const timer = new Timer('chat_completion');
    const requestId = Math.random().toString(36).substring(7);
    
    const { model, messages, temperature, max_tokens, stream = false } = req.body;
    const nimModel = MODEL_MAPPING[model] || "deepseek-ai/deepseek-v3.2";
    const isDeepSeek = nimModel.includes("deepseek");

    // FIX 1: Send headers IMMEDIATELY if streaming. 
    // This stops Vercel from killing the process during the "Thinking" phase.
    if (stream) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");
    }

    try {
        const totalInputTokens = messages?.reduce((sum, msg) => sum + estimateTokens(msg.content), 0) || 0;
        const optimalMaxTokens = calculateOptimalTokens(max_tokens, totalInputTokens, isDeepSeek);
        
        const nimRequest = {
            model: nimModel,
            messages: messages,
            temperature: Math.max(temperature ?? 0.7, DEEPSEEK_CONFIG.minTemperature),
            max_tokens: optimalMaxTokens,
            stream: stream,
            stop: isDeepSeek ? DEEPSEEK_CONFIG.stop : undefined,
            // FIX 2: Added penalties to prevent the MoE "repetition loop"
            frequency_penalty: isDeepSeek ? 0.1 : 0,
            presence_penalty: isDeepSeek ? 0.1 : 0
        };

        if (ENABLE_THINKING_MODE && isDeepSeek) {
            nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };
        }

        logInfo('Calling NVIDIA NIM', { requestId, model: nimModel, stream });

        const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
            headers: { Authorization: `Bearer ${NIM_API_KEY}`, "Content-Type": "application/json" },
            responseType: stream ? "stream" : "json",
            timeout: DEEPSEEK_CONFIG.requestTimeout,
        });

        if (stream) {
            handleStreaming(response, res, model, requestId, timer);
        } else {
            const openaiResponse = {
                ...response.data,
                model: model,
                choices: response.data.choices.map(c => ({
                    ...c,
                    message: { ...c.message, content: formatResponseContent(c.message, SHOW_REASONING) }
                }))
            };
            logInfo('Response sent (Non-Stream)', { requestId, duration: `${timer.end()}ms` });
            res.json(openaiResponse);
        }
    } catch (error) {
        logError('Request failed', { requestId, error: error.message });
        if (!res.headersSent) res.status(500).json({ error: error.message });
        else res.end();
    }
});

function handleStreaming(response, res, originalModel, requestId, timer) {
    let buffer = ""; // FIX 3: Buffer to handle partial JSON chunks
    let reasoningStarted = false;
    let tokensGenerated = 0;

    response.data.on("data", (chunk) => {
        // We append to buffer because chunks often split in the middle of a JSON line
        buffer += chunk.toString();
        let lines = buffer.split("\n");
        buffer = lines.pop(); // Keep partial line for next chunk

        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || trimmed === "data: [DONE]") {
                if (trimmed === "data: [DONE]") res.write("data: [DONE]\n\n");
                continue;
            }

            try {
                const data = JSON.parse(trimmed.replace(/^data: /, ""));
                const delta = data.choices[0].delta;
                tokensGenerated++;

                let combinedContent = "";
                const reasoning = delta.reasoning_content || delta.reasoning;
                const content = delta.content;

                if (SHOW_REASONING) {
                    if (reasoning) {
                        if (!reasoningStarted) {
                            combinedContent = "<think>\n" + reasoning;
                            reasoningStarted = true;
                        } else combinedContent = reasoning;
                    } else if (content) {
                        if (reasoningStarted) {
                            combinedContent = "\n</think>\n\n" + content;
                            reasoningStarted = false;
                        } else combinedContent = content;
                    }
                } else {
                    combinedContent = content || "";
                }

                if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    data.model = originalModel;
                    res.write(`data: ${JSON.stringify(data)}\n\n`);
                }
            } catch (e) {
                // Parsing error usually means the line was cut off; ignore and wait for next chunk
            }
        }
    });

    response.data.on("end", () => {
        logInfo('Stream completed', { requestId, tokens: tokensGenerated, duration: `${timer.end()}ms` });
        res.end();
    });
}

module.exports = app;

