// hello-claude.js
import Anthropic from "@anthropic-ai/sdk";
import dotenv from "dotenv";
dotenv.config();

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

const response = await client.messages.create({
  model: "claude-3-opus-20240229",
  max_tokens: 64,
  messages: [{ role: "user", content: "こんにちは、Claude!" }],
});

console.log(response.content[0].text);