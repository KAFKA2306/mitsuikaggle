import Anthropic from "@anthropic-ai/sdk";
import 'dotenv/config';

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const run = async () => {
  const response = await client.messages.create({
    model: "claude-3-sonnet-20240229",
    max_tokens: 100,
    temperature: 0.7,
    messages: [
      { role: "user", content: "こんにちは。俳句をひとつください。" },
    ],
  });

  console.log(response.content[0].text.trim());
};

run().catch(console.error);