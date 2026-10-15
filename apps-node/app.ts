
import "dotenv/config";

import express, {
  type Request,
  type Response,
} from "express";

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { StringOutputParser } from "@langchain/core/output_parsers";

const app = express();
const PORT = 3000;

app.use(express.json());
app.use(express.static("public"));

// Initialize Gemini
const llm = new ChatGoogleGenerativeAI({
  model: "gemini-3.5-flash-lite",
  temperature: 0.7,
});

// LangChain output parser
const outputParser = new StringOutputParser();

// Create a chain: LLM -> output parser
const chain = llm.pipe(outputParser);

// Define the request body type
interface ChatRequest {
  query: string;
}

// Chat API
app.post(
  "/api/chat",
  async (
    req: Request<object, object, ChatRequest>,
    res: Response,
  ): Promise<void> => {
    try {
      const { query } = req.body;

      if (typeof query !== "string" || !query.trim()) {
        res.status(400).json({
          error: "Please enter a valid message.",
        });
        return;
      }

      // Equivalent to llm.invoke(query) + StrOutputParser
      const result = await chain.invoke(query);

      res.json({
        answer: result,
      });
    } catch (error: unknown) {
      console.error("Gemini error:", error);

      const message =
        error instanceof Error
          ? error.message
          : "An unexpected error occurred.";

      res.status(500).json({
        error: message,
      });
    }
  },
);

app.listen(PORT, () => {
  console.log(`Chatbot running at http://localhost:${PORT}`);
});
