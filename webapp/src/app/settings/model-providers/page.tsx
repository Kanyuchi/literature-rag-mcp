"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const defaultModels = [
  { id: "llm", label: "LLM", required: true },
  { id: "embedding", label: "Embedding", required: true },
  { id: "vlm", label: "VLM", required: false },
  { id: "asr", label: "ASR", required: false },
  { id: "rerank", label: "Rerank", required: false },
  { id: "tts", label: "TTS", required: false },
];

const availableProviders = [
  {
    id: "openai",
    name: "OpenAI",
    capabilities: ["LLM", "TEXT EMBEDDING", "TEXT RE-RANK", "TTS", "SPEECHTOTEXT", "MODERATION"],
  },
  {
    id: "anthropic",
    name: "Anthropic",
    capabilities: ["LLM"],
  },
  {
    id: "gemini",
    name: "Gemini",
    capabilities: ["LLM", "TEXT EMBEDDING", "IMAGETOTEXT"],
  },
  {
    id: "moonshot",
    name: "Moonshot",
    capabilities: ["LLM", "TEXT EMBEDDING", "IMAGETOTEXT"],
  },
];

const addedModels = [
  { id: "baai", name: "BAAI", hasApiKey: true },
  { id: "deepseek", name: "DeepSeek", hasApiKey: true },
];

export default function ModelProvidersPage() {
  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-semibold">Set default models</h1>
        <p className="text-muted-foreground">
          Please complete these settings before beginning
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Default Models */}
        <div className="space-y-4">
          {defaultModels.map((model) => (
            <div key={model.id} className="flex items-center gap-4">
              <label className="text-sm w-24">
                {model.required && <span className="text-destructive">*</span>}
                {model.label}
              </label>
              <Select>
                <SelectTrigger className="flex-1 bg-secondary border-none">
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">Select model</SelectItem>
                </SelectContent>
              </Select>
            </div>
          ))}

          {/* Added Models */}
          <div className="mt-8">
            <h3 className="text-lg font-medium mb-4">Added models</h3>
            <div className="space-y-3">
              {addedModels.map((model) => (
                <div
                  key={model.id}
                  className="flex items-center justify-between p-3 bg-card rounded-lg"
                >
                  <span className="font-medium">{model.name}</span>
                  <div className="flex items-center gap-2">
                    <Button variant="outline" size="sm">
                      Share
                    </Button>
                    <Badge
                      variant={model.hasApiKey ? "default" : "secondary"}
                      className="text-xs"
                    >
                      API-Key
                    </Badge>
                    <Button variant="outline" size="sm">
                      View models
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Available Models */}
        <div>
          <h3 className="text-lg font-medium mb-4">Available models</h3>
          <div className="space-y-3">
            {availableProviders.map((provider) => (
              <Card
                key={provider.id}
                className="bg-card hover:bg-secondary/30 transition-colors cursor-pointer"
              >
                <CardContent className="p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center">
                      <span className="text-sm font-bold">
                        {provider.name.charAt(0)}
                      </span>
                    </div>
                    <span className="font-medium">{provider.name}</span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {provider.capabilities.map((cap) => (
                      <Badge
                        key={cap}
                        variant="secondary"
                        className="text-xs"
                      >
                        {cap}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
