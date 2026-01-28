"use client";

import { Card, CardContent } from "@/components/ui/card";

const dataSources = [
  {
    id: "confluence",
    name: "Confluence",
    description: "Integrate your Confluence workspace to search documentation.",
    icon: "🔗",
    color: "bg-blue-500/10",
  },
  {
    id: "s3",
    name: "S3",
    description: "Connect to your AWS S3 bucket to import and sync stored files.",
    icon: "☁️",
    color: "bg-orange-500/10",
  },
  {
    id: "google-drive",
    name: "Google Drive",
    description: "Connect your Google Drive via OAuth and sync specific folders or drives.",
    icon: "📁",
    color: "bg-green-500/10",
  },
  {
    id: "discord",
    name: "Discord",
    description: "Link your Discord server to access and analyze chat data.",
    icon: "💬",
    color: "bg-indigo-500/10",
  },
  {
    id: "notion",
    name: "Notion",
    description: "Sync pages and databases from Notion for knowledge retrieval.",
    icon: "📝",
    color: "bg-gray-500/10",
  },
  {
    id: "jira",
    name: "Jira",
    description: "Connect your Jira workspace to sync issues, comments, and attachments.",
    icon: "🎫",
    color: "bg-blue-600/10",
  },
];

export default function DataSourcesPage() {
  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-semibold">Data sources</h1>
        <p className="text-muted-foreground">
          Manage your data source and connections
        </p>
      </div>

      <div className="mb-8">
        <h2 className="text-lg font-medium mb-2">Available sources</h2>
        <p className="text-sm text-muted-foreground mb-4">
          Select a data source to add
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {dataSources.map((source) => (
            <Card
              key={source.id}
              className="bg-card hover:bg-secondary/30 transition-colors cursor-pointer"
            >
              <CardContent className="flex items-start gap-4 p-4">
                <div className={`p-3 rounded-lg ${source.color}`}>
                  <span className="text-2xl">{source.icon}</span>
                </div>
                <div>
                  <h3 className="font-medium">{source.name}</h3>
                  <p className="text-sm text-muted-foreground">
                    {source.description}
                  </p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
