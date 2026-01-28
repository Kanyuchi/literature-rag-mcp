"use client";

import { Plus, Upload, Settings2 } from "lucide-react";
import { PageHeader } from "@/components/layout/page-header";
import { Pagination } from "@/components/layout/pagination";
import { Button } from "@/components/ui/button";
import { useState } from "react";

export default function McpPage() {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-semibold">MCP servers</h1>
        <p className="text-muted-foreground">
          Customize the list of MCP servers
        </p>
      </div>

      <div className="flex items-center justify-end gap-3 mb-6">
        <Button variant="outline">
          <Settings2 className="h-4 w-4 mr-2" />
          Bulk manage
        </Button>
        <Button variant="outline">
          <Upload className="h-4 w-4 mr-2" />
          Import
        </Button>
        <Button>
          <Plus className="h-4 w-4 mr-2" />
          Add MCP
        </Button>
      </div>

      {/* Empty state */}
      <div className="text-center py-24 text-muted-foreground border border-dashed border-border rounded-lg">
        <Settings2 className="h-12 w-12 mx-auto mb-4 opacity-50" />
        <p>No MCP servers configured</p>
        <p className="text-sm mt-2">
          Add an MCP server to extend functionality
        </p>
      </div>

      <Pagination
        total={0}
        page={page}
        pageSize={pageSize}
        onPageChange={setPage}
        onPageSizeChange={(size) => {
          setPageSize(size);
          setPage(1);
        }}
      />
    </div>
  );
}
