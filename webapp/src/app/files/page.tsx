"use client";

import { useState } from "react";
import { FolderOpen, Upload, Folder, FileText, MoreHorizontal, Trash2, Eye } from "lucide-react";
import { PageHeader } from "@/components/layout/page-header";
import { Pagination } from "@/components/layout/pagination";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface FileItem {
  id: string;
  name: string;
  type: "folder" | "file";
  uploadDate: string;
  size: string;
  dataset: string;
}

// Mock data - in production, this would come from the API
const mockFiles: FileItem[] = [
  {
    id: "1",
    name: "knowledgebase",
    type: "folder",
    uploadDate: "04/06/2025 17:38:28",
    size: "0 B",
    dataset: "",
  },
];

export default function FilesPage() {
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);
  const [searchQuery, setSearchQuery] = useState("");

  const filteredFiles = mockFiles.filter((file) =>
    file.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const toggleSelect = (id: string) => {
    setSelectedFiles((prev) =>
      prev.includes(id) ? prev.filter((f) => f !== id) : [...prev, id]
    );
  };

  const toggleSelectAll = () => {
    if (selectedFiles.length === filteredFiles.length) {
      setSelectedFiles([]);
    } else {
      setSelectedFiles(filteredFiles.map((f) => f.id));
    }
  };

  return (
    <div>
      <PageHeader
        title="Files"
        icon={<FolderOpen className="h-6 w-6" />}
        showFilter={false}
        showSearch={true}
        searchPlaceholder="Search files..."
        searchValue={searchQuery}
        onSearchChange={setSearchQuery}
        actionLabel="Add file"
        actionIcon={<Upload className="h-4 w-4" />}
        onAction={() => {}}
      />

      {/* Files Table */}
      <div className="rounded-lg border border-border overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow className="bg-card hover:bg-card">
              <TableHead className="w-12">
                <Checkbox
                  checked={
                    filteredFiles.length > 0 &&
                    selectedFiles.length === filteredFiles.length
                  }
                  onCheckedChange={toggleSelectAll}
                />
              </TableHead>
              <TableHead>
                <div className="flex items-center gap-1">
                  Name
                  <span className="text-xs">↕</span>
                </div>
              </TableHead>
              <TableHead>
                <div className="flex items-center gap-1">
                  Upload Date
                  <span className="text-xs">↕</span>
                </div>
              </TableHead>
              <TableHead>
                <div className="flex items-center gap-1">
                  Size
                  <span className="text-xs">↕</span>
                </div>
              </TableHead>
              <TableHead>Dataset</TableHead>
              <TableHead className="w-20">Action</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredFiles.length > 0 ? (
              filteredFiles.map((file) => (
                <TableRow key={file.id} className="bg-background">
                  <TableCell>
                    <Checkbox
                      checked={selectedFiles.includes(file.id)}
                      onCheckedChange={() => toggleSelect(file.id)}
                    />
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      {file.type === "folder" ? (
                        <Folder className="h-4 w-4 text-yellow-500" />
                      ) : (
                        <FileText className="h-4 w-4 text-primary" />
                      )}
                      <span className="font-medium">{file.name}</span>
                    </div>
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {file.uploadDate}
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {file.size}
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {file.dataset || "-"}
                  </TableCell>
                  <TableCell>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="icon" className="h-8 w-8">
                          <MoreHorizontal className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem>
                          <Eye className="h-4 w-4 mr-2" />
                          View
                        </DropdownMenuItem>
                        <DropdownMenuItem className="text-destructive">
                          <Trash2 className="h-4 w-4 mr-2" />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </TableCell>
                </TableRow>
              ))
            ) : (
              <TableRow>
                <TableCell colSpan={6} className="text-center py-12 text-muted-foreground">
                  No files found
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>

      <Pagination
        total={filteredFiles.length}
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
