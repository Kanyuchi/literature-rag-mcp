"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface PaginationProps {
  total: number;
  page: number;
  pageSize: number;
  onPageChange: (page: number) => void;
  onPageSizeChange: (size: number) => void;
}

export function Pagination({
  total,
  page,
  pageSize,
  onPageChange,
  onPageSizeChange,
}: PaginationProps) {
  const totalPages = Math.ceil(total / pageSize);

  return (
    <div className="flex items-center justify-end gap-4 mt-6 text-sm text-muted-foreground">
      <span>Total {total}</span>

      <div className="flex items-center gap-1">
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          disabled={page <= 1}
          onClick={() => onPageChange(page - 1)}
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>

        <span className="px-2">{page}</span>

        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          disabled={page >= totalPages}
          onClick={() => onPageChange(page + 1)}
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>

      <Select
        value={String(pageSize)}
        onValueChange={(value) => onPageSizeChange(Number(value))}
      >
        <SelectTrigger className="w-24 h-8 bg-secondary border-none">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="10">10 /Page</SelectItem>
          <SelectItem value="20">20 /Page</SelectItem>
          <SelectItem value="50">50 /Page</SelectItem>
          <SelectItem value="100">100 /Page</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
}
