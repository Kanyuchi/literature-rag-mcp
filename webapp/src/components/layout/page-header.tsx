"use client";

import { ReactNode } from "react";
import { Filter, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface PageHeaderProps {
  title: string;
  icon?: ReactNode;
  showFilter?: boolean;
  showSearch?: boolean;
  searchPlaceholder?: string;
  onSearchChange?: (value: string) => void;
  searchValue?: string;
  actionLabel?: string;
  actionIcon?: ReactNode;
  onAction?: () => void;
  children?: ReactNode;
}

export function PageHeader({
  title,
  icon,
  showFilter = true,
  showSearch = true,
  searchPlaceholder = "Search...",
  onSearchChange,
  searchValue,
  actionLabel,
  actionIcon = <Plus className="h-4 w-4" />,
  onAction,
  children,
}: PageHeaderProps) {
  return (
    <div className="flex items-center justify-between mb-6">
      <div className="flex items-center gap-3">
        {icon && <span className="text-muted-foreground">{icon}</span>}
        <h1 className="text-2xl font-semibold">{title}</h1>
      </div>

      <div className="flex items-center gap-3">
        {children}

        {showFilter && (
          <Button variant="ghost" size="icon" className="text-muted-foreground">
            <Filter className="h-4 w-4" />
          </Button>
        )}

        {showSearch && (
          <div className="relative">
            <Input
              type="search"
              placeholder={searchPlaceholder}
              className="w-48 bg-secondary border-none"
              value={searchValue}
              onChange={(e) => onSearchChange?.(e.target.value)}
            />
          </div>
        )}

        {actionLabel && (
          <Button onClick={onAction} className="gap-2">
            {actionIcon}
            {actionLabel}
          </Button>
        )}
      </div>
    </div>
  );
}
