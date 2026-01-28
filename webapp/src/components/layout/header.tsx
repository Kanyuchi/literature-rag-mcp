"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Home,
  Database,
  MessageSquare,
  Search,
  Bot,
  FolderOpen,
  Globe,
  Clock,
  RefreshCw,
  User,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";

const navigation = [
  { name: "Home", href: "/", icon: Home },
  { name: "Dataset", href: "/datasets", icon: Database },
  { name: "Chat", href: "/chat", icon: MessageSquare },
  { name: "Search", href: "/search", icon: Search },
  { name: "Agent", href: "/agents", icon: Bot },
  { name: "File Management", href: "/files", icon: FolderOpen },
];

export function Header() {
  const pathname = usePathname();

  const isActive = (href: string) => {
    if (href === "/") {
      return pathname === "/";
    }
    return pathname.startsWith(href);
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex h-16 items-center px-6">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2 mr-8">
          <div className="flex items-center justify-center">
            <svg
              width="32"
              height="32"
              viewBox="0 0 32 32"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              className="text-primary"
            >
              <path
                d="M8 8h4v4H8V8zm6 0h4v4h-4V8zm6 0h4v4h-4V8zM8 14h4v4H8v-4zm6 0h4v4h-4v-4zm6 0h4v4h-4v-4zM8 20h4v4H8v-4zm6 0h4v4h-4v-4zm6 0h4v4h-4v-4z"
                fill="currentColor"
              />
            </svg>
          </div>
        </Link>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1">
          {navigation.map((item) => {
            const Icon = item.icon;
            const active = isActive(item.href);
            return (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-colors",
                  active
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground hover:bg-secondary"
                )}
              >
                {item.name === "Home" ? (
                  <Icon className="h-4 w-4" />
                ) : (
                  <>
                    {item.name}
                  </>
                )}
                {item.name === "Home" && !active && null}
                {item.name !== "Home" && null}
              </Link>
            );
          })}
        </nav>

        {/* Right side actions */}
        <div className="ml-auto flex items-center gap-2">
          <Button variant="ghost" size="icon" className="text-muted-foreground">
            <Globe className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="icon" className="text-muted-foreground">
            <RefreshCw className="h-4 w-4" />
          </Button>
          <span className="text-sm text-muted-foreground">English</span>
          <Button variant="ghost" size="icon" className="text-muted-foreground">
            <Clock className="h-4 w-4" />
          </Button>

          {/* User Menu */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="rounded-full">
                <Avatar className="h-8 w-8">
                  <AvatarFallback className="bg-primary text-primary-foreground text-xs">
                    U
                  </AvatarFallback>
                </Avatar>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              <DropdownMenuItem asChild>
                <Link href="/settings/profile">
                  <User className="mr-2 h-4 w-4" />
                  Profile
                </Link>
              </DropdownMenuItem>
              <DropdownMenuItem asChild>
                <Link href="/settings/data-sources">
                  Data Sources
                </Link>
              </DropdownMenuItem>
              <DropdownMenuItem asChild>
                <Link href="/settings/model-providers">
                  Model Providers
                </Link>
              </DropdownMenuItem>
              <DropdownMenuItem asChild>
                <Link href="/settings/mcp">
                  MCP Servers
                </Link>
              </DropdownMenuItem>
              <DropdownMenuItem asChild>
                <Link href="/settings/team">
                  Team
                </Link>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </header>
  );
}
