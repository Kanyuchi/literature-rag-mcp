"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Database, Cpu, Server, Users, User } from "lucide-react";
import { cn } from "@/lib/utils";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";

const sidebarItems = [
  { name: "Data sources", href: "/settings/data-sources", icon: Database },
  { name: "Model providers", href: "/settings/model-providers", icon: Cpu },
  { name: "MCP", href: "/settings/mcp", icon: Server },
  { name: "Team", href: "/settings/team", icon: Users },
  { name: "Profile", href: "/settings/profile", icon: User },
];

export default function SettingsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();

  return (
    <div className="flex gap-8">
      {/* Sidebar */}
      <aside className="w-56 shrink-0">
        <div className="sticky top-24 space-y-6">
          {/* Profile section */}
          <div className="flex items-center gap-3 px-3">
            <Avatar className="h-10 w-10">
              <AvatarFallback className="bg-primary text-primary-foreground">
                U
              </AvatarFallback>
            </Avatar>
            <div className="text-sm">
              <p className="font-medium">user@example.com</p>
            </div>
          </div>

          {/* Navigation */}
          <nav className="space-y-1">
            {sidebarItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors",
                    isActive
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:text-foreground hover:bg-secondary"
                  )}
                >
                  <Icon className="h-4 w-4" />
                  {item.name}
                </Link>
              );
            })}
          </nav>

          {/* Theme toggle placeholder */}
          <div className="px-3 pt-4 border-t border-border">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span>Theme</span>
            </div>
          </div>

          {/* Logout */}
          <div className="px-3">
            <button className="text-sm text-muted-foreground hover:text-foreground">
              Log out
            </button>
          </div>
        </div>
      </aside>

      {/* Content */}
      <div className="flex-1 min-w-0">{children}</div>
    </div>
  );
}
