"use client";

import { useState } from "react";
import { Edit2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export default function ProfilePage() {
  const [name, setName] = useState("User");
  const [timezone, setTimezone] = useState("UTC+0");

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-semibold">Profile</h1>
        <p className="text-muted-foreground">
          Update your photo and personal details here.
        </p>
      </div>

      <div className="space-y-6 max-w-xl">
        {/* Name */}
        <div className="flex items-center justify-between py-4 border-b border-border">
          <div>
            <label className="text-sm font-medium">Name</label>
          </div>
          <div className="flex items-center gap-3">
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-64 bg-secondary border-none"
            />
            <Button variant="ghost" size="sm">
              <Edit2 className="h-4 w-4 mr-1" />
              Edit
            </Button>
          </div>
        </div>

        {/* Avatar */}
        <div className="flex items-center justify-between py-4 border-b border-border">
          <div>
            <label className="text-sm font-medium">Avatar</label>
          </div>
          <div className="flex items-center gap-3">
            <Avatar className="h-12 w-12">
              <AvatarFallback className="bg-secondary text-foreground">
                {name.charAt(0).toUpperCase()}
              </AvatarFallback>
            </Avatar>
            <span className="text-sm text-muted-foreground">
              This will be displayed on your profile.
            </span>
          </div>
        </div>

        {/* Time zone */}
        <div className="flex items-center justify-between py-4 border-b border-border">
          <div>
            <label className="text-sm font-medium">Time zone</label>
          </div>
          <div className="flex items-center gap-3">
            <Select value={timezone} onValueChange={setTimezone}>
              <SelectTrigger className="w-64 bg-secondary border-none">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="UTC-12">UTC-12</SelectItem>
                <SelectItem value="UTC-8">UTC-8 Pacific</SelectItem>
                <SelectItem value="UTC-5">UTC-5 Eastern</SelectItem>
                <SelectItem value="UTC+0">UTC+0 London</SelectItem>
                <SelectItem value="UTC+1">UTC+1 Berlin</SelectItem>
                <SelectItem value="UTC+8">UTC+8 Shanghai</SelectItem>
                <SelectItem value="UTC+9">UTC+9 Tokyo</SelectItem>
              </SelectContent>
            </Select>
            <Button variant="ghost" size="sm">
              <Edit2 className="h-4 w-4 mr-1" />
              Edit
            </Button>
          </div>
        </div>

        {/* Email */}
        <div className="flex items-center justify-between py-4 border-b border-border">
          <div>
            <label className="text-sm font-medium">Email</label>
          </div>
          <div>
            <p className="text-sm">user@example.com</p>
            <p className="text-xs text-muted-foreground">
              Once registered, E-mail cannot be changed.
            </p>
          </div>
        </div>

        {/* Password */}
        <div className="flex items-center justify-between py-4 border-b border-border">
          <div>
            <label className="text-sm font-medium">Password</label>
          </div>
          <div className="flex items-center gap-3">
            <Input
              type="password"
              value="********"
              disabled
              className="w-64 bg-secondary border-none"
            />
            <Button variant="ghost" size="sm">
              <Edit2 className="h-4 w-4 mr-1" />
              Edit
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
