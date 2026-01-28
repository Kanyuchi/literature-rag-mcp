"use client";

import { useState } from "react";
import { UserPlus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

const joinedTeams = [
  {
    id: "1",
    name: "LitRAG",
    date: "25/01/2026 08:13:55",
    email: "user@example.com",
  },
];

export default function TeamPage() {
  const [searchMembers, setSearchMembers] = useState("");
  const [searchTeams, setSearchTeams] = useState("");

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-semibold">LitRAG workspace</h1>
      </div>

      {/* Team members */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-medium">Team members</h2>
          <div className="flex items-center gap-3">
            <Input
              placeholder="Search..."
              value={searchMembers}
              onChange={(e) => setSearchMembers(e.target.value)}
              className="w-48 bg-secondary border-none"
            />
            <Button>
              <UserPlus className="h-4 w-4 mr-2" />
              Invite member
            </Button>
          </div>
        </div>

        <div className="rounded-lg border border-border overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow className="bg-card hover:bg-card">
                <TableHead>Name</TableHead>
                <TableHead>
                  <div className="flex items-center gap-1">
                    Date
                    <span className="text-xs">↕</span>
                  </div>
                </TableHead>
                <TableHead>Email</TableHead>
                <TableHead>State</TableHead>
                <TableHead>Action</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              <TableRow>
                <TableCell colSpan={5} className="text-center py-8 text-muted-foreground">
                  No data
                </TableCell>
              </TableRow>
            </TableBody>
          </Table>
        </div>
      </div>

      {/* Joined teams */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-medium">Joined teams</h2>
          <Input
            placeholder="Search..."
            value={searchTeams}
            onChange={(e) => setSearchTeams(e.target.value)}
            className="w-48 bg-secondary border-none"
          />
        </div>

        <div className="rounded-lg border border-border overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow className="bg-card hover:bg-card">
                <TableHead>Name</TableHead>
                <TableHead>
                  <div className="flex items-center gap-1">
                    Date
                    <span className="text-xs">↕</span>
                  </div>
                </TableHead>
                <TableHead>Email</TableHead>
                <TableHead>Action</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {joinedTeams.map((team) => (
                <TableRow key={team.id}>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <Avatar className="h-6 w-6">
                        <AvatarFallback className="bg-primary text-primary-foreground text-xs">
                          {team.name.charAt(0)}
                        </AvatarFallback>
                      </Avatar>
                      <span>{team.name}</span>
                    </div>
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {team.date}
                  </TableCell>
                  <TableCell className="text-muted-foreground">
                    {team.email}
                  </TableCell>
                  <TableCell>-</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>
    </div>
  );
}
