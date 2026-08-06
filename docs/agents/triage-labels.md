# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the status strings used by this repo's local org-mode issue tracker.

For local org tickets, write the mapped value in the ticket metadata as `#+STATUS: <value>`.

| Label in mattpocock/skills | Status in this tracker | Meaning                                  |
| -------------------------- | ---------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`         | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`           | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`      | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`      | Requires human implementation            |
| `wontfix`                  | `wontfix`              | Will not be actioned                     |

When a skill mentions a role, use the corresponding status string from this table.
