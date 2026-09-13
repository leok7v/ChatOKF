---
type: Project
title: Backups, and the 3-2-1 rule
description: Three copies, two kinds of media, one off site, which in
  practice is the laptops, a NAS in the basement and a cloud sync.
tags: [tech, backup, storage, data]
---

Three copies of anything that matters, on two different kinds of storage,
with one copy off site. That is the whole rule and everything else is
implementation.

Here:

1. The laptops themselves.
2. A two-bay NAS in the basement, mirrored, doing automatic Time Machine
   style backups over [ethernet](/tech/ethernet-runs.md).
3. An encrypted cloud sync of the irreplaceable folders only: photos,
   documents, tax records. Not the whole disk, because
   [the upload](/tech/isp-plan.md) is 35 Mbps.

A mirrored NAS is not a backup. It survives a dead drive, not a deleted
file or a fire.

Test a restore once a year. Not doing this is the most common way backups
turn out not to exist.
