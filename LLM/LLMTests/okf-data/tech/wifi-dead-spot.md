---
type: Problem
title: The dead spot in the back bedroom
description: The back bedroom is the far corner from the router with a
  chimney in between, and it is also the home office, which is why it
  mattered.
tags: [tech, wifi, problem, office]
---

Video calls dropped two or three times a day in the back bedroom, which is
[the office](/person/work.md).

Diagnosis: it is the far corner of the house diagonally from the basement
router, and the masonry chimney is directly in the path. Signal read -78
dBm, which is barely usable, and the phone kept clinging to the basement
node rather than roaming.

What did not fix it: moving the router, a better antenna, a wifi extender
in the hall, which halved the throughput as extenders do.

What fixed it: a [mesh node](/tech/wifi-mesh.md) upstairs with a wired
backhaul. -52 dBm now.

The [ethernet run](/tech/ethernet-runs.md) to get there was the actual
work.
