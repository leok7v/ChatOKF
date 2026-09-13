---
type: Project
title: The mesh wifi setup
description: Three TP-Link Deco X55 units wired back to the router,
  which fixed the dead spots that a single router never could.
tags: [tech, wifi, network, home]
---

Wi-Fi carries the network over radio instead of wires, and radio does not
go through a 1994 house with plaster and a brick chimney in the middle of
it.

Three TP-Link Deco X55 units: one at the router in the basement, one in
the living room, one upstairs in the hall.

The thing that made it work was wiring two of them back to the router over
[ethernet](/tech/ethernet-runs.md) instead of letting them relay
wirelessly. A mesh node that backhauls over wifi spends half its radio
time talking to the other nodes. Wired backhaul roughly doubled the
throughput upstairs.

Mesh networking means each node talks to more than one other node and the
traffic finds a path. It is reliable because there is more than one route,
not because the radios are better.

Single network name, devices roam between nodes on their own.

Gadgets are on [a separate network](/tech/iot-network.md).
