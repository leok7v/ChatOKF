---
type: Reference
title: Router settings worth remembering
description: Channel choices, DHCP reservations and the two settings
  that were on by default and should not have been.
tags: [tech, network, router, reference]
---

- 2.4 GHz on channel 1, 20 MHz wide. Eleven neighbors are on 6 and 11.
- 5 GHz on auto, 80 MHz.
- Band steering on, which is what lets one network name serve both.
- UPnP off. Nothing here needs it.
- Remote management off. It was on out of the box.
- WPS off, same reason.

DHCP reservations by MAC for: [the NAS](/tech/nas-backup.md),
[the printer](/tech/printer.md), and
[the home server](/tech/home-server.md), so their addresses stop moving.

Admin password is in [the password manager](/tech/password-manager.md),
not on the sticker.

Firmware updates are manual on these. Check twice a year.
