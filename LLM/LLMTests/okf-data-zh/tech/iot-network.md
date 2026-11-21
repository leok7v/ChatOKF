---
type: Note
title: 给小电器单独开的网络
description: 智能灯泡、门铃、电视和温控器住在自己的SSID上，这样它们看不见笔记本。
tags: [tech, network, security, smart-home]
---

第二个SSID，开访客式隔离，所有便宜又联网的东西都住在上面：
[灯泡](/tech/smart-bulbs.md)、[门铃](/tech/doorbell-camera.md)、
[扫地机器人](/tech/robot-vacuum.md)、[温控器](/house/smart-thermostat.md)、
[电视](/tech/smart-tv-livingroom.md)。

理由不是疑神疑鬼，而是这些设备大概只有两年的安全更新，之后再也没有，
可它们会在网络上待十年。

烦人的地方：手机投屏到电视需要两边在同一个网络上，
所以投屏的时候手机要连到小电器那个网络。有一个跨SSID发现的设置，
大概一半的时候管用。

笔记本、手机和[备份盒子](/tech/nas-backup.md)留在主网络上。
