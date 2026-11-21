---
type: Reference
title: 值得记住的路由器设置
description: 信道的选择、DHCP保留，还有两个出厂默认打开但不该开的设置。
tags: [tech, network, router, reference]
---

- 2.4 GHz用信道1，20 MHz宽。十一家邻居在6和11上。
- 5 GHz自动，80 MHz。
- 频段引导开，这样一个网络名就能同时服务两个频段。
- UPnP关。这里没东西需要它。
- 远程管理关。出厂是开着的。
- WPS关，同样的理由。

按MAC做DHCP保留的有：[NAS](/tech/nas-backup.md)、[打印机](/tech/printer.md)
和[家庭服务器](/tech/home-server.md)，这样它们的地址就不再变了。

管理员密码在[密码管理器](/tech/password-manager.md)里，不在贴纸上。

这些机器的固件更新是手动的。一年查两次。
