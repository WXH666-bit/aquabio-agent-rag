# 小型数据方案

## 当前原型

使用 Wikimedia Commons API 抓取 5 个类别各 1 张图：

- starfish / 海星
- sea urchin / 海胆
- sea cucumber / 海参
- scallop / 扇贝
- jellyfish / 水母

优点是规模极小、可重复下载、每张图都记录授权和作者。缺点是只有查询词形成的弱标签，不含检测框。

## 正式实验建议

| 数据 | 任务 | 建议规模 | 备注 |
|---|---|---:|---|
| UIEB | 水下增强 | 50 对 | 仅从官方入口下载，不在项目中再分发 |
| Aquarium Data COTS | 目标检测演示 | 100-200 张 | 638 张、7 类、带 bbox，适合快速微调 YOLO |
| DUO | 海底生物检测 benchmark | 100-300 张或完整集 | 4 类：holothurian、echinus、scallop、starfish |
| WoRMS | 物种知识 | 4-20 类 | 用官方 REST API 补标准学名和分类 |

建议第一轮只训练 Aquarium 或 DUO 中的一套类别，避免 VLM、检测器和知识卡片出现类别空间不一致。

## PDF

- UIEB 论文：https://arxiv.org/abs/1901.05495
- DUO 论文/仓库：https://github.com/chongweiliu/DUO

## 重要限制

UIEB 官方项目页写明数据仅供非商业研究使用，且不得再分发。原型下载脚本只抓论文，不抓 UIEB 数据。

来源：

- https://li-chongyi.github.io/proj_benchmark.html
- https://public.roboflow.com/object-detection/aquarium
- https://github.com/chongweiliu/DUO
- https://www.marinespecies.org/rest/
- https://commons.wikimedia.org/wiki/Commons:API

