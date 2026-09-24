# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.15.0](https://github.com/0verL1nk/PaperSage/compare/v1.14.0...v1.15.0) (2026-09-24)


### Features

* **agent:** evidence tiering (P2), citation audit (P3-lite), production feedback loop ([#177](https://github.com/0verL1nk/PaperSage/issues/177)) ([f7c6377](https://github.com/0verL1nk/PaperSage/commit/f7c63773778f11d6a969f940bfed29101b8978f5))


### Bug Fixes

* **desktop:** blockmaps are optional in the R2 hop ([#174](https://github.com/0verL1nk/PaperSage/issues/174)) ([0486cfe](https://github.com/0verL1nk/PaperSage/commit/0486cfe93b668603a1a32acc52d1894fb1cb8e6e))


### Documentation

* **agent:** add machine-readable readiness card ([#149](https://github.com/0verL1nk/PaperSage/issues/149)) ([b1ad685](https://github.com/0verL1nk/PaperSage/commit/b1ad685f9997a1d6ef769c505363e152882f4552))
* **evals:** P2 tiering A/B verdict - rejected, stays opt-in ([#178](https://github.com/0verL1nk/PaperSage/issues/178)) ([0d31445](https://github.com/0verL1nk/PaperSage/commit/0d314451ecd3f4004522f42c30861075059361e6))

## [1.14.0](https://github.com/0verL1nk/PaperSage/compare/v1.13.1...v1.14.0) (2026-08-30)


### Features

* add async policy routing with configurable router model ([575d8f8](https://github.com/0verL1nk/PaperSage/commit/575d8f86e13e812aa52159fd9806d9569d528bdb))
* add emit_tool_load_event flag to control tool load event emissions ([83a745c](https://github.com/0verL1nk/PaperSage/commit/83a745cce4d8f6bd1d55f47d9c99dce8aaa2837c))
* add environment variables for Claude Code and PR Review workflows ([ca4f709](https://github.com/0verL1nk/PaperSage/commit/ca4f70957675aade15f977ceb48353e655767a26))
* add GitHub desktop updates ([a13e3b6](https://github.com/0verL1nk/PaperSage/commit/a13e3b6f05e5db917cf70c9025f58c1198d11ac1))
* add GitHub desktop updates ([795f580](https://github.com/0verL1nk/PaperSage/commit/795f580d5df7667c5e21ff8efa47b985f680be7e))
* add list_document tool to agent capabilities ([76e1d57](https://github.com/0verL1nk/PaperSage/commit/76e1d57d26c03fa54b3e03d1be8a7fd608f3bdac))
* add localized document evidence previews ([#61](https://github.com/0verL1nk/PaperSage/issues/61)) ([6b61cec](https://github.com/0verL1nk/PaperSage/commit/6b61cec6f532abc7bf8211b8287037f653298d54))
* add MinerU document parsing integration ([#5](https://github.com/0verL1nk/PaperSage/issues/5)) ([1b77b77](https://github.com/0verL1nk/PaperSage/commit/1b77b7732b010f880d9725d292c8d748db951e05))
* Add support for fetching PR head branch in cross-repository scenarios ([7472ad5](https://github.com/0verL1nk/PaperSage/commit/7472ad5c19a5151d7a55b5093231a85f08af0a7b))
* add tool search ([#15](https://github.com/0verL1nk/PaperSage/issues/15)) ([d4a6be1](https://github.com/0verL1nk/PaperSage/commit/d4a6be106a613691bb2454fb6b2e9fdc7640b857))
* advance durable research runtime ([032aaa3](https://github.com/0verL1nk/PaperSage/commit/032aaa31dc58ba538c689bf4c72bf0b99aff51c2))
* advance durable research runtime ([7e46967](https://github.com/0verL1nk/PaperSage/commit/7e4696720fd7e78039126f0a35f3ffdba848b5d1))
* agent centric orchestration ([#20](https://github.com/0verL1nk/PaperSage/issues/20)) ([a4a8416](https://github.com/0verL1nk/PaperSage/commit/a4a84167580ca91181297b482590bd81c95a6932))
* **agent:** builtin research subagents with agentic search and paper ingestion ([#144](https://github.com/0verL1nk/PaperSage/issues/144)) ([2703028](https://github.com/0verL1nk/PaperSage/commit/2703028fdbb9769af47e049b83e92aca2d0e3772))
* **agent:** Codex-parity subagent controls ([#147](https://github.com/0verL1nk/PaperSage/issues/147)) ([9dde03e](https://github.com/0verL1nk/PaperSage/commit/9dde03e93f113e8cee93e49195a5a2e0779572c1))
* automate release version selection ([835afca](https://github.com/0verL1nk/PaperSage/commit/835afca2c11a3296821728c4f7a4584e9d2a40d6))
* complete checkpointer-based summarization migration ([#18](https://github.com/0verL1nk/PaperSage/issues/18)) ([673bba7](https://github.com/0verL1nk/PaperSage/commit/673bba7e55b4dc67fea3cf9cf0b5ff63ef364ee8))
* **desktop:** resilient update checks with mirror fallback and visible failures ([#157](https://github.com/0verL1nk/PaperSage/issues/157)) ([63876a9](https://github.com/0verL1nk/PaperSage/commit/63876a9285baa987a9b19bf55611b3d458026f03))
* Enhance document handling and retrieval capabilities ([5491042](https://github.com/0verL1nk/PaperSage/commit/54910423f01cf5dac7760da8a0a60259d4a12060))
* Enhance project with new cleanup and revision policies ([51688f2](https://github.com/0verL1nk/PaperSage/commit/51688f21faf69ae65f7d22fae7e3d7df12eba3ef))
* enhance prompt handling and memory management in agent center ([924e012](https://github.com/0verL1nk/PaperSage/commit/924e012672674e14dd49c72b09fde4a5f86a3825))
* enhance team runtime output with native report rendering ([50de246](https://github.com/0verL1nk/PaperSage/commit/50de246f49a22b5c9f4ce59e4624c887e63ff22f))
* **evals:** live task-completion eval system, agent fixes, and docs knowledge base ([b3c0a06](https://github.com/0verL1nk/PaperSage/commit/b3c0a0617d501b454032f85208c6988e81222637))
* extract vector store builder into dedicated module ([#8](https://github.com/0verL1nk/PaperSage/issues/8)) ([b06edba](https://github.com/0verL1nk/PaperSage/commit/b06edba662817df85f3d0dd20430a1686c7ecc6a))
* implement lazy tool activation and usage with structured arguments ([b403576](https://github.com/0verL1nk/PaperSage/commit/b40357653d77b4e564da0eab6fdc0e1db744f528))
* implement session selector UID override logic and corresponding tests ([754e99c](https://github.com/0verL1nk/PaperSage/commit/754e99c7d435a84f4bdec09db573a5f3e216f304))
* Improve orchestration and planning modes with enhanced documentation and structured prompts ([872695e](https://github.com/0verL1nk/PaperSage/commit/872695e93c58bc6ca0b1fc2ebf9dee05b8a3eb10))
* log final answer in execute_turn_core and add corresponding test ([985a8c3](https://github.com/0verL1nk/PaperSage/commit/985a8c3c5be42cf3548603fab157734e50ea1a52))
* make research surfaces complement answers ([#67](https://github.com/0verL1nk/PaperSage/issues/67)) ([d8dc12d](https://github.com/0verL1nk/PaperSage/commit/d8dc12d1c1e22fd771aef5017d58930814f01557))
* plan mode cleanup ([#12](https://github.com/0verL1nk/PaperSage/issues/12)) ([bef6dd8](https://github.com/0verL1nk/PaperSage/commit/bef6dd8b292372c01a73cc06950df643fa99253d))
* progressive tool disclosure and relevance-based document retrieval ([#9](https://github.com/0verL1nk/PaperSage/issues/9)) ([77a2e84](https://github.com/0verL1nk/PaperSage/commit/77a2e84659a89556249ac20e004db6500a6417a0))
* refactor agent center workspace and context observability ([0da84a1](https://github.com/0verL1nk/PaperSage/commit/0da84a1bdb8f009209c44a8d761ba187c6f10800))
* replace ANTHROPIC_AUTH_TOKEN with ANTHROPIC_API_KEY in workflow files ([a941db3](https://github.com/0verL1nk/PaperSage/commit/a941db3cac2e8c06778ab7733ffb8a718c77c92d))
* replace CLAUDE_CODE_OAUTH_TOKEN with ANTHROPIC_API_KEY in workflow files ([91b2fbb](https://github.com/0verL1nk/PaperSage/commit/91b2fbbf506af7e6e2a8e08586739f912b6537d6))
* replay research surfaces by identity ([#68](https://github.com/0verL1nk/PaperSage/issues/68)) ([bf304fc](https://github.com/0verL1nk/PaperSage/commit/bf304fc115523c4265c8a10dbcffc3cf27a83138))
* **research:** model-generated follow-up suggestions per session ([#131](https://github.com/0verL1nk/PaperSage/issues/131)) ([476f5c3](https://github.com/0verL1nk/PaperSage/commit/476f5c397ec86165e5319b27d612618ad1109104))
* **skills:** user-overlay skill directory with priority resolution ([#111](https://github.com/0verL1nk/PaperSage/issues/111)) ([8c866f4](https://github.com/0verL1nk/PaperSage/commit/8c866f4b73746908eb5f9487ddb76eefafdbbdd0))
* slash command palette with builtin commands and skills ([#152](https://github.com/0verL1nk/PaperSage/issues/152)) ([18558e1](https://github.com/0verL1nk/PaperSage/commit/18558e10f497ebbd87dca667bf3e340aa1d56572))
* stream inline research surfaces ([b6447e7](https://github.com/0verL1nk/PaperSage/commit/b6447e77f8b88dcebecc7d76383e8442cafdef04))
* stream inline research surfaces ([4181f87](https://github.com/0verL1nk/PaperSage/commit/4181f87b233be83a139f1c449dce7760583f6e65))
* team architecture upgrade ([#28](https://github.com/0verL1nk/PaperSage/issues/28)) ([ade3cf8](https://github.com/0verL1nk/PaperSage/commit/ade3cf8f828333520d50aa068934b452e74ffe07))
* tool manifest lazy schema ([#6](https://github.com/0verL1nk/PaperSage/issues/6)) ([63621d2](https://github.com/0verL1nk/PaperSage/commit/63621d21cab1c0d1f606b0a1f1e112a7a7e98404))
* **web:** interleaved message timeline with inline citations ([#121](https://github.com/0verL1nk/PaperSage/issues/121)) ([2327f5d](https://github.com/0verL1nk/PaperSage/commit/2327f5d7c8b04f767e37bb36f485fdee4e80a324))
* **web:** interleaved reasoning/tool timeline with inline citations ([#130](https://github.com/0verL1nk/PaperSage/issues/130)) ([64b5e23](https://github.com/0verL1nk/PaperSage/commit/64b5e2352800a5bc4b217e770b3643b8c20d5fda))
* **web:** pop the mode switch from a compact active-mode indicator ([#134](https://github.com/0verL1nk/PaperSage/issues/134)) ([5da1301](https://github.com/0verL1nk/PaperSage/commit/5da1301ba375c40c86b1a9ed4cea1877fa91f267))
* **web:** visualize session context composition ([#113](https://github.com/0verL1nk/PaperSage/issues/113)) ([13dac58](https://github.com/0verL1nk/PaperSage/commit/13dac5892648e4699ca2da104702aa8ac7cdfd66))
* 更新发布工作流，启用自动生成发布说明并简化版本处理 ([41f08b4](https://github.com/0verL1nk/PaperSage/commit/41f08b4ae65d6d57e9a98d6c4307f8ee5895640f))
* 更新版本到1.0.0，添加新功能和修复，更新项目链接 ([6e9a649](https://github.com/0verL1nk/PaperSage/commit/6e9a6492b1c110461ead2cc431a04219d9588cf3))


### Bug Fixes

* **agent:** drop orphaned tool turns and keep the context card after compact ([#165](https://github.com/0verL1nk/PaperSage/issues/165)) ([0c0ce3d](https://github.com/0verL1nk/PaperSage/commit/0c0ce3d4d71c09f5f52e8169d1d5d19ab3b1e17a))
* **agent:** propagate mid-stream model failures instead of leaking them into answers ([#150](https://github.com/0verL1nk/PaperSage/issues/150)) ([0fbe799](https://github.com/0verL1nk/PaperSage/commit/0fbe799591fed8cb8d867e8a9508b5ca54b8142d))
* **agent:** salvage unterminated UI fragments and repair update_plan tool dispatch ([#142](https://github.com/0verL1nk/PaperSage/issues/142)) ([956ec13](https://github.com/0verL1nk/PaperSage/commit/956ec13ee92c17a7d5f5ffa01332787c99c32072))
* **agent:** sanitize replayed history at the provider boundary ([#160](https://github.com/0verL1nk/PaperSage/issues/160)) ([e2da9e6](https://github.com/0verL1nk/PaperSage/commit/e2da9e6108e1e2712d689e711dc32e095d792cf8))
* align PyPI package version ([6a47549](https://github.com/0verL1nk/PaperSage/commit/6a475494a6ec4ffff777caaeaf8621028889c542))
* align runtime package version ([a30b082](https://github.com/0verL1nk/PaperSage/commit/a30b082e343fde7a00ae132a5ba4262bdb2d4913))
* align streamdown shiki dependencies ([3be5a94](https://github.com/0verL1nk/PaperSage/commit/3be5a949e48f1d5c4da70dbd038f70497d4adda2))
* align streamdown shiki dependencies ([4fc18ce](https://github.com/0verL1nk/PaperSage/commit/4fc18cefdfc53347c121eff89a29d3c60dd59be0))
* allow bot-triggered claude review workflow ([10d5e45](https://github.com/0verL1nk/PaperSage/commit/10d5e45f3dfd4568a864b204779c4202436e3bce))
* allow manual PyPI publication ([62304a7](https://github.com/0verL1nk/PaperSage/commit/62304a7bddf3b219ae207e8c053ecfb2fe7fbe81))
* allow manual PyPI publication ([674367c](https://github.com/0verL1nk/PaperSage/commit/674367c0e108858747b201cdbe78e4b6b7d04cd6))
* bind updater asset upload to repository ([4159f5a](https://github.com/0verL1nk/PaperSage/commit/4159f5ab0223d6e7671a18d87572e1a2f47ac44c))
* centralize GitHub release asset publishing ([b827ace](https://github.com/0verL1nk/PaperSage/commit/b827aced1c1029326b232eae5bb5935707a41fc7))
* **ci:** release train must squash-merge under squash-only settings ([#103](https://github.com/0verL1nk/PaperSage/issues/103)) ([ca012b8](https://github.com/0verL1nk/PaperSage/commit/ca012b88a247657cc7c51c6fb6d3533ed0130c63))
* close release train state ([0d13cee](https://github.com/0verL1nk/PaperSage/commit/0d13ceea284d1f53d30d12674703f3265400b0ee))
* close release train state ([31bb64b](https://github.com/0verL1nk/PaperSage/commit/31bb64bbf0e1f7fdbfe44fa052881de4f09b1e3c))
* consolidate release assets ([#45](https://github.com/0verL1nk/PaperSage/issues/45)) ([f5bf5bf](https://github.com/0verL1nk/PaperSage/commit/f5bf5bfc3d61efe1ca978989df7f10db8d23f98f))
* dangling a2ui placeholders, dead renderer logging, context card design ([#136](https://github.com/0verL1nk/PaperSage/issues/136)) ([9f346e1](https://github.com/0verL1nk/PaperSage/commit/9f346e1b0e7324e1255bf4d023840fc2b684fd41))
* declare Electron package entry ([#41](https://github.com/0verL1nk/PaperSage/issues/41)) ([30bcd4b](https://github.com/0verL1nk/PaperSage/commit/30bcd4bc1c1a48fac5a305bb6a18477a8d119207))
* derive preview paths from owned documents ([#63](https://github.com/0verL1nk/PaperSage/issues/63)) ([1069635](https://github.com/0verL1nk/PaperSage/commit/10696352e5f39241f5e78a916c8da41da8bcb55f))
* **desktop:** bundle agent skills and tolerate reasoning-wrapped JSON ([#109](https://github.com/0verL1nk/PaperSage/issues/109)) ([f7ced2d](https://github.com/0verL1nk/PaperSage/commit/f7ced2dd7a9f14cd0ee920cf9a17ba0fe8ce989f))
* **desktop:** bundle alembic so packaged backend can start ([eece72b](https://github.com/0verL1nk/PaperSage/commit/eece72bd403bc80f674e8ed21b7be21a37f3f968))
* **desktop:** bundle alembic with the packaged backend ([f8cba69](https://github.com/0verL1nk/PaperSage/commit/f8cba697a9d64cb7845e76ba20d0929cbe917268))
* **desktop:** copy NVIDIA DLLs explicitly in GPU bundles ([#126](https://github.com/0verL1nk/PaperSage/issues/126)) ([5f9761b](https://github.com/0verL1nk/PaperSage/commit/5f9761b199d1f8024ba2f6925c0e9cb58e2ec69b))
* **desktop:** gather scattered cu13 CUDA DLLs into one loader directory ([#129](https://github.com/0verL1nk/PaperSage/issues/129)) ([eb9f9ab](https://github.com/0verL1nk/PaperSage/commit/eb9f9ab4041c894cac1c54d0aa3d0153e89b9684))
* **desktop:** heal silent updates over legacy installs and exit cleanly ([#133](https://github.com/0verL1nk/PaperSage/issues/133)) ([19610bb](https://github.com/0verL1nk/PaperSage/commit/19610bb50620282a9e53aaf1a4d8fb7c5053f292))
* **desktop:** keep the launch window visible through backend boot ([#110](https://github.com/0verL1nk/PaperSage/issues/110)) ([12b408d](https://github.com/0verL1nk/PaperSage/commit/12b408d18227cb3c81fb16f1f35839c2a780fe31))
* **desktop:** keep updater metadata aligned with installer ([#80](https://github.com/0verL1nk/PaperSage/issues/80)) ([406acf1](https://github.com/0verL1nk/PaperSage/commit/406acf1825e84c98d812f3f6f23e58c01cd823e2))
* **desktop:** repair packaged ingestion, surface failures, install-dir model cache, GPU tier gating ([#117](https://github.com/0verL1nk/PaperSage/issues/117)) ([9c4af73](https://github.com/0verL1nk/PaperSage/commit/9c4af73b788fe988623cd092c4d97e2f83e3fefb))
* **desktop:** repair tray icon, add restart-to-update and visible version ([c4729a5](https://github.com/0verL1nk/PaperSage/commit/c4729a5fdce2aa22576a07d40963500bcbd200b2))
* **desktop:** repair tray icon, add restart-to-update and visible version ([f0d835b](https://github.com/0verL1nk/PaperSage/commit/f0d835b6ac5038b70dcf1298925ff53d775c3d1d))
* **desktop:** retry the single-instance lock across update relaunches ([#127](https://github.com/0verL1nk/PaperSage/issues/127)) ([ce5e047](https://github.com/0verL1nk/PaperSage/commit/ce5e04743407e29f7d125006fd7e811636c38c4a))
* **desktop:** serve updates from GitHub Releases; drop personal R2 CDN ([ba47735](https://github.com/0verL1nk/PaperSage/commit/ba477350e8c09031d8e4c1a0296d24ec63bc5e9c))
* **desktop:** use the Electron net module for the GPU pack download ([#138](https://github.com/0verL1nk/PaperSage/issues/138)) ([a51ddd6](https://github.com/0verL1nk/PaperSage/commit/a51ddd6c2657783a18d0c7c99a27b83ced8c1de7))
* **desktop:** use the Electron net module for the GPU pack download ([#140](https://github.com/0verL1nk/PaperSage/issues/140)) ([17219e2](https://github.com/0verL1nk/PaperSage/commit/17219e2e186650d8b838005a46b9e476c7d06d5d))
* discover unprefixed release tags ([14e33cc](https://github.com/0verL1nk/PaperSage/commit/14e33ccb18b17ee44ecc27d21b608eb93419fa89))
* discover unprefixed release tags ([e6166b7](https://github.com/0verL1nk/PaperSage/commit/e6166b7aed76b7001e1dc10c896dd5268d9170f7))
* dispatch canonical release inputs ([1635a9d](https://github.com/0verL1nk/PaperSage/commit/1635a9d98b1a5e44ec3071e75a73c56575652134))
* dispatch canonical release inputs ([5c23a2d](https://github.com/0verL1nk/PaperSage/commit/5c23a2d4ddcdc4a5bfd12c0e1cb3062d389f648d))
* enforce mindmap tag contract for parsing ([842a913](https://github.com/0verL1nk/PaperSage/commit/842a913c14428bb04123ba731035a077407e64fa))
* grant release label permission ([3821db1](https://github.com/0verL1nk/PaperSage/commit/3821db10aa55c383240796989c5243c573ed170f))
* grant release label permission ([26a79f4](https://github.com/0verL1nk/PaperSage/commit/26a79f4da82f2fcd6fd280506b77011d88a5c0f3))
* grant release workflow PR permissions ([#79](https://github.com/0verL1nk/PaperSage/issues/79)) ([c6c2c97](https://github.com/0verL1nk/PaperSage/commit/c6c2c97a56da81df5c4ba3ec429b3c28e16ab889))
* improve desktop update and chat recovery ([#77](https://github.com/0verL1nk/PaperSage/issues/77)) ([fa23b76](https://github.com/0verL1nk/PaperSage/commit/fa23b7663325071fb013310e369aa1459f0fef62))
* improve desktop update and installation flow ([22c8736](https://github.com/0verL1nk/PaperSage/commit/22c8736e33e39bc9b7e4a186eac36f8f78d7bcc3))
* improve desktop update experience ([17411d1](https://github.com/0verL1nk/PaperSage/commit/17411d1183b29ddae3ed1f94073b5cd4d9727d75))
* improve desktop update experience ([481f029](https://github.com/0verL1nk/PaperSage/commit/481f0298205835f82d88946b5c378c3b00440b7c))
* improve desktop update feedback ([863e80f](https://github.com/0verL1nk/PaperSage/commit/863e80fca5698fbbf5da2ffd16a71822a8321c7f))
* keep memory changes out of runtime release ([c8db72e](https://github.com/0verL1nk/PaperSage/commit/c8db72eb0d6da3726d4debd1b2e8996ea6bccabc))
* keep workspace within debt baseline ([08eb4ad](https://github.com/0verL1nk/PaperSage/commit/08eb4adc8b75d8319c94f4397fbbe4fdc2a62eca))
* lint issue ([39f8620](https://github.com/0verL1nk/PaperSage/commit/39f8620e039c210d7fb7bb8f29f5712f9dc9b03f))
* normalize Windows install directory ([f1a0658](https://github.com/0verL1nk/PaperSage/commit/f1a06584f224deb1247224528e9da862fe9177c3))
* **orm:** reconcile drifted agent_run_events schema ([78fc1ba](https://github.com/0verL1nk/PaperSage/commit/78fc1ba6a363bfcaa259bb5f03951e76d95f1db3))
* **orm:** reconcile drifted agent_run_events schema ([c1c7e46](https://github.com/0verL1nk/PaperSage/commit/c1c7e46e67ebd9306fa29a7081c1ee646d89e30b))
* **orm:** reconcile legacy event columns before run-item backfill ([#107](https://github.com/0verL1nk/PaperSage/issues/107)) ([fe3c16c](https://github.com/0verL1nk/PaperSage/commit/fe3c16caa0a4216922d3af6e2155e01c34f2e96c))
* preserve dependency lock versions ([bcd2095](https://github.com/0verL1nk/PaperSage/commit/bcd209540c2e1a24cb3184f4ee17e432e465fb9b))
* publish Linux deb package ([#44](https://github.com/0verL1nk/PaperSage/issues/44)) ([1551915](https://github.com/0verL1nk/PaperSage/commit/1551915ae41ce085b666fd3c4b47bb9c8619fed5))
* refine desktop workspace navigation ([f69ab3b](https://github.com/0verL1nk/PaperSage/commit/f69ab3b0aa18c8fde98961378bf0180a085eb9da))
* refine desktop workspace navigation ([65d5f21](https://github.com/0verL1nk/PaperSage/commit/65d5f21d83bcc333c411e84d407e4d6e694c6018))
* refresh project lock metadata ([8f6f7b5](https://github.com/0verL1nk/PaperSage/commit/8f6f7b55ac55c0f5b0f1fbabd1d0ef40fb777df9))
* release desktop OCR packaging metadata ([27e3111](https://github.com/0verL1nk/PaperSage/commit/27e31119ad452d9786415efeac347df9113ccba1))
* release desktop OCR packaging metadata ([6b22a40](https://github.com/0verL1nk/PaperSage/commit/6b22a4022e18d968584c6848892029412124da85))
* rename pages to ASCII, fix PyPI packaging and Windows CLI ([#7](https://github.com/0verL1nk/PaperSage/issues/7)) ([e46d0a1](https://github.com/0verL1nk/PaperSage/commit/e46d0a1629119b1287e930b70f3fd1948f28996e))
* repair cross-platform desktop release ([#43](https://github.com/0verL1nk/PaperSage/issues/43)) ([803b5ec](https://github.com/0verL1nk/PaperSage/commit/803b5ec7808f1402c357a8cadf5d37129d63c3d9))
* resolve type hints and improve code clarity across multiple files ([de78edd](https://github.com/0verL1nk/PaperSage/commit/de78eddfdcf780bfd35e5374fa6f51ee55f4cf86))
* restore runtime migration compatibility ([7a0c757](https://github.com/0verL1nk/PaperSage/commit/7a0c7578f17b65fded3650700f904dffc81aa796))
* route desktop updates through R2 ([d00ae51](https://github.com/0verL1nk/PaperSage/commit/d00ae51c5d74dd158a5487a89825b9462c577230))
* route desktop updates through R2 ([89c4c69](https://github.com/0verL1nk/PaperSage/commit/89c4c699660d7bf4d1f11ebd88f7fdf8bf94dca7))
* show summary text and enhance mindmap fullscreen theme ([27c6ba3](https://github.com/0verL1nk/PaperSage/commit/27c6ba3ac8060e9f0d0e4775d09f5dda5f8b9e26))
* sort runtime migration imports ([cc58e80](https://github.com/0verL1nk/PaperSage/commit/cc58e80556959b38a21e23ed6d203071a83e7a36))
* structured model JSON schema injection and UTF-8 backend stdio ([#115](https://github.com/0verL1nk/PaperSage/issues/115)) ([4c1deff](https://github.com/0verL1nk/PaperSage/commit/4c1deff9226822c0276c591514169354a63aa4be))
* synchronize runtime version during releases ([59eed79](https://github.com/0verL1nk/PaperSage/commit/59eed79cd711148c0ed946f39a9deeaaf8e44c79))
* trigger Claude workflow on PR commits ([#33](https://github.com/0verL1nk/PaperSage/issues/33)) ([3231082](https://github.com/0verL1nk/PaperSage/commit/3231082b93a941438abdf66b1a3122e0e98e259a))
* **ui:** remove mindmap iframe scrollbar via adaptive height ([#4](https://github.com/0verL1nk/PaperSage/issues/4)) ([052dabf](https://github.com/0verL1nk/PaperSage/commit/052dabf064c94a2c51a506902db24dc2bace9072))
* update Chroma import and clarify agent settings for max staleness ([18d3527](https://github.com/0verL1nk/PaperSage/commit/18d3527cf29044a83fb8a9c1c16488fe2af8634e))
* Update orchestration middleware to include team mode in complexity result ([5b52aa6](https://github.com/0verL1nk/PaperSage/commit/5b52aa6f438db832e6e8e5fc54a9987b2cef98be))
* upload large desktop updates to R2 ([036117a](https://github.com/0verL1nk/PaperSage/commit/036117a961eaa8a78230988ab83e63708690d951))
* upload large desktop updates to R2 ([e8a0e9a](https://github.com/0verL1nk/PaperSage/commit/e8a0e9a5209f24a224cf86092c8c1f8228aed62a))
* **web:** compact response indicator and null terminal result tolerance ([#112](https://github.com/0verL1nk/PaperSage/issues/112)) ([a6a48f1](https://github.com/0verL1nk/PaperSage/commit/a6a48f1bd735138c130760e49c7ca6300a3c53ed))
* **web:** keep a2ui mindmaps inline after a turn completes ([#139](https://github.com/0verL1nk/PaperSage/issues/139)) ([abc0f10](https://github.com/0verL1nk/PaperSage/commit/abc0f103828dce74d47ec9528e2e3224c3a919f3))
* **web:** keep code-block async tokens fresh without render-phase refs ([1aafbb1](https://github.com/0verL1nk/PaperSage/commit/1aafbb1278a76aa176324d175da4682eb24d9b99))
* **web:** merge duplicate settings entries into the account menu ([deb63d5](https://github.com/0verL1nk/PaperSage/commit/deb63d51b6e9f66576824e90603e370db86826b4))
* **web:** merge duplicate settings entries into the account menu ([d681296](https://github.com/0verL1nk/PaperSage/commit/d6812960cdd0676710261d87e34f967de9a5d8f7))
* **web:** readable tool rows in the assistant timeline with expandable detail ([#151](https://github.com/0verL1nk/PaperSage/issues/151)) ([aaf9730](https://github.com/0verL1nk/PaperSage/commit/aaf973042e43179dbcd9a022321083bd35ec5605))
* **web:** visible marker for unvalidated a2ui parts + turn-engine baseline reset ([#153](https://github.com/0verL1nk/PaperSage/issues/153)) ([0e38275](https://github.com/0verL1nk/PaperSage/commit/0e38275be3185a096710510317f00dc79fced8aa))
* 一些修复 ([6652d82](https://github.com/0verL1nk/PaperSage/commit/6652d82d1a7a732f4a24ff4cc8e53d22fded9652))


### Performance Improvements

* **desktop:** defer lancedb, fastembed and langchain_community out of startup ([#158](https://github.com/0verL1nk/PaperSage/issues/158)) ([c552421](https://github.com/0verL1nk/PaperSage/commit/c5524219f144480bee9393e86dce881db90419d3))


### Documentation

* add release and merge conventions ([#102](https://github.com/0verL1nk/PaperSage/issues/102)) ([62669af](https://github.com/0verL1nk/PaperSage/commit/62669af71e534674baea97904f65c6f29c3d4878))
* clarify product workflow and local setup ([68811c7](https://github.com/0verL1nk/PaperSage/commit/68811c770eaa741934f8e00abd2902c286be2e6f))
* **design:** add OpenPencil design workflow and account-menu v1.4.4 artifacts ([#105](https://github.com/0verL1nk/PaperSage/issues/105)) ([dd3eb5b](https://github.com/0verL1nk/PaperSage/commit/dd3eb5b8962b8c59e48cefc4a4a910b7df81bd9d))
* explain verifiable agent design ([572c983](https://github.com/0verL1nk/PaperSage/commit/572c983638e25742003629faad7d4de101d9c03b))
* refresh brand and project overview ([7a3f46e](https://github.com/0verL1nk/PaperSage/commit/7a3f46e1913084a8b97df2c9dc89c2c77d2680e7))
* release and CI debugging playbook ([#163](https://github.com/0verL1nk/PaperSage/issues/163)) ([b793a0f](https://github.com/0verL1nk/PaperSage/commit/b793a0ffe5765ba21aecdfa0a8a2448e56e8bdc7))
* restore and update project README ([#42](https://github.com/0verL1nk/PaperSage/issues/42)) ([b6a2f17](https://github.com/0verL1nk/PaperSage/commit/b6a2f173fac58230b3745b9a20a654e16c1fed72))
* specify natural A2UI output contract ([248ae70](https://github.com/0verL1nk/PaperSage/commit/248ae703ef5b2af9fdedda4fdfa7e4e7b1a3dc1e))

## [1.13.1](https://github.com/0verL1nk/PaperSage/compare/v1.13.0...v1.13.1) (2026-08-30)


### Bug Fixes

* **desktop:** serve updates from GitHub Releases; drop personal R2 CDN ([ba47735](https://github.com/0verL1nk/PaperSage/commit/ba477350e8c09031d8e4c1a0296d24ec63bc5e9c))

## [1.13.0](https://github.com/0verL1nk/PaperSage/compare/v1.12.1...v1.13.0) (2026-08-30)


### Features

* **evals:** live task-completion eval system, agent fixes, and docs knowledge base ([b3c0a06](https://github.com/0verL1nk/PaperSage/commit/b3c0a0617d501b454032f85208c6988e81222637))

## [1.12.1](https://github.com/0verL1nk/PaperSage/compare/v1.12.0...v1.12.1) (2026-08-23)


### Bug Fixes

* **agent:** drop orphaned tool turns and keep the context card after compact ([#165](https://github.com/0verL1nk/PaperSage/issues/165)) ([0c0ce3d](https://github.com/0verL1nk/PaperSage/commit/0c0ce3d4d71c09f5f52e8169d1d5d19ab3b1e17a))

## [1.12.0](https://github.com/0verL1nk/PaperSage/compare/v1.11.1...v1.12.0) (2026-08-23)


### Features

* slash command palette with builtin commands and skills ([#152](https://github.com/0verL1nk/PaperSage/issues/152)) ([18558e1](https://github.com/0verL1nk/PaperSage/commit/18558e10f497ebbd87dca667bf3e340aa1d56572))


### Documentation

* release and CI debugging playbook ([#163](https://github.com/0verL1nk/PaperSage/issues/163)) ([b793a0f](https://github.com/0verL1nk/PaperSage/commit/b793a0ffe5765ba21aecdfa0a8a2448e56e8bdc7))

## [1.11.1](https://github.com/0verL1nk/PaperSage/compare/v1.11.0...v1.11.1) (2026-08-23)


### Bug Fixes

* **agent:** sanitize replayed history at the provider boundary ([#160](https://github.com/0verL1nk/PaperSage/issues/160)) ([e2da9e6](https://github.com/0verL1nk/PaperSage/commit/e2da9e6108e1e2712d689e711dc32e095d792cf8))

## [1.11.0](https://github.com/0verL1nk/PaperSage/compare/v1.10.2...v1.11.0) (2026-08-23)


### Features

* **desktop:** resilient update checks with mirror fallback and visible failures ([#157](https://github.com/0verL1nk/PaperSage/issues/157)) ([63876a9](https://github.com/0verL1nk/PaperSage/commit/63876a9285baa987a9b19bf55611b3d458026f03))


### Performance Improvements

* **desktop:** defer lancedb, fastembed and langchain_community out of startup ([#158](https://github.com/0verL1nk/PaperSage/issues/158)) ([c552421](https://github.com/0verL1nk/PaperSage/commit/c5524219f144480bee9393e86dce881db90419d3))

## [1.10.2](https://github.com/0verL1nk/PaperSage/compare/v1.10.1...v1.10.2) (2026-08-23)


### Bug Fixes

* **web:** visible marker for unvalidated a2ui parts + turn-engine baseline reset ([#153](https://github.com/0verL1nk/PaperSage/issues/153)) ([0e38275](https://github.com/0verL1nk/PaperSage/commit/0e38275be3185a096710510317f00dc79fced8aa))

## [1.10.1](https://github.com/0verL1nk/PaperSage/compare/v1.10.0...v1.10.1) (2026-08-17)


### Bug Fixes

* **agent:** propagate mid-stream model failures instead of leaking them into answers ([#150](https://github.com/0verL1nk/PaperSage/issues/150)) ([0fbe799](https://github.com/0verL1nk/PaperSage/commit/0fbe799591fed8cb8d867e8a9508b5ca54b8142d))
* **web:** readable tool rows in the assistant timeline with expandable detail ([#151](https://github.com/0verL1nk/PaperSage/issues/151)) ([aaf9730](https://github.com/0verL1nk/PaperSage/commit/aaf973042e43179dbcd9a022321083bd35ec5605))

## [1.10.0](https://github.com/0verL1nk/PaperSage/compare/v1.9.0...v1.10.0) (2026-08-16)


### Features

* **agent:** Codex-parity subagent controls ([#147](https://github.com/0verL1nk/PaperSage/issues/147)) ([9dde03e](https://github.com/0verL1nk/PaperSage/commit/9dde03e93f113e8cee93e49195a5a2e0779572c1))

## [1.9.0](https://github.com/0verL1nk/PaperSage/compare/v1.8.4...v1.9.0) (2026-08-16)


### Features

* **agent:** builtin research subagents with agentic search and paper ingestion ([#144](https://github.com/0verL1nk/PaperSage/issues/144)) ([2703028](https://github.com/0verL1nk/PaperSage/commit/2703028fdbb9769af47e049b83e92aca2d0e3772))

## [1.8.4](https://github.com/0verL1nk/PaperSage/compare/v1.8.3...v1.8.4) (2026-08-16)


### Bug Fixes

* **agent:** salvage unterminated UI fragments and repair update_plan tool dispatch ([#142](https://github.com/0verL1nk/PaperSage/issues/142)) ([956ec13](https://github.com/0verL1nk/PaperSage/commit/956ec13ee92c17a7d5f5ffa01332787c99c32072))

## [1.8.3](https://github.com/0verL1nk/PaperSage/compare/v1.8.2...v1.8.3) (2026-08-16)


### Bug Fixes

* **desktop:** use the Electron net module for the GPU pack download ([#140](https://github.com/0verL1nk/PaperSage/issues/140)) ([17219e2](https://github.com/0verL1nk/PaperSage/commit/17219e2e186650d8b838005a46b9e476c7d06d5d))
* **web:** keep a2ui mindmaps inline after a turn completes ([#139](https://github.com/0verL1nk/PaperSage/issues/139)) ([abc0f10](https://github.com/0verL1nk/PaperSage/commit/abc0f103828dce74d47ec9528e2e3224c3a919f3))

## [1.8.2](https://github.com/0verL1nk/PaperSage/compare/v1.8.1...v1.8.2) (2026-08-16)


### Bug Fixes

* **desktop:** use the Electron net module for the GPU pack download ([#138](https://github.com/0verL1nk/PaperSage/issues/138)) ([a51ddd6](https://github.com/0verL1nk/PaperSage/commit/a51ddd6c2657783a18d0c7c99a27b83ced8c1de7))

## [1.8.1](https://github.com/0verL1nk/PaperSage/compare/v1.8.0...v1.8.1) (2026-08-16)


### Bug Fixes

* dangling a2ui placeholders, dead renderer logging, context card design ([#136](https://github.com/0verL1nk/PaperSage/issues/136)) ([9f346e1](https://github.com/0verL1nk/PaperSage/commit/9f346e1b0e7324e1255bf4d023840fc2b684fd41))

## [1.8.0](https://github.com/0verL1nk/PaperSage/compare/v1.7.0...v1.8.0) (2026-08-16)


### Features

* **web:** pop the mode switch from a compact active-mode indicator ([#134](https://github.com/0verL1nk/PaperSage/issues/134)) ([5da1301](https://github.com/0verL1nk/PaperSage/commit/5da1301ba375c40c86b1a9ed4cea1877fa91f267))


### Bug Fixes

* **desktop:** heal silent updates over legacy installs and exit cleanly ([#133](https://github.com/0verL1nk/PaperSage/issues/133)) ([19610bb](https://github.com/0verL1nk/PaperSage/commit/19610bb50620282a9e53aaf1a4d8fb7c5053f292))

## [1.7.0](https://github.com/0verL1nk/PaperSage/compare/v1.6.0...v1.7.0) (2026-08-16)


### Features

* **research:** model-generated follow-up suggestions per session ([#131](https://github.com/0verL1nk/PaperSage/issues/131)) ([476f5c3](https://github.com/0verL1nk/PaperSage/commit/476f5c397ec86165e5319b27d612618ad1109104))
* **web:** interleaved reasoning/tool timeline with inline citations ([#130](https://github.com/0verL1nk/PaperSage/issues/130)) ([64b5e23](https://github.com/0verL1nk/PaperSage/commit/64b5e2352800a5bc4b217e770b3643b8c20d5fda))


### Bug Fixes

* **desktop:** copy NVIDIA DLLs explicitly in GPU bundles ([#126](https://github.com/0verL1nk/PaperSage/issues/126)) ([5f9761b](https://github.com/0verL1nk/PaperSage/commit/5f9761b199d1f8024ba2f6925c0e9cb58e2ec69b))
* **desktop:** gather scattered cu13 CUDA DLLs into one loader directory ([#129](https://github.com/0verL1nk/PaperSage/issues/129)) ([eb9f9ab](https://github.com/0verL1nk/PaperSage/commit/eb9f9ab4041c894cac1c54d0aa3d0153e89b9684))
* **desktop:** retry the single-instance lock across update relaunches ([#127](https://github.com/0verL1nk/PaperSage/issues/127)) ([ce5e047](https://github.com/0verL1nk/PaperSage/commit/ce5e04743407e29f7d125006fd7e811636c38c4a))

## [1.6.0](https://github.com/0verL1nk/PaperSage/compare/v1.5.1...v1.6.0) (2026-08-15)


### Features

* **web:** interleaved message timeline with inline citations ([#121](https://github.com/0verL1nk/PaperSage/issues/121)) ([2327f5d](https://github.com/0verL1nk/PaperSage/commit/2327f5d7c8b04f767e37bb36f485fdee4e80a324))

## [1.5.1](https://github.com/0verL1nk/PaperSage/compare/v1.5.0...v1.5.1) (2026-08-15)


### Bug Fixes

* **desktop:** repair packaged ingestion, surface failures, install-dir model cache, GPU tier gating ([#117](https://github.com/0verL1nk/PaperSage/issues/117)) ([9c4af73](https://github.com/0verL1nk/PaperSage/commit/9c4af73b788fe988623cd092c4d97e2f83e3fefb))
* structured model JSON schema injection and UTF-8 backend stdio ([#115](https://github.com/0verL1nk/PaperSage/issues/115)) ([4c1deff](https://github.com/0verL1nk/PaperSage/commit/4c1deff9226822c0276c591514169354a63aa4be))

## [1.5.0](https://github.com/0verL1nk/PaperSage/compare/v1.4.4...v1.5.0) (2026-08-15)


### Features

* **skills:** user-overlay skill directory with priority resolution ([#111](https://github.com/0verL1nk/PaperSage/issues/111)) ([8c866f4](https://github.com/0verL1nk/PaperSage/commit/8c866f4b73746908eb5f9487ddb76eefafdbbdd0))
* **web:** visualize session context composition ([#113](https://github.com/0verL1nk/PaperSage/issues/113)) ([13dac58](https://github.com/0verL1nk/PaperSage/commit/13dac5892648e4699ca2da104702aa8ac7cdfd66))


### Bug Fixes

* **desktop:** bundle agent skills and tolerate reasoning-wrapped JSON ([#109](https://github.com/0verL1nk/PaperSage/issues/109)) ([f7ced2d](https://github.com/0verL1nk/PaperSage/commit/f7ced2dd7a9f14cd0ee920cf9a17ba0fe8ce989f))
* **desktop:** keep the launch window visible through backend boot ([#110](https://github.com/0verL1nk/PaperSage/issues/110)) ([12b408d](https://github.com/0verL1nk/PaperSage/commit/12b408d18227cb3c81fb16f1f35839c2a780fe31))
* **orm:** reconcile legacy event columns before run-item backfill ([#107](https://github.com/0verL1nk/PaperSage/issues/107)) ([fe3c16c](https://github.com/0verL1nk/PaperSage/commit/fe3c16caa0a4216922d3af6e2155e01c34f2e96c))
* **web:** compact response indicator and null terminal result tolerance ([#112](https://github.com/0verL1nk/PaperSage/issues/112)) ([a6a48f1](https://github.com/0verL1nk/PaperSage/commit/a6a48f1bd735138c130760e49c7ca6300a3c53ed))


### Documentation

* **design:** add OpenPencil design workflow and account-menu v1.4.4 artifacts ([#105](https://github.com/0verL1nk/PaperSage/issues/105)) ([dd3eb5b](https://github.com/0verL1nk/PaperSage/commit/dd3eb5b8962b8c59e48cefc4a4a910b7df81bd9d))

## [1.4.4](https://github.com/0verL1nk/PaperSage/compare/v1.4.3...v1.4.4) (2026-08-15)


### Bug Fixes

* **ci:** release train must squash-merge under squash-only settings ([#103](https://github.com/0verL1nk/PaperSage/issues/103)) ([ca012b8](https://github.com/0verL1nk/PaperSage/commit/ca012b88a247657cc7c51c6fb6d3533ed0130c63))
* **orm:** reconcile drifted agent_run_events schema ([78fc1ba](https://github.com/0verL1nk/PaperSage/commit/78fc1ba6a363bfcaa259bb5f03951e76d95f1db3))
* **orm:** reconcile drifted agent_run_events schema ([c1c7e46](https://github.com/0verL1nk/PaperSage/commit/c1c7e46e67ebd9306fa29a7081c1ee646d89e30b))
* **web:** merge duplicate settings entries into the account menu ([deb63d5](https://github.com/0verL1nk/PaperSage/commit/deb63d51b6e9f66576824e90603e370db86826b4))
* **web:** merge duplicate settings entries into the account menu ([d681296](https://github.com/0verL1nk/PaperSage/commit/d6812960cdd0676710261d87e34f967de9a5d8f7))


### Documentation

* add release and merge conventions ([#102](https://github.com/0verL1nk/PaperSage/issues/102)) ([62669af](https://github.com/0verL1nk/PaperSage/commit/62669af71e534674baea97904f65c6f29c3d4878))

## [1.4.3](https://github.com/0verL1nk/PaperSage/compare/v1.4.2...v1.4.3) (2026-08-15)


### Bug Fixes

* **desktop:** bundle alembic so packaged backend can start ([eece72b](https://github.com/0verL1nk/PaperSage/commit/eece72bd403bc80f674e8ed21b7be21a37f3f968))
* **desktop:** bundle alembic with the packaged backend ([f8cba69](https://github.com/0verL1nk/PaperSage/commit/f8cba697a9d64cb7845e76ba20d0929cbe917268))

## [1.4.2](https://github.com/0verL1nk/PaperSage/compare/v1.4.1...v1.4.2) (2026-08-14)


### Bug Fixes

* **desktop:** repair tray icon, add restart-to-update and visible version ([c4729a5](https://github.com/0verL1nk/PaperSage/commit/c4729a5fdce2aa22576a07d40963500bcbd200b2))
* **desktop:** repair tray icon, add restart-to-update and visible version ([f0d835b](https://github.com/0verL1nk/PaperSage/commit/f0d835b6ac5038b70dcf1298925ff53d775c3d1d))
* **web:** keep code-block async tokens fresh without render-phase refs ([1aafbb1](https://github.com/0verL1nk/PaperSage/commit/1aafbb1278a76aa176324d175da4682eb24d9b99))

## [1.4.1](https://github.com/0verL1nk/PaperSage/compare/v1.4.0...v1.4.1) (2026-08-14)


### Bug Fixes

* align streamdown shiki dependencies ([3be5a94](https://github.com/0verL1nk/PaperSage/commit/3be5a949e48f1d5c4da70dbd038f70497d4adda2))
* align streamdown shiki dependencies ([4fc18ce](https://github.com/0verL1nk/PaperSage/commit/4fc18cefdfc53347c121eff89a29d3c60dd59be0))

## [1.4.0](https://github.com/0verL1nk/PaperSage/compare/v1.3.6...v1.4.0) (2026-08-14)


### Features

* advance durable research runtime ([032aaa3](https://github.com/0verL1nk/PaperSage/commit/032aaa31dc58ba538c689bf4c72bf0b99aff51c2))
* advance durable research runtime ([7e46967](https://github.com/0verL1nk/PaperSage/commit/7e4696720fd7e78039126f0a35f3ffdba848b5d1))


### Bug Fixes

* keep memory changes out of runtime release ([c8db72e](https://github.com/0verL1nk/PaperSage/commit/c8db72eb0d6da3726d4debd1b2e8996ea6bccabc))
* keep workspace within debt baseline ([08eb4ad](https://github.com/0verL1nk/PaperSage/commit/08eb4adc8b75d8319c94f4397fbbe4fdc2a62eca))
* restore runtime migration compatibility ([7a0c757](https://github.com/0verL1nk/PaperSage/commit/7a0c7578f17b65fded3650700f904dffc81aa796))
* sort runtime migration imports ([cc58e80](https://github.com/0verL1nk/PaperSage/commit/cc58e80556959b38a21e23ed6d203071a83e7a36))

## [1.3.6](https://github.com/0verL1nk/PaperSage/compare/v1.3.5...v1.3.6) (2026-08-09)


### Bug Fixes

* upload large desktop updates to R2 ([036117a](https://github.com/0verL1nk/PaperSage/commit/036117a961eaa8a78230988ab83e63708690d951))
* upload large desktop updates to R2 ([e8a0e9a](https://github.com/0verL1nk/PaperSage/commit/e8a0e9a5209f24a224cf86092c8c1f8228aed62a))

## [1.3.5](https://github.com/0verL1nk/PaperSage/compare/v1.3.4...v1.3.5) (2026-08-09)


### Bug Fixes

* route desktop updates through R2 ([d00ae51](https://github.com/0verL1nk/PaperSage/commit/d00ae51c5d74dd158a5487a89825b9462c577230))
* route desktop updates through R2 ([89c4c69](https://github.com/0verL1nk/PaperSage/commit/89c4c699660d7bf4d1f11ebd88f7fdf8bf94dca7))

## [1.3.4](https://github.com/0verL1nk/PaperSage/compare/v1.3.3...v1.3.4) (2026-08-09)


### Bug Fixes

* release desktop OCR packaging metadata ([27e3111](https://github.com/0verL1nk/PaperSage/commit/27e31119ad452d9786415efeac347df9113ccba1))
* release desktop OCR packaging metadata ([6b22a40](https://github.com/0verL1nk/PaperSage/commit/6b22a4022e18d968584c6848892029412124da85))

## [1.3.3](https://github.com/0verL1nk/PaperSage/compare/v1.3.2...v1.3.3) (2026-08-09)


### Bug Fixes

* **desktop:** keep updater metadata aligned with installer ([#80](https://github.com/0verL1nk/PaperSage/issues/80)) ([406acf1](https://github.com/0verL1nk/PaperSage/commit/406acf1825e84c98d812f3f6f23e58c01cd823e2))
* grant release workflow PR permissions ([#79](https://github.com/0verL1nk/PaperSage/issues/79)) ([c6c2c97](https://github.com/0verL1nk/PaperSage/commit/c6c2c97a56da81df5c4ba3ec429b3c28e16ab889))

## [1.3.2](https://github.com/0verL1nk/PaperSage/compare/v1.3.1...v1.3.2) (2026-08-09)


### Bug Fixes

* dispatch canonical release inputs ([1635a9d](https://github.com/0verL1nk/PaperSage/commit/1635a9d98b1a5e44ec3071e75a73c56575652134))
* dispatch canonical release inputs ([5c23a2d](https://github.com/0verL1nk/PaperSage/commit/5c23a2d4ddcdc4a5bfd12c0e1cb3062d389f648d))
* improve desktop update and chat recovery ([#77](https://github.com/0verL1nk/PaperSage/issues/77)) ([fa23b76](https://github.com/0verL1nk/PaperSage/commit/fa23b7663325071fb013310e369aa1459f0fef62))

## [1.3.1](https://github.com/0verL1nk/PaperSage/compare/v1.3.0...v1.3.1) (2026-08-09)


### Bug Fixes

* grant release label permission ([3821db1](https://github.com/0verL1nk/PaperSage/commit/3821db10aa55c383240796989c5243c573ed170f))
* grant release label permission ([26a79f4](https://github.com/0verL1nk/PaperSage/commit/26a79f4da82f2fcd6fd280506b77011d88a5c0f3))
* preserve dependency lock versions ([bcd2095](https://github.com/0verL1nk/PaperSage/commit/bcd209540c2e1a24cb3184f4ee17e432e465fb9b))
* refresh project lock metadata ([8f6f7b5](https://github.com/0verL1nk/PaperSage/commit/8f6f7b55ac55c0f5b0f1fbabd1d0ef40fb777df9))

## [1.3.0](https://github.com/0verL1nk/PaperSage/compare/v1.2.0...v1.3.0) (2026-08-09)


### Features

* make research surfaces complement answers ([#67](https://github.com/0verL1nk/PaperSage/issues/67)) ([d8dc12d](https://github.com/0verL1nk/PaperSage/commit/d8dc12d1c1e22fd771aef5017d58930814f01557))
* replay research surfaces by identity ([#68](https://github.com/0verL1nk/PaperSage/issues/68)) ([bf304fc](https://github.com/0verL1nk/PaperSage/commit/bf304fc115523c4265c8a10dbcffc3cf27a83138))
* stream inline research surfaces ([b6447e7](https://github.com/0verL1nk/PaperSage/commit/b6447e77f8b88dcebecc7d76383e8442cafdef04))
* stream inline research surfaces ([4181f87](https://github.com/0verL1nk/PaperSage/commit/4181f87b233be83a139f1c449dce7760583f6e65))


### Bug Fixes

* allow manual PyPI publication ([62304a7](https://github.com/0verL1nk/PaperSage/commit/62304a7bddf3b219ae207e8c053ecfb2fe7fbe81))
* allow manual PyPI publication ([674367c](https://github.com/0verL1nk/PaperSage/commit/674367c0e108858747b201cdbe78e4b6b7d04cd6))
* close release train state ([0d13cee](https://github.com/0verL1nk/PaperSage/commit/0d13ceea284d1f53d30d12674703f3265400b0ee))
* close release train state ([31bb64b](https://github.com/0verL1nk/PaperSage/commit/31bb64bbf0e1f7fdbfe44fa052881de4f09b1e3c))
* discover unprefixed release tags ([14e33cc](https://github.com/0verL1nk/PaperSage/commit/14e33ccb18b17ee44ecc27d21b608eb93419fa89))
* discover unprefixed release tags ([e6166b7](https://github.com/0verL1nk/PaperSage/commit/e6166b7aed76b7001e1dc10c896dd5268d9170f7))
* synchronize runtime version during releases ([59eed79](https://github.com/0verL1nk/PaperSage/commit/59eed79cd711148c0ed946f39a9deeaaf8e44c79))


### Documentation

* clarify product workflow and local setup ([68811c7](https://github.com/0verL1nk/PaperSage/commit/68811c770eaa741934f8e00abd2902c286be2e6f))
* explain verifiable agent design ([572c983](https://github.com/0verL1nk/PaperSage/commit/572c983638e25742003629faad7d4de101d9c03b))
* specify natural A2UI output contract ([248ae70](https://github.com/0verL1nk/PaperSage/commit/248ae703ef5b2af9fdedda4fdfa7e4e7b1a3dc1e))

## [1.2.0](https://github.com/0verL1nk/PaperSage/compare/paper-sage-v1.1.13...paper-sage-v1.2.0) (2026-08-08)


### Features

* add async policy routing with configurable router model ([575d8f8](https://github.com/0verL1nk/PaperSage/commit/575d8f86e13e812aa52159fd9806d9569d528bdb))
* add emit_tool_load_event flag to control tool load event emissions ([83a745c](https://github.com/0verL1nk/PaperSage/commit/83a745cce4d8f6bd1d55f47d9c99dce8aaa2837c))
* add environment variables for Claude Code and PR Review workflows ([ca4f709](https://github.com/0verL1nk/PaperSage/commit/ca4f70957675aade15f977ceb48353e655767a26))
* add GitHub desktop updates ([a13e3b6](https://github.com/0verL1nk/PaperSage/commit/a13e3b6f05e5db917cf70c9025f58c1198d11ac1))
* add GitHub desktop updates ([795f580](https://github.com/0verL1nk/PaperSage/commit/795f580d5df7667c5e21ff8efa47b985f680be7e))
* add list_document tool to agent capabilities ([76e1d57](https://github.com/0verL1nk/PaperSage/commit/76e1d57d26c03fa54b3e03d1be8a7fd608f3bdac))
* add localized document evidence previews ([#61](https://github.com/0verL1nk/PaperSage/issues/61)) ([6b61cec](https://github.com/0verL1nk/PaperSage/commit/6b61cec6f532abc7bf8211b8287037f653298d54))
* add MinerU document parsing integration ([#5](https://github.com/0verL1nk/PaperSage/issues/5)) ([1b77b77](https://github.com/0verL1nk/PaperSage/commit/1b77b7732b010f880d9725d292c8d748db951e05))
* Add support for fetching PR head branch in cross-repository scenarios ([7472ad5](https://github.com/0verL1nk/PaperSage/commit/7472ad5c19a5151d7a55b5093231a85f08af0a7b))
* add tool search ([#15](https://github.com/0verL1nk/PaperSage/issues/15)) ([d4a6be1](https://github.com/0verL1nk/PaperSage/commit/d4a6be106a613691bb2454fb6b2e9fdc7640b857))
* agent centric orchestration ([#20](https://github.com/0verL1nk/PaperSage/issues/20)) ([a4a8416](https://github.com/0verL1nk/PaperSage/commit/a4a84167580ca91181297b482590bd81c95a6932))
* automate release version selection ([835afca](https://github.com/0verL1nk/PaperSage/commit/835afca2c11a3296821728c4f7a4584e9d2a40d6))
* complete checkpointer-based summarization migration ([#18](https://github.com/0verL1nk/PaperSage/issues/18)) ([673bba7](https://github.com/0verL1nk/PaperSage/commit/673bba7e55b4dc67fea3cf9cf0b5ff63ef364ee8))
* Enhance document handling and retrieval capabilities ([5491042](https://github.com/0verL1nk/PaperSage/commit/54910423f01cf5dac7760da8a0a60259d4a12060))
* Enhance project with new cleanup and revision policies ([51688f2](https://github.com/0verL1nk/PaperSage/commit/51688f21faf69ae65f7d22fae7e3d7df12eba3ef))
* enhance prompt handling and memory management in agent center ([924e012](https://github.com/0verL1nk/PaperSage/commit/924e012672674e14dd49c72b09fde4a5f86a3825))
* enhance team runtime output with native report rendering ([50de246](https://github.com/0verL1nk/PaperSage/commit/50de246f49a22b5c9f4ce59e4624c887e63ff22f))
* extract vector store builder into dedicated module ([#8](https://github.com/0verL1nk/PaperSage/issues/8)) ([b06edba](https://github.com/0verL1nk/PaperSage/commit/b06edba662817df85f3d0dd20430a1686c7ecc6a))
* implement lazy tool activation and usage with structured arguments ([b403576](https://github.com/0verL1nk/PaperSage/commit/b40357653d77b4e564da0eab6fdc0e1db744f528))
* implement session selector UID override logic and corresponding tests ([754e99c](https://github.com/0verL1nk/PaperSage/commit/754e99c7d435a84f4bdec09db573a5f3e216f304))
* Improve orchestration and planning modes with enhanced documentation and structured prompts ([872695e](https://github.com/0verL1nk/PaperSage/commit/872695e93c58bc6ca0b1fc2ebf9dee05b8a3eb10))
* log final answer in execute_turn_core and add corresponding test ([985a8c3](https://github.com/0verL1nk/PaperSage/commit/985a8c3c5be42cf3548603fab157734e50ea1a52))
* plan mode cleanup ([#12](https://github.com/0verL1nk/PaperSage/issues/12)) ([bef6dd8](https://github.com/0verL1nk/PaperSage/commit/bef6dd8b292372c01a73cc06950df643fa99253d))
* progressive tool disclosure and relevance-based document retrieval ([#9](https://github.com/0verL1nk/PaperSage/issues/9)) ([77a2e84](https://github.com/0verL1nk/PaperSage/commit/77a2e84659a89556249ac20e004db6500a6417a0))
* refactor agent center workspace and context observability ([0da84a1](https://github.com/0verL1nk/PaperSage/commit/0da84a1bdb8f009209c44a8d761ba187c6f10800))
* replace ANTHROPIC_AUTH_TOKEN with ANTHROPIC_API_KEY in workflow files ([a941db3](https://github.com/0verL1nk/PaperSage/commit/a941db3cac2e8c06778ab7733ffb8a718c77c92d))
* replace CLAUDE_CODE_OAUTH_TOKEN with ANTHROPIC_API_KEY in workflow files ([91b2fbb](https://github.com/0verL1nk/PaperSage/commit/91b2fbbf506af7e6e2a8e08586739f912b6537d6))
* team architecture upgrade ([#28](https://github.com/0verL1nk/PaperSage/issues/28)) ([ade3cf8](https://github.com/0verL1nk/PaperSage/commit/ade3cf8f828333520d50aa068934b452e74ffe07))
* tool manifest lazy schema ([#6](https://github.com/0verL1nk/PaperSage/issues/6)) ([63621d2](https://github.com/0verL1nk/PaperSage/commit/63621d21cab1c0d1f606b0a1f1e112a7a7e98404))
* 更新发布工作流，启用自动生成发布说明并简化版本处理 ([41f08b4](https://github.com/0verL1nk/PaperSage/commit/41f08b4ae65d6d57e9a98d6c4307f8ee5895640f))
* 更新版本到1.0.0，添加新功能和修复，更新项目链接 ([6e9a649](https://github.com/0verL1nk/PaperSage/commit/6e9a6492b1c110461ead2cc431a04219d9588cf3))


### Bug Fixes

* align PyPI package version ([6a47549](https://github.com/0verL1nk/PaperSage/commit/6a475494a6ec4ffff777caaeaf8621028889c542))
* align runtime package version ([a30b082](https://github.com/0verL1nk/PaperSage/commit/a30b082e343fde7a00ae132a5ba4262bdb2d4913))
* allow bot-triggered claude review workflow ([10d5e45](https://github.com/0verL1nk/PaperSage/commit/10d5e45f3dfd4568a864b204779c4202436e3bce))
* bind updater asset upload to repository ([4159f5a](https://github.com/0verL1nk/PaperSage/commit/4159f5ab0223d6e7671a18d87572e1a2f47ac44c))
* centralize GitHub release asset publishing ([b827ace](https://github.com/0verL1nk/PaperSage/commit/b827aced1c1029326b232eae5bb5935707a41fc7))
* consolidate release assets ([#45](https://github.com/0verL1nk/PaperSage/issues/45)) ([f5bf5bf](https://github.com/0verL1nk/PaperSage/commit/f5bf5bfc3d61efe1ca978989df7f10db8d23f98f))
* declare Electron package entry ([#41](https://github.com/0verL1nk/PaperSage/issues/41)) ([30bcd4b](https://github.com/0verL1nk/PaperSage/commit/30bcd4bc1c1a48fac5a305bb6a18477a8d119207))
* derive preview paths from owned documents ([#63](https://github.com/0verL1nk/PaperSage/issues/63)) ([1069635](https://github.com/0verL1nk/PaperSage/commit/10696352e5f39241f5e78a916c8da41da8bcb55f))
* enforce mindmap tag contract for parsing ([842a913](https://github.com/0verL1nk/PaperSage/commit/842a913c14428bb04123ba731035a077407e64fa))
* improve desktop update and installation flow ([22c8736](https://github.com/0verL1nk/PaperSage/commit/22c8736e33e39bc9b7e4a186eac36f8f78d7bcc3))
* improve desktop update experience ([17411d1](https://github.com/0verL1nk/PaperSage/commit/17411d1183b29ddae3ed1f94073b5cd4d9727d75))
* improve desktop update experience ([481f029](https://github.com/0verL1nk/PaperSage/commit/481f0298205835f82d88946b5c378c3b00440b7c))
* improve desktop update feedback ([863e80f](https://github.com/0verL1nk/PaperSage/commit/863e80fca5698fbbf5da2ffd16a71822a8321c7f))
* lint issue ([39f8620](https://github.com/0verL1nk/PaperSage/commit/39f8620e039c210d7fb7bb8f29f5712f9dc9b03f))
* normalize Windows install directory ([f1a0658](https://github.com/0verL1nk/PaperSage/commit/f1a06584f224deb1247224528e9da862fe9177c3))
* publish Linux deb package ([#44](https://github.com/0verL1nk/PaperSage/issues/44)) ([1551915](https://github.com/0verL1nk/PaperSage/commit/1551915ae41ce085b666fd3c4b47bb9c8619fed5))
* refine desktop workspace navigation ([f69ab3b](https://github.com/0verL1nk/PaperSage/commit/f69ab3b0aa18c8fde98961378bf0180a085eb9da))
* refine desktop workspace navigation ([65d5f21](https://github.com/0verL1nk/PaperSage/commit/65d5f21d83bcc333c411e84d407e4d6e694c6018))
* rename pages to ASCII, fix PyPI packaging and Windows CLI ([#7](https://github.com/0verL1nk/PaperSage/issues/7)) ([e46d0a1](https://github.com/0verL1nk/PaperSage/commit/e46d0a1629119b1287e930b70f3fd1948f28996e))
* repair cross-platform desktop release ([#43](https://github.com/0verL1nk/PaperSage/issues/43)) ([803b5ec](https://github.com/0verL1nk/PaperSage/commit/803b5ec7808f1402c357a8cadf5d37129d63c3d9))
* resolve type hints and improve code clarity across multiple files ([de78edd](https://github.com/0verL1nk/PaperSage/commit/de78eddfdcf780bfd35e5374fa6f51ee55f4cf86))
* show summary text and enhance mindmap fullscreen theme ([27c6ba3](https://github.com/0verL1nk/PaperSage/commit/27c6ba3ac8060e9f0d0e4775d09f5dda5f8b9e26))
* trigger Claude workflow on PR commits ([#33](https://github.com/0verL1nk/PaperSage/issues/33)) ([3231082](https://github.com/0verL1nk/PaperSage/commit/3231082b93a941438abdf66b1a3122e0e98e259a))
* **ui:** remove mindmap iframe scrollbar via adaptive height ([#4](https://github.com/0verL1nk/PaperSage/issues/4)) ([052dabf](https://github.com/0verL1nk/PaperSage/commit/052dabf064c94a2c51a506902db24dc2bace9072))
* update Chroma import and clarify agent settings for max staleness ([18d3527](https://github.com/0verL1nk/PaperSage/commit/18d3527cf29044a83fb8a9c1c16488fe2af8634e))
* Update orchestration middleware to include team mode in complexity result ([5b52aa6](https://github.com/0verL1nk/PaperSage/commit/5b52aa6f438db832e6e8e5fc54a9987b2cef98be))
* 一些修复 ([6652d82](https://github.com/0verL1nk/PaperSage/commit/6652d82d1a7a732f4a24ff4cc8e53d22fded9652))


### Documentation

* refresh brand and project overview ([7a3f46e](https://github.com/0verL1nk/PaperSage/commit/7a3f46e1913084a8b97df2c9dc89c2c77d2680e7))
* restore and update project README ([#42](https://github.com/0verL1nk/PaperSage/issues/42)) ([b6a2f17](https://github.com/0verL1nk/PaperSage/commit/b6a2f173fac58230b3745b9a20a654e16c1fed72))

## [Unreleased]

## [1.1.12] - 2026-08-08

### Fixed
- Regenerated Python dependency exports from the canonical uv lockfile and upgraded `idna` to 3.18, ensuring packaged and exported installations use the same patched dependency set.

## [1.1.11] - 2026-08-08

### Fixed
- Updated Python and frontend dependencies to address reported security advisories and regenerated Python requirements from the canonical uv lockfile.
- Made optional Claude-based PR review skip cleanly when repository credentials are unavailable, so Dependabot and forked PRs retain reliable quality gates.

## [1.1.10] - 2026-08-02

### Fixed
- Show clear, user-facing update download progress, completion, and failure feedback in the desktop app.
- Normalize drive-root installation choices to an absolute path such as `E:\PaperSage`.

## [1.1.9] - 2026-08-01

### Fixed
- Kept the runtime package version aligned with Python package metadata so tagged PyPI releases can publish successfully.

## [1.1.8] - 2026-08-01

### Fixed
- Kept desktop conversation scrolling inside the message region and show a definitive result after an update check.
- Place a Windows installation selected at a drive root in a `PaperSage` subfolder automatically.

### Changed
- Replaced bundled local RapidOCR/OpenCV with configured vision-model OCR for scanned PDFs, reducing the packaged backend by about 126 MB.

## [1.1.7] - 2026-08-01

### Added
- Added opt-in desktop update checks backed by PaperSage GitHub Releases, including download and restart confirmation.

### Changed
- Configured Electron Builder's GitHub update provider and macOS ZIP update artifact; DEB remains managed by the operating system package manager.

## [1.1.6] - 2026-08-01

### Fixed
- Kept the active project available when opening global settings, with a direct return path to the project workspace.
- Contained Electron scrolling in the workspace and applied the product scrollbar treatment to native overflow regions.

## [1.1.5] - 2026-08-01

### Fixed
- Consolidated release assets before publishing to avoid duplicate checksum and debug-file names across platforms.

## [1.1.4] - 2026-08-01

### Fixed
- Added the required maintainer metadata so Linux releases include both AppImage and DEB packages.

## [1.1.3] - 2026-08-01

### Fixed
- Added required Electron package metadata for Linux packages and skipped macOS signing cleanly when no signing certificate is configured.

## [1.1.2] - 2026-07-31

### Fixed
- Declared the Electron main-process entry point so native installers can pass Electron Builder's package integrity check.

## [1.1.1] - 2026-07-31

### Added
- Cross-platform Electron packaging, installer checksums, and GitHub build provenance for the workspace application.

### Changed
- Migrated the web workspace and release workflows from npm to pnpm.

## [1.1.0] - 2026-03-22

## [1.0.5] - 2026-03-16

## [1.0.4] - 2026-03-16

### Added
- **Tool Search Mechanism:** Replaced the hardcoded `activate_tool` mechanism with a dynamic `search_tools` capability based on the "Just-in-Time Retrieval" design.
- **Hybrid Tool Registry:** Introduced a new `ToolRegistry` (`agent/tools/registry.py`) that utilizes a 3-way hybrid retrieval engine (Regex intersection, BM25 sparse search, and FastEmbed dense vector search) to discover relevant tools and skills dynamically.
- **Skill Ecosystem Integration:** Updated `SkillLoader` to parse and index `keywords` from `SKILL.md` frontmatter, making dynamically loaded skills discoverable through the central tool search via the `use_skill` proxy.

### Changed

- Progressive Tool Disclosure Middleware now extracts discovered tools from the `search_tools` JSON response to un-defer their schema definitions instead of relying on explicit activation names.
- Added a shared runtime agent builder so paper agent, team member agent, and A2A agents use one assembly path for tools, middleware, and checkpointer setup.
- Team member agents now load the shared tool system with `start_plan` / `start_team` removed at load time, preventing nested mode spawning inside team execution.

## [1.0.3] - 2026-03-07

## [1.0.2] - 2026-03-07

## [1.0.1] - 2026-03-07

## [1.0.0] - 2026-03-07

### Added

- Async policy interception loop with lightweight router model, periodic context refresh, and in-loop mode switching.
- User-level policy router configuration in Settings Center (model / base URL / API key).
- User-level runtime tuning in Settings Center for async policy and memory controls:
  `AGENT_POLICY_ASYNC_*`, `RAG_INDEX_BATCH_SIZE`, `AGENT_DOCUMENT_TEXT_CACHE_MAX_CHARS`,
  `LOCAL_RAG_PROJECT_MAX_CHARS`, `LOCAL_RAG_PROJECT_MAX_CHUNKS`.
- `.env.example` with router, async-policy, RAG, memory, and tooling configuration examples.
- New unit tests covering logging configuration, user-setting migrations, tool-load tracing,
  runtime cache pruning, and provider thinking flags.

### Changed

- Project-level RAG vector index build path now uses batched insertion to reduce peak memory usage.
- Agent Center now applies per-user runtime tuning overrides at startup.
- High-frequency async interceptor decision logs are now `DEBUG` (keeps `INFO` focused on main-path decisions).
- Repository metadata and release links switched to `PaperSage`.

### Fixed

- Concurrent project retriever build race by introducing per-project build lock.
- OOM-prone document text cache growth by pruning with total-char budget.
- Thinking-related provider tests after settings schema expansion.

## [0.1.0] - 2026-03-06

### Added

- Multi-mode Agent workflow with automatic routing (ReAct / Plan-Act / Plan-Act-RePlan)
- Leader-centric Multi-Agent team orchestration with dependency-based task dispatch
- Local Hybrid RAG pipeline (Dense + BM25 + RRF + FlashRank Rerank)
- Structured evidence output with document-level traceability (chunk_id / page_no / offset)
- Long-term memory system with episodic / semantic / procedural classification and TTL
- Context governance with automatic compression and fact-anchor extraction
- 14+ built-in tools (search_document, read_file, write_file, bash, search_papers, search_web, etc.)
- 6 pluggable skills loaded from SKILL.md (summary, critical_reading, method_compare, translation, mindmap, agentic_search)
- Project-based workspace with document binding and scoped retrieval
- A2A SDK dual-stack compatibility (v0.3 + v1, streaming support)
- LLM provider adapter (OpenAI / DashScope with reasoning_effort / enable_thinking)
- Output sanitizer to filter CoT reasoning from user-facing answers
- Session metrics tracking (queries, latency, workflow counts, replan rounds)
- SQLite auto-migration for agent_outputs and memory tables
- Redis + RQ async task queue with synchronous fallback
- Docker deployment with docker-compose
- 53 unit tests + 6 integration tests + eval baselines
- CLI entry point: `paper-sage`

[Unreleased]: https://github.com/0verL1nk/PaperSage/compare/v1.1.5...HEAD
[1.1.5]: https://github.com/0verL1nk/PaperSage/compare/v1.1.4...v1.1.5
[1.1.4]: https://github.com/0verL1nk/PaperSage/compare/v1.1.3...v1.1.4
[1.1.3]: https://github.com/0verL1nk/PaperSage/compare/v1.1.2...v1.1.3
[1.1.2]: https://github.com/0verL1nk/PaperSage/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/0verL1nk/PaperSage/compare/v1.1.0...v1.1.1
[1.0.0]: https://github.com/0verL1nk/PaperSage/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/0verL1nk/PaperSage/releases/tag/v0.1.0
[1.0.1]: https://github.com/0verL1nk/PaperSage/compare/v1.0.0...v1.0.1
[1.0.2]: https://github.com/0verL1nk/PaperSage/compare/v1.0.1...v1.0.2
[1.0.3]: https://github.com/0verL1nk/PaperSage/compare/v1.0.2...v1.0.3
[1.0.4]: https://github.com/0verL1nk/PaperSage/compare/v1.0.3...v1.0.4
[1.0.5]: https://github.com/0verL1nk/PaperSage/compare/v1.0.4...v1.0.5
[1.1.0]: https://github.com/0verL1nk/PaperSage/compare/v1.0.5...v1.1.0
