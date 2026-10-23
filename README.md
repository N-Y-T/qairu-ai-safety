# QAIRU AI Safety Midterm

Interactive prompt-injection and RAG evaluation lab by Nursultan Yessenbek.

**[Live website](https://qairu-injection-lab.brown-sky-2597.chatgpt.site/)** · **[RAG chat](https://qairu-injection-lab.brown-sky-2597.chatgpt.site/rag)** · **[Test journal](https://qairu-injection-lab.brown-sky-2597.chatgpt.site/journal)**

The hosted website is private and requires the owner's authorized access. This repository contains its source, synthetic test evidence, reproducible evaluation code and the English midterm presentation.

## Presentation and documents

- [English PowerPoint — 8 slides](presentation/Midterm_Slides_EN.pptx)
- [English presentation — PDF](presentation/Midterm_Presentation_EN.pdf)
- [English midterm report](docs/Midterm_Report_EN.pdf)
- [Russian defense notes](docs/Defense_Notes_RU.pdf)

The presentation/report cover the original **21 October 2026 access-control experiment** (288 measured calls). The separate **22 October 2026 RAG extension** adds seven tools and 20 real model responses. The older presentation has been preserved as supplied; do not attribute the RAG results to those 288 calls.

## What is included

| Path | Contents |
| --- | --- |
| `app/page.tsx`, `app/api/lab/route.ts` | Interactive access-control lab and server decision policy |
| `app/rag/`, `lib/rag.ts` | Synthetic RAG chat, retrieval and context-injection cases |
| `app/journal/` | Published evidence and browser-local test history |
| `public/model-worker.js` | Real browser inference with Qwen2.5-0.5B-Instruct |
| `public/evaluations/` | Raw requests, contexts, responses, tool reports and checksums |
| `evaluation/` | Reproduction scripts and pinned evaluation dependencies |
| `colab/`, `public/QAIRU_Live_Web_Chat.ipynb` | Optional Colab inference bridge and launcher |
| `.devcontainer/` | Isolated Node 24 development environment for GitHub Codespaces |

## Run the website

Use Node 24 and pnpm 11.25.0:

```bash
npm install --global pnpm@11.25.0
pnpm install --frozen-lockfile
pnpm dev
```

Open `http://localhost:5173`. In Codespaces, forward port 5173 and open its browser preview. The container installs dependencies; run `pnpm dev` in its terminal. Creating this repository does not automatically start a billable Codespace.

```bash
pnpm exec tsc --noEmit
node scripts/check-lab.mjs
pnpm build
```

The project uses Vinext/React with a Cloudflare Worker API. GitHub Pages alone cannot host its server routes. The existing live URL remains the running deployment; uploading this repository does not change the hosted Site's source binding or audience.

The browser downloads approximately 0.5–0.8 GB of model files on first use. WebGPU or CPU/WASM runs inference on the device. Model download and generation were not successfully verified in the connected browser during the RAG evaluation: the download encountered a network error.

## Recorded RAG evaluation

Twenty actual **local ONNX CPU/q4** responses were generated using the website's shared retrieval/prompt source: 10 cases × baseline/protected policies. Browser quantization and hardware may produce different responses. No external corporate RAG was tested.

| Tool | Operation and observed result |
| --- | --- |
| PyRIT | Base64 and ROT13 converters; 2 payloads, 4 associated model responses |
| Promptfoo | Replay assertions: 19/20; no new model calls |
| IBM ART | HopSkipJump on continuous retrieval features: 4/4 closest-document changes, L2 approximately 1.29 |
| Giskard | Custom replay checks: 18/20; two VPN responses included unsupported UI steps |
| Inspect AI | Custom replay solver/scorer: accuracy 0.95 on the same 20 outputs |
| SHAP | Exact explanations of retrieval scores; additivity error 0 on four examples |
| Ragas | ID-based retrieval precision/recall: 1.0 on nine eligible cases |

No attack marker appeared in the 12 attack responses. This small set does not establish general robustness. Promptfoo/Inspect's single failure is a literal UNKNOWN check despite a semantically valid missing-information response. Giskard's VPN rule is exploratory and was written after inspecting the outputs. ART and SHAP evaluate retrieval features, not LLM jailbreak resistance. Ragas faithfulness was not measured. garak was not rerun for this extension.

See [evaluation instructions](evaluation/README.md), [complete report](public/evaluations/report.json), [model responses](public/evaluations/live-results.json) and [checksums](public/evaluations/sha256.json). The same saved responses are reused by evaluators; their checks are not independent attacks.

## Test history

The published 20-record report is committed to this repository. New access-control and RAG chat records are saved in the current browser's local storage and can be exported as JSON. Clearing browser data removes those local records. They are not automatically uploaded to GitHub.

All demonstrated credentials and business documents are synthetic. The access-control example intentionally uses the fictional code ORCHID-4827; it is not production authentication.

## Source provenance

Exported from the published QAIRU Site source commit `9ebc2b6b9a419d6842de530fd008aea953186774`. Presentation documents were downloaded from the user's saved midterm materials and preserved byte-for-byte. No model weights, dependency folders, authentication credentials or local runtime caches are part of the export.
