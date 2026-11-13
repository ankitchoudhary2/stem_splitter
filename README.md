# Stem Splitter (browser version)
Static site: runs the HTDemucs AI model inside the visitor's browser (WebGPU, falling back to WebAssembly). No server, no uploads.

## Deploy on Vercel
1. Create a GitHub repo and upload everything in this folder (all files, no folders).
2. vercel.com -> Add New -> Project -> import the repo. Framework preset: **Other**. No build command, no output directory. Deploy.
3. `vercel.json` sets the COOP/COEP headers that allow multi-threaded WebAssembly. Keep it.

## Test locally first (needs those headers too)
Any server that sends `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`,
or simply: `npx vercel dev` inside this folder.

## Notes
- Model (~170 MB) is fetched from Hugging Face on first use and cached in the browser. Runtime: onnxruntime-web 1.30.0 via jsDelivr.
- Stems: Vocal/Drum/Bass/Other from HTDemucs (4-stem). Music = drums+bass+other. Ambience = diffuse residual of Other (approximation).
- Licenses: Demucs (MIT), demucs-web (MIT, in vendor/), onnxruntime-web (MIT).
