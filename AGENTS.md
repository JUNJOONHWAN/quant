# AGENTS Guidelines

- Do not ship, reference, or silently fall back to any sample or placeholder dataset in production builds or GitHub Pages deployments.
- The frontend must load only `static_site/data/precomputed.json` (or the exact file indicated by `data/version.json`).
- If the dataset is missing or invalid, surface a clear error to the user; do not auto-load `precomputed-sample.json` or any other fallback.
- Keep Classic and Enhanced regime logic side-by-side, but the default behavior must remain Classic unless explicitly toggled by the user.
- Do not ask the user to refresh/reload or clear caches; handle issues through code or configuration changes instead.
