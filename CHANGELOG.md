# Changelog

All notable changes to Coherence Diagnostic will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] — 2026-10-25

### Changed
- **The narration is checked in code before it is sent.** Each draft is read for judging words and phrases (`JUDGING` in `backend/main.py`), including words the student used; a draft that uses one is rewritten, twice at most, and if every draft judges or comes back empty, the narration is built from the Stage 2 rules' own findings, with no model.
- **The narration prompt no longer asks the model to "name what's weak"**; it asks what is vague or missing, and why the tool cannot tell.
- **`/analyse/stream` sends the narration as one event**, after it is checked, instead of word by word.
- **Stage 2's overall label describes instead of grading:** `overall_status` is `GAPS_FOUND` or `NO_GAPS_FOUND` (it was `NEEDS_WORK` or `COHERENT`).

### Known limit
- The check reads only what is on its list; a judging phrase off it can still pass.

## [1.2.0] — 2026-02-16

### Added

- **Logout functionality** — header link to clear stored password and return to login
- **User status endpoint** (`/user/status`) — returns remaining analyses without consuming one
- **Usage counter on page load** — authenticated users see remaining analyses immediately

### Changed

- **OpenRouter API** — Stage 3 now uses OpenRouter (`anthropic/claude-haiku-4.5`) instead of direct Anthropic API
- **First-deploy model download** — DeBERTa model downloads via `entrypoint.sh` on first container start, persists in volume
- **Environment variable** — `ANTHROPIC_API_KEY` replaced with `OPENROUTER_API_KEY`
- **Dockerfile** — removed multi-stage build, model downloaded at runtime instead of bundled

### Removed

- `pre-commit-check.sh` — unused pre-commit hook script
- `STATUS.md` — replaced by `COMMIT-STATUS.md` (gitignored)

---

## [1.1.0] — 2026-02-14

### Added

- **User management system** with SQLite database
  - Each password = unique user with 10 lifetime analyses
  - Maximum 10 new users created per day
  - Remaining analyses counter shown to users
- **Admin panel** (`/admin`)
  - Password-protected admin access
  - Create new user passwords
  - View all users with usage stats
  - Dashboard with daily and total statistics
- **Three-mode analysis toggle**
  - Koher Architecture (3-stage pipeline)
  - Direct AI (single Claude call)
  - Side-by-Side comparison
- **Sample concepts button** with strong/weak/middle examples
- **History tabs** separating Koher and Direct AI analyses
- **60-second timeout** with refresh prompt
- **Auto-loading of `.env`** file via python-dotenv

### Changed

- Blues/Teals colour palette (distinct from Koher orange)
- API responses now include `remaining_analyses` field
- Button disables and shows "Limit Reached" when analyses exhausted

### Fixed

- Button hover colours now use teal (`#2d6980`) not orange

---

## [1.0.0] — 2026-02-13

### Added

- Initial release
- DeBERTa model for concept classification (98.38% accuracy)
- Stage 2 deterministic rules
- Stage 3 Haiku diagnosis with streaming
- Password-protected demo access
- Local history storage
- Docker and CapRover deployment configs

