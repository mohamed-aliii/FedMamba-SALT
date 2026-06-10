> ⚠️ **HUMAN DECISION REQUIRED:**

1. **Token expiry policy:** How long should access tokens last? (e.g., 15 min access + 7-day refresh) — depends on hospital security policy.
2. **Hospital API key mechanism:** Hospitals authenticate their local training nodes via API key. Confirm: HMAC-SHA256 of key stored in DB, never the raw key.
3. **Password policy:** Minimum requirements (length, complexity) — set according to healthcare compliance requirements.

```

---

# claude response

Good question. Here's what I'd actually recommend for each, with the reasoning behind it:

---

## 1. Token Expiry Policy

**Recommended:** 15-minute access token + 7-day refresh token, with sliding refresh window.

**Why 15 minutes for access:**
The access token is a bearer credential — whoever holds it can act. If it leaks (XSS, log exposure, network interception), 15 minutes limits the damage window. The user never feels this because the refresh token silently renews it in the background.

**Why 7 days for refresh:**
Doctors in a clinical workflow open the app, do their work, close it. If the refresh token expires while they're on a night shift or weekend rotation and forces a re-login mid-case, it's a patient safety friction point. 7 days is a reasonable balance. The refresh token is httpOnly cookie only (never accessible to JavaScript), which neutralizes XSS against it.

**The sliding window detail:**
Every time the refresh token is used to get a new access token, issue a new refresh token and invalidate the old one. This means an actively-used session never expires. A session that goes completely idle for 7 days expires. This is better than a fixed expiry because it doesn't punish active users.

**One thing to add that the document missed:**
Implement **refresh token rotation** with a reuse detection flag. If an already-invalidated refresh token is presented (meaning someone stole and used it before the legitimate user), immediately invalidate the entire session family and alert the admin. This is the standard defense against refresh token theft.

---

## 2. Hospital API Key Mechanism

**Your current proposal (HMAC-SHA256) is correct but incomplete. Here's what to fully specify:**

**Key generation:**
Generate 32 random bytes using `secrets.token_bytes(32)`, then base64url-encode it. This gives a 43-character key that looks like `sk_hosp_Xk9mN2...`. The `sk_hosp_` prefix makes it visually identifiable in logs so it can be scrubbed.

**Storage:**
Store `SHA-256(key)` in the database, not HMAC. HMAC requires your `SECRET_KEY` as the HMAC key, which means if `SECRET_KEY` rotates, all stored API key hashes become invalid and every hospital needs a new key. Plain SHA-256 of a sufficiently random key (32 bytes of entropy) is cryptographically safe and rotation-independent.

**Transmission:**
The key is shown exactly once at creation time in the Admin UI. The hospital copies it and configures it in their training node environment. After that, it's transmitted in the `Authorization: Bearer sk_hosp_...` header on every API call from the hospital node.

**Comparison:**
Use `hmac.compare_digest(stored_hash, sha256(presented_key))` for constant-time comparison. Never use `==`.

**Key rotation:**
Allow issuing a new key while the old one stays valid for a 24-hour grace period. This lets the hospital update their config without a hard cutover. The database needs `api_key_hash`, `api_key_hash_previous`, and `key_rotated_at` fields.

**One thing the document missed entirely:**
The API key mechanism is only needed if hospital training nodes call back to the platform (e.g., to report metrics or receive model updates). Based on Phase 3's architecture where the platform orchestrates everything centrally, hospitals may not need API keys at all in v1. Clarify this with the research team before implementing it — don't build authentication for a communication channel that doesn't exist yet.

---

## 3. Password Policy

**Recommended concrete settings for healthcare:**

| Requirement | Value | Reasoning |
|---|---|---|
| Minimum length | 12 characters | NIST SP 800-63B and most healthcare frameworks |
| Maximum length | 128 characters | Prevent DoS via bcrypt on very long inputs |
| Complexity | None required | NIST explicitly removed complexity requirements in 2017 — they cause weak passwords like `P@ssw0rd1!` |
| Breach check | Yes — check against HaveIBeenPwned API | NIST recommends checking against known compromised passwords |
| History | Last 5 passwords rejected | Prevents immediate cycling back to old password |
| Account lockout | 5 failed attempts → 15-minute lockout | Softlock, not hard lock (hard locks create DoS risk) |
| Force reset | On first login (for accounts created by admin) | Doctors invited by hospital managers get a temp password |
| Session on password change | Invalidate all other sessions | Standard security practice |

**What to avoid:**
- Mandatory periodic expiry (e.g., every 90 days) — NIST removed this recommendation because it causes users to make predictable incremental changes (`Summer2024!` → `Fall2024!`). Only force a reset if there's evidence of compromise.
- Complexity rules that are checked on the client only — always re-validate server-side.

**HaveIBeenPwned integration:**
Use the k-anonymity API (send only the first 5 characters of the SHA-1 hash, never the full password or hash). This is a single HTTP call on registration/password change. If the password appears in a breach, reject it with a clear message: "This password has appeared in a data breach. Please choose a different one." No bcrypt cost, no privacy risk.

**bcrypt work factor:**
Set to 12 minimum. On modern hardware this takes ~250ms per hash, which is imperceptible to a user but makes brute-force 2048x harder than work factor 6 (common misconfigured default). Test this on your actual deployment hardware — on a weak server you may need to drop to 11.

---