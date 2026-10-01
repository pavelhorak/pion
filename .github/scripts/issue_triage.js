// Pure triage logic for .github/workflows/issue-triage.yml — no GitHub API
// calls here, so it can be tested with plain node:
//   node .github/scripts/issue_triage.test.js
//
// Issue forms render each field as "### <Label>\n\n<value>". We read the
// Platform and Version fields of the bug / performance templates.

function field(body, label) {
  const re = new RegExp(`^###\\s+${label}\\s*\\n+([\\s\\S]*?)(?=\\n###\\s|$)`, "mi");
  const m = (body || "").match(re);
  const v = m ? m[1].trim() : "";
  return v === "_No response_" ? "" : v;
}

function platformLabels(text) {
  const t = text.toLowerCase();
  const out = [];
  if (/\b(mac|macos|os x|darwin|apple|m[1-5]( pro| max| ultra)?)\b/.test(t)) out.push("platform: mac");
  if (/\b(linux|ubuntu|debian|fedora|rhel|centos|arch|alpine|epyc|xeon|graviton|wsl)\b/.test(t)) out.push("platform: linux");
  return out;
}

// "pion-server 0.9.0+abc1234 (...)" -> [0, 9, 0]; null if no x.y.z present.
function semver(text) {
  const m = (text || "").match(/(\d+)\.(\d+)\.(\d+)/);
  return m ? m.slice(1, 4).map(Number) : null;
}

function older(a, b) {
  for (let i = 0; i < 3; i++) if (a[i] !== b[i]) return a[i] < b[i];
  return false;
}

// Returns { labels: [...], comment: string|null }.
// latestTag is the newest release tag ("v0.9.1") or null when there is none.
function triage(body, latestTag) {
  const labels = ["needs-triage", ...platformLabels(field(body, "Platform"))];
  let comment = null;
  const reported = semver(field(body, "Version"));
  const latest = semver(latestTag || "");
  if (reported && latest && older(reported, latest)) {
    comment =
      `Thanks for the report. It names version ${reported.join(".")}, and the latest ` +
      `release is ${latest.join(".")}. Could you check whether it still happens on ` +
      `${latest.join(".")}? Several bugs have been fixed between releases — if it ` +
      `still reproduces, the same report is exactly what we need.`;
  }
  return { labels, comment };
}

module.exports = { triage, field, platformLabels, semver };
