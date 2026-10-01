// node .github/scripts/issue_triage.test.js   (exit 0 = pass)
const assert = require("assert");
const { triage, field } = require("./issue_triage.js");

const bug = (version, platform) =>
  `### Version\n\n${version}\n\n### How was the server started?\n\n./pion-server -w 1\n\n` +
  `### What you sent and what came back\n\n\`\`\`shell\nGET k\n\`\`\`\n\n### Platform\n\n${platform}\n`;

let n = 0;
const check = (name, fn) => { fn(); n++; console.log(`  PASS  ${name}`); };

check("every new issue gets needs-triage", () =>
  assert.deepStrictEqual(triage("free text, no form", null).labels, ["needs-triage"]));
check("mac platform", () =>
  assert.deepStrictEqual(triage(bug("pion-server 0.9.0+abc", "macOS 15 / M4"), "v0.9.0").labels,
                         ["needs-triage", "platform: mac"]));
check("linux platform", () =>
  assert.deepStrictEqual(triage(bug("0.9.0", "Ubuntu 24.04 / EPYC 7313P"), "v0.9.0").labels,
                         ["needs-triage", "platform: linux"]));
check("both mentioned -> both labels", () =>
  assert.strictEqual(triage(bug("0.9.0", "macOS host, Linux in Docker"), null).labels.length, 3));
check("current version -> no comment", () =>
  assert.strictEqual(triage(bug("pion-server 0.9.1+abc (2026-10-01)", "macOS"), "v0.9.1").comment, null));
check("older version -> one retry comment naming both versions", () => {
  const c = triage(bug("pion-server 0.9.0+abc", "macOS"), "v0.9.2").comment;
  assert.ok(c && c.includes("0.9.0") && c.includes("0.9.2"));
});
check("newer than latest (built from main) -> no comment", () =>
  assert.strictEqual(triage(bug("0.10.0+dev", "Linux"), "v0.9.2").comment, null));
check("no release yet -> no comment", () =>
  assert.strictEqual(triage(bug("0.9.0", "Linux"), null).comment, null));
check("empty optional field is not a value", () =>
  assert.strictEqual(field("### Platform\n\n_No response_\n", "Platform"), ""));
check("'m4' inside a word does not match mac", () =>
  assert.deepStrictEqual(triage(bug("0.9.0", "custom4 board"), null).labels, ["needs-triage"]));

console.log(`\n${n} passed`);
