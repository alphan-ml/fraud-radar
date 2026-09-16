# CLAUDE.md — rules for automated work in this repository

Read this first. Do not ask the owner to confirm anything written here.

## Identity
- Commit as `Alpha N <45754668+alphan-ml@users.noreply.github.com>`. Never a real mailbox address.
- No assistant attribution anywhere: no Co-Authored-By trailers, no session links, no "generated with" lines, no assistant names in files or commits.
- Do not write the owner's personal name anywhere in this repository. Say "the owner".

## Ship path
- Work on a branch. Open a pull request. Never push to `main` directly.
- Run `scripts/hygiene.sh` before pushing; it must print `HYGIENE OK`.
- Tests and lint must pass. Do not weaken a test to make it pass.
- Data never goes to git. Raw data is re-fetched or read from S3.

## Owner-only actions (do not attempt; stop and comment on the issue)
- Creating repositories, writing secrets, changing IAM or account settings, changing the live website.

## Night tasks
- Work arrives as an issue labeled `night` with: Goal / Repo / Done means / Do not / Stop if.
- Do exactly the Goal. Stop at "Stop if". Open a PR and post a status comment on the issue with what was done, what was not, and the test counts.
- If anything blocked the work, say so in the issue comment in one line: what blocked, root cause, suggested rule.

## Writing style
- Plain English, short sentences, bullets. Title Case headers.
