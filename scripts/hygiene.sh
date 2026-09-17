#!/usr/bin/env bash
# Fails the build if the repo or its history carries anything that must not go public.
set -u
fail=0
say(){ echo "HYGIENE FAIL: $1"; fail=1; }
# 1. Banned identity strings in files (case-insensitive), excluding this script and git internals
# Excluded: aws-lambda/cat_code_maps.json -- the IEEE-CIS category map holds the email domain 'cableone.net', whose letters contain 'leon'
if git grep -Iil -E 'leon|alto|adair' -- . ':!scripts/hygiene.sh' ':!.git' ':!aws-lambda/cat_code_maps.json' | grep -q .; then
  git grep -Iil -E 'leon|alto|adair' -- . ':!scripts/hygiene.sh' ':!aws-lambda/cat_code_maps.json'; say "banned identity string in files"; fi
# 2. Same in full history (commit messages and authors), plus assistant attribution patterns.
#    Model ids such as claude-sonnet-5 are allowed in messages; attribution trailers are not.
if git log --all --format='%an %ae %s %b' | grep -iqE 'leon|alto|adair|co-authored-by: claude|claude-session|claude\.ai/code|noreply@anthropic'; then say "banned string or Claude attribution in git history"; fi
# 3. Claude attribution trailers or names in files
if git grep -Iil -E 'co-authored-by: claude|claude-session|generated with \[?claude|noreply@anthropic' -- . ':!scripts/hygiene.sh' | grep -q .; then say "Claude attribution in files"; fi
# 4. Real mailbox in author/committer. Author must be the noreply alias; committer may also be
#    GitHub's own merge identity (noreply@github.com), which every squash merge on GitHub sets.
if git log --all --format='%ae %ce' | grep -vqE '^45754668\+alphan-ml@users\.noreply\.github\.com (45754668\+alphan-ml@users\.noreply\.github\.com|noreply@github\.com)$'; then say "a commit uses a non-noreply email"; fi
# 5. Secret shapes
if git grep -IE 'AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|sk-ant-[A-Za-z0-9_-]{20,}|-----BEGIN (RSA|OPENSSH) PRIVATE KEY' -- . ':!scripts/hygiene.sh' | grep -q .; then say "secret-shaped string in files"; fi
# 6. .env tracked
if git ls-files | grep -qE '(^|/)\.env$'; then say ".env is tracked"; fi
[ $fail -eq 0 ] && echo "HYGIENE OK"
exit $fail
