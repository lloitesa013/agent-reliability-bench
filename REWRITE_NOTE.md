# Rewrite note (2026-07-15)

On 2026-07-15 the full history of this repository was rewritten to normalize commit
author/committer metadata to the repository's account identity
(`lloitesa013 <lloitesa013@users.noreply.github.com>`). No file content was changed
by the rewrite.

## Effect on the sealed baseline (`SEAL_MANIFEST_baseline.txt`)

- The manifest records the sealed baseline commit as `ba1e71f53cb890c634219ab9dcd6022ab6f1fd79`
  — a pre-rewrite SHA that no longer exists in this history.
- The corresponding post-rewrite commit is `e34a284805b4947fccd519618e1175c1713022be`.
- Proof the sealed content is unchanged: both commits share the identical root tree
  `1502ab50bde856297d5f8095c93c5389c9f6a244` (commit metadata is not part of the tree).
  Verify with `git rev-parse e34a2848^{tree}`.

All other commit SHAs changed likewise; file trees are unchanged throughout.
