# Git Terminal Workflow README

## Purpose

This guide explains how to add, remove, commit, and push files from the terminal in a branch-based Git repository, using the same kind of workflow used for the MGCLS-HI repository.

It is written for someone who is working locally first, then pushing changes to GitHub afterward.

---

## Core idea

Git tracks changes **per branch**.

That means:

- if you add a file on `main`, it is added on `main` only,
- if you delete a file on `templates`, it is deleted on `templates` only,
- if you edit a file on `static`, the edit belongs to `static` only,
- nothing affects another branch unless you switch there and make the change there too.

So the first rule is always:

**Make sure you are on the correct branch before changing anything.**

---

## Basic safe workflow

Whenever you want to make a change, the safest pattern is:

1. switch to the correct branch,
2. check what branch you are on,
3. inspect the current files if needed,
4. make your changes,
5. stage the changes,
6. commit the changes,
7. push the branch.

---

## 1. Check where you are

Before doing anything, run:

```bash
git branch
```

The branch with the `*` is your current branch.

You can also run:

```bash
git status
```

This tells you:

- which branch you are on,
- whether files were modified,
- whether files are staged,
- whether there is anything to commit.

---

## 2. Switch to the correct branch

To move to a branch:

```bash
git switch main
```

```bash
git switch templates
```

```bash
git switch static
```

If the branch does not exist locally yet but does exist remotely, create a local tracking branch:

```bash
git switch --track -c templates origin/templates
```

```bash
git switch --track -c static origin/static
```

---

## 3. See what files are currently in the branch

To list tracked files in the current branch:

```bash
git ls-tree -r --name-only HEAD
```

This is useful before deleting or reorganising files.

---

## 4. Add new files

### Add one file

```bash
git add README.md
```

### Add multiple files

```bash
git add app.py utils_UPDATED_single_metadata_v3_previews.py MGCLS_HI_15clusters.txt
```

### Add a whole folder

```bash
git add templates
```

```bash
git add static
```

### Add everything changed in the current branch

```bash
git add .
```

Use `git add .` carefully. It stages **all** changes in the current working tree under the current directory.

---

## 5. Commit changes

After staging files, create a commit:

```bash
git commit -m "Add Flask backend and metadata"
```

Examples:

```bash
git commit -m "Add Flask templates"
```

```bash
git commit -m "Add static assets"
```

```bash
git commit -m "Remove stray Python files from static assets"
```

```bash
git commit -m "Update README with local setup instructions"
```

### If Git says “nothing to commit”

That means one of these happened:

- you already committed the change,
- you did not actually modify the file,
- you forgot to stage the file,
- the file is unchanged.

Check with:

```bash
git status
```

---

## 6. Push your branch to GitHub

Once committed, push the current branch:

```bash
git push origin main
```

```bash
git push origin templates
```

```bash
git push origin static
```

If this is the first push for a local branch, use:

```bash
git push -u origin main
```

The `-u` sets the upstream so later you can often just use:

```bash
git push
```

---

## 7. Remove files from a branch

### Remove one file from Git and disk

```bash
git rm templates/download_select.html
```

Then commit and push:

```bash
git commit -m "Remove download_select template"
git push origin templates
```

### Remove multiple files from Git and disk

```bash
git rm static/footnote_logos/analyze_xray_levels.py static/footnote_logos/anotherTRY.py
```

Then:

```bash
git commit -m "Remove stray Python files from static assets"
git push origin static
```

### Remove a folder from Git and disk

```bash
git rm -r old_folder
```

Then commit and push.

---

## 8. Stop tracking a file but keep it on your computer

If you want GitHub to stop tracking a file, but you do **not** want to delete the local copy:

```bash
git rm --cached path/to/file
```

Then commit and push:

```bash
git commit -m "Stop tracking file"
git push origin <branch>
```

This removes the file from Git tracking, but leaves it on disk locally.

---

## 9. Rename or move files

### Rename a file

```bash
git mv old_name.txt new_name.txt
```

### Move a file into a folder

```bash
git mv download_select.html templates/download_select.html
```

Then commit and push:

```bash
git commit -m "Move download_select into templates folder"
git push origin templates
```

---

## 10. Create a new file from the terminal

Example:

```bash
printf "# MGCLS-HI\n\nMGCLS-HI repository.\n" > README.md
```

Then stage and commit:

```bash
git add README.md
git commit -m "Add README"
git push origin main
```

---

## 11. Copy files into the repo from somewhere else

Example:

```bash
cp /path/to/app.py .
cp /path/to/utils_UPDATED_single_metadata_v3_previews.py .
cp /path/to/MGCLS_HI_15clusters.txt .
```

Then:

```bash
git add app.py utils_UPDATED_single_metadata_v3_previews.py MGCLS_HI_15clusters.txt
git commit -m "Add backend files and metadata"
git push origin main
```

Example for templates:

```bash
cp /path/to/index.html templates/
cp /path/to/search.html templates/
cp /path/to/layout.html templates/
```

Then:

```bash
git add templates
git commit -m "Add templates"
git push origin templates
```

---

## 12. Inspect changes before committing

To see which files changed:

```bash
git status
```

To see exact line-by-line changes:

```bash
git diff
```

To see staged changes only:

```bash
git diff --staged
```

This is very useful before committing.

---

## 13. Undo mistakes before committing

### Unstage a file

If you added a file by mistake:

```bash
git restore --staged path/to/file
```

### Discard local edits to a file

Warning: this throws away local changes.

```bash
git restore path/to/file
```

### Discard all local unstaged changes

Warning: this throws away local changes in tracked files.

```bash
git restore .
```

---

## 14. Undo the last commit, but keep the files changed

If you committed too early and want to redo the commit message or contents:

```bash
git reset --soft HEAD~1
```

This removes the last commit but keeps the changes staged.

If you want to uncommit and unstage the changes:

```bash
git reset HEAD~1
```

Use these carefully.

---

## 15. Fetch remote changes

To update your knowledge of the remote repository:

```bash
git fetch origin
```

This does not merge anything yet. It just refreshes what Git knows about the remote.

To see local and remote branches:

```bash
git branch -a
```

---

## 16. Pull remote changes

To bring the current branch up to date:

```bash
git pull origin main
```

Or for a cleaner history:

```bash
git pull --rebase origin main
```

Use the branch name that matches your current branch.

Examples:

```bash
git pull --rebase origin templates
```

```bash
git pull --rebase origin static
```

---

## 17. If Git says push was rejected

A push may be rejected if the remote contains commits you do not have locally.

The usual fix is:

```bash
git fetch origin
git pull --rebase origin <branch>
git push origin <branch>
```

Example:

```bash
git fetch origin
git pull --rebase origin main
git push origin main
```

### If a rebase causes a conflict

Git will stop and ask you to resolve the conflict manually.

After fixing the file:

```bash
git add conflicted_file
git rebase --continue
```

If you want to abandon the rebase:

```bash
git rebase --abort
```

### Force push, only if you truly mean to overwrite remote history

```bash
git push --force-with-lease origin main
```

Use `--force-with-lease` very carefully. It is safer than `--force`, but it still rewrites remote history.

---

## 18. Create a backup branch before risky changes

Before deleting or reorganising many files, create a backup branch:

```bash
git switch templates
git branch backup-templates-before-cleanup
```

Example:

```bash
git switch static
git branch backup-static-before-cleanup
```

This gives you a recovery point.

---

## 19. Common branch-specific workflows

### A. Add backend files to `main`

```bash
git switch main
cp /path/to/app.py .
cp /path/to/utils_UPDATED_single_metadata_v3_previews.py .
cp /path/to/MGCLS_HI_15clusters.txt .
git add app.py utils_UPDATED_single_metadata_v3_previews.py MGCLS_HI_15clusters.txt
git commit -m "Add backend and metadata"
git push origin main
```

### B. Add templates to `templates`

```bash
git switch templates
mkdir -p templates
cp /path/to/index.html templates/
cp /path/to/search.html templates/
cp /path/to/layout.html templates/
git add templates
git commit -m "Add templates"
git push origin templates
```

### C. Add assets to `static`

```bash
git switch static
mkdir -p static
cp -r /path/to/static/* static/
git add static
git commit -m "Add static assets"
git push origin static
```

### D. Remove files from `static`

```bash
git switch static
git rm static/footnote_logos/analyze_xray_levels.py static/footnote_logos/anotherTRY.py
git commit -m "Remove stray Python files from static assets"
git push origin static
```

### E. Remove a template from `templates`

```bash
git switch templates
git rm templates/download_select.html
git commit -m "Remove download_select template"
git push origin templates
```

---

## 20. Good commit message habits

A commit message should say what changed.

Good examples:

- `Add Flask backend and metadata`
- `Add templates for search and cluster detail pages`
- `Clean static branch and keep only static assets`
- `Remove unused files from static branch`
- `Update README with local setup and deployment notes`

Avoid vague messages like:

- `update`
- `changes`
- `fix stuff`

---

## 21. Useful command summary

### Check status

```bash
git status
```

### See current branch

```bash
git branch
```

### Switch branch

```bash
git switch <branch>
```

### List files in current branch

```bash
git ls-tree -r --name-only HEAD
```

### Add file(s)

```bash
git add <file>
```

### Commit

```bash
git commit -m "Message"
```

### Push

```bash
git push origin <branch>
```

### Remove file(s)

```bash
git rm <file>
```

### Remove folder

```bash
git rm -r <folder>
```

### Stop tracking file but keep local copy

```bash
git rm --cached <file>
```

### Rename or move file

```bash
git mv old_path new_path
```

### Show unstaged diff

```bash
git diff
```

### Show staged diff

```bash
git diff --staged
```

### Fetch remote state

```bash
git fetch origin
```

### Pull with rebase

```bash
git pull --rebase origin <branch>
```

### Abort a rebase

```bash
git rebase --abort
```

---

## 22. Final safety rules

Before changing anything:

- confirm the branch,
- check `git status`,
- check the file path carefully,
- do not use `git add .` blindly,
- do not use force push unless you understand why,
- make a backup branch before major cleanup,
- push only after local verification.

For a multi-branch project like MGCLS-HI, always think in this order:

1. **Which branch am I on?**
2. **Which file am I changing?**
3. **Should this file live on this branch?**
4. **Have I checked the result before pushing?**

---

## Minimal everyday workflow

If you only want the shortest repeatable pattern:

```bash
git switch <branch>
git status
# make your edits
git add <files>
git commit -m "Describe the change"
git push origin <branch>
```

That is the basic everyday Git terminal workflow.

