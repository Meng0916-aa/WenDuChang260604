<#
.SYNOPSIS
    Safe GitHub sync for the WenDuChang project.

.DESCRIPTION
    Stages ONLY an explicit allowlist of source / config / doc / test files,
    then audits the staged set against a forbidden-pattern list (data, results,
    model weights, experiment data, figures, tables). If anything forbidden is
    staged, it ABORTS without committing. If nothing is staged, it prints
    "No source/document changes to commit." and exits.

    NEVER uses `git add .`. Default push target: origin main.

.PARAMETER Message
    Commit message. Required to commit; if omitted the script stages + audits
    and then stops (dry run), so you can review before committing.

.PARAMETER Push
    Push to origin main after a successful commit. Default: $true.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools/safe_git_sync.ps1 -Message "Update docs"

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools/safe_git_sync.ps1
    # dry run: stage + audit + show status, no commit
#>

[CmdletBinding()]
param(
    [string]$Message = "",
    [bool]$Push = $true
)

$ErrorActionPreference = "Stop"

# Move to the repository root (parent of this tools/ directory).
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# --- Allowlist: the ONLY paths that may be staged --------------------------
$AllowList = @(
    "README.md",
    "CLAUDE.md",
    "requirements.txt",
    "configs",
    "src",
    "scripts",
    "tests",
    "docs",
    "tools",
    ".gitignore",
    ".vscode/settings.json"
)

# --- Forbidden patterns: must NEVER be committed ---------------------------
# Matched against staged file paths (forward-slash, repo-relative).
$ForbiddenDirs = @("data/", "results/")
$ForbiddenExt  = @(
    ".xtherm", ".npy", ".npz", ".h5", ".hdf5",
    ".pt", ".pth", ".ckpt", ".onnx",
    ".png", ".jpg", ".jpeg", ".pdf",
    ".avi", ".mp4",
    ".csv", ".xlsx", ".xls"
)

function Fail($msg) {
    Write-Host "[safe_git_sync] ABORT: $msg" -ForegroundColor Red
    exit 1
}

# --- 0. Sanity: are we in a git repo? --------------------------------------
git rev-parse --is-inside-work-tree *> $null
if ($LASTEXITCODE -ne 0) { Fail "not a git repository: $RepoRoot" }

Write-Host "[safe_git_sync] repo: $RepoRoot"
Write-Host "[safe_git_sync] status before staging:"
git status --short

# --- 1. Stage only allowlisted paths that actually exist -------------------
$staged = $false
foreach ($path in $AllowList) {
    if (Test-Path $path) {
        git add -- $path
        $staged = $true
    }
}
if (-not $staged) {
    Fail "none of the allowlisted paths exist — refusing to continue."
}

# --- 2. Audit the staged set ------------------------------------------------
# Get staged paths (repo-relative, forward slashes from git).
$stagedFiles = git diff --cached --name-only
if (-not $stagedFiles) {
    Write-Host "No source/document changes to commit."
    exit 0
}

$violations = @()
foreach ($f in $stagedFiles) {
    $lower = $f.ToLower()
    foreach ($d in $ForbiddenDirs) {
        if ($lower.StartsWith($d)) { $violations += $f; break }
    }
    foreach ($e in $ForbiddenExt) {
        if ($lower.EndsWith($e)) { $violations += $f; break }
    }
}
$violations = $violations | Select-Object -Unique

if ($violations.Count -gt 0) {
    Write-Host "[safe_git_sync] Forbidden files were staged:" -ForegroundColor Red
    $violations | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
    Write-Host "[safe_git_sync] Unstaging everything and aborting (no commit)." -ForegroundColor Red
    git reset -q
    Fail "forbidden data/result files detected in staging area."
}

Write-Host "[safe_git_sync] staged files (audited OK):"
$stagedFiles | ForEach-Object { Write-Host "    $_" }

# --- 3. Commit (only if a message was supplied) ----------------------------
if ([string]::IsNullOrWhiteSpace($Message)) {
    Write-Host "[safe_git_sync] No -Message provided: dry run only (staged + audited, NOT committed)." -ForegroundColor Yellow
    Write-Host "[safe_git_sync] Re-run with -Message ""...""  to commit, e.g.:"
    Write-Host '    powershell -ExecutionPolicy Bypass -File tools/safe_git_sync.ps1 -Message "Update docs"'
    exit 0
}

git commit -m $Message
if ($LASTEXITCODE -ne 0) { Fail "git commit failed (check user.name/user.email config)." }

# --- 4. Push to origin main -------------------------------------------------
if ($Push) {
    git push origin main
    if ($LASTEXITCODE -ne 0) { Fail "git push failed." }
    Write-Host "[safe_git_sync] pushed to origin main." -ForegroundColor Green
} else {
    Write-Host "[safe_git_sync] committed locally; push skipped (-Push `$false)." -ForegroundColor Yellow
}

Write-Host "[safe_git_sync] done." -ForegroundColor Green
