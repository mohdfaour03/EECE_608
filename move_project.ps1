# move_project.ps1 - relocate EECE_608 out of OneDrive (non-destructive copy + config repoint)
# Written by Fable, 2026-07-16, at Mohamad's direction. Safe to re-run.
# Copies the tree to %USERPROFILE%\EECE_608, repoints agent configs, freezes the OneDrive copy.
# NEVER deletes anything. Full log: MIGRATION_REPORT.txt in both locations.

$ErrorActionPreference = 'Continue'
$src = Join-Path $env:USERPROFILE 'OneDrive - American University of Beirut\Documents\EECE_608'
$dst = Join-Path $env:USERPROFILE 'EECE_608'
$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$report = New-Object System.Collections.Generic.List[string]
$report.Add("EECE_608 relocation report - $stamp")
$report.Add("Source (frozen after this): $src")
$report.Add("Destination (new canonical): $dst")
$report.Add('')

if (-not (Test-Path $src)) {
    $report.Add("FATAL: source not found: $src")
} else {
    $log = Join-Path $env:TEMP 'eece608_robocopy.log'
    robocopy $src $dst /E /COPY:DAT /DCOPY:DAT /R:2 /W:2 /XD '$RECYCLE.BIN' /NP /NFL /NDL /LOG:$log | Out-Null
    $rc = $LASTEXITCODE
    $report.Add("robocopy exit code: $rc (0-3 clean; 4-7 minor mismatches; >=8 FAILURES - see $log)")

    if ($rc -lt 8) {
        $srcEsc = $src.Replace('\', '\\'); $dstEsc = $dst.Replace('\', '\\')

        # Repoint configs & docs that reference the old path (handles both plain and JSON/TOML-escaped forms)
        $targets = @(
            (Join-Path $dst '.mcp.json'),
            (Join-Path $env:USERPROFILE '.codex\config.toml'),
            (Join-Path $dst 'agent_bus\SETUP.md')
        )
        foreach ($f in $targets) {
            if (Test-Path $f) {
                Copy-Item $f "$f.pre_migration.bak" -Force
                $t = [IO.File]::ReadAllText($f)
                $n = $t.Replace($srcEsc, $dstEsc).Replace($src, $dst)
                if ($n -ne $t) {
                    [IO.File]::WriteAllText($f, $n)
                    $report.Add("repointed: $f (backup: $f.pre_migration.bak)")
                } else {
                    $report.Add("no old-path references found in: $f")
                }
            } else {
                $report.Add("MISSING (repoint manually if needed): $f")
            }
        }

        # Sanity checks on the copy
        foreach ($k in @('.git\HEAD', 'CLAUDE.md', 'agent_bus\server.py', 'Fable\DECISIONS.md', 'src', 'GPT_SOL')) {
            if (Test-Path (Join-Path $dst $k)) { $report.Add("OK in copy: $k") }
            else { $report.Add("MISSING IN COPY: $k") }
        }

        # Informational: remaining old-path mentions in text files (fix at leisure)
        $exts = '.md', '.json', '.toml', '.yml', '.yaml', '.py', '.cfg', '.txt'
        $hits = Get-ChildItem $dst -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $exts -contains $_.Extension -and $_.FullName -notmatch '\\\.git\\' -and $_.Name -notmatch 'pre_migration' } |
            Select-String -SimpleMatch 'OneDrive - American University of Beirut' -List -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty Path -Unique
        $report.Add('')
        $report.Add('Files still mentioning the OneDrive path (docs/history - informational only):')
        if ($hits) { $hits | ForEach-Object { $report.Add("  $_") } } else { $report.Add('  none') }

        # Freeze marker in the old location
        $marker = "This project RELOCATED to $dst on $stamp.`r`n" +
                  "The OneDrive copy is a FROZEN BACKUP - DO NOT EDIT. Delete after verifying the new location.`r`n" +
                  "Both agents: restart pointing at the new path. bus.db stays at %LOCALAPPDATA%\agent_bus\bus.db."
        [IO.File]::WriteAllText((Join-Path $src '_RELOCATED_DO_NOT_EDIT.txt'), $marker)
        $report.Add('')
        $report.Add('Freeze marker written to source. Next steps:')
        $report.Add('  1. Mohamad: restart Cowork (Fable) selecting the NEW folder; restart Sol (Codex) in the NEW folder.')
        $report.Add('  2. Verify, then delete the OneDrive copy whenever convenient.')
    }
}

$txt = ($report -join "`r`n") + "`r`n"
try { [IO.File]::WriteAllText((Join-Path $dst 'MIGRATION_REPORT.txt'), $txt) } catch {}
try { [IO.File]::WriteAllText((Join-Path $src 'MIGRATION_REPORT.txt'), $txt) } catch {}
Write-Host $txt
