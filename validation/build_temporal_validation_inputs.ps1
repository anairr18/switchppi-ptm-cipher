param(
    [string]$ProjectDir = (Split-Path -Parent $PSScriptRoot),
    [int]$BatchSize = 200,
    [int]$SleepMilliseconds = 350
)

$ErrorActionPreference = "Stop"

$benchmarkPath = Join-Path $ProjectDir "results\tables\benchmark_dataset.tsv"
$pmidOut = Join-Path $PSScriptRoot "pmid_date_manifest.tsv"
$keysOut = Join-Path $PSScriptRoot "benchmark_validation_keys.tsv"

if (-not (Test-Path -LiteralPath $benchmarkPath)) {
    throw "Benchmark table not found: $benchmarkPath"
}

function Get-PairKey {
    param(
        [string]$A,
        [string]$B
    )
    $ordered = @($A, $B) | Sort-Object
    return "$($ordered[0])||$($ordered[1])"
}

function Get-PublicationDate {
    param(
        [string]$SortPubDate
    )
    if ($SortPubDate -match '^(\d{4})/(\d{2})/(\d{2})') {
        return "$($Matches[1])-$($Matches[2])-$($Matches[3])"
    }
    return ""
}

$rows = Import-Csv -LiteralPath $benchmarkPath -Delimiter "`t"

$keyRows = foreach ($row in $rows) {
    $siteKey = "$($row.modified_uniprot)|$($row.ptm_type)|$($row.residue)|$($row.position)"
    $rowKey = "$($row.modified_uniprot)|$($row.partner_uniprot)|$($row.ptm_type)|$($row.residue)|$($row.position)"
    [pscustomobject]@{
        row_key = $rowKey
        site_key = $siteKey
        pair_key = Get-PairKey -A $row.modified_uniprot -B $row.partner_uniprot
        pmid = $row.pmid
        effect_label = $row.effect_label
    }
}

$keyRows |
    Sort-Object row_key, pmid, effect_label -Unique |
    Export-Csv -LiteralPath $keysOut -Delimiter "`t" -NoTypeInformation

$pmids = @(
    $rows |
        Where-Object { $_.pmid -match '^\d+$' } |
        Select-Object -ExpandProperty pmid -Unique |
        Sort-Object
)

$dateRows = New-Object System.Collections.Generic.List[object]

for ($i = 0; $i -lt $pmids.Count; $i += $BatchSize) {
    $last = [Math]::Min($i + $BatchSize - 1, $pmids.Count - 1)
    $chunk = @($pmids[$i..$last])
    $url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id=$($chunk -join ',')"
    $response = Invoke-RestMethod -Uri $url -TimeoutSec 60

    foreach ($uid in $response.result.uids) {
        $item = $response.result.$uid
        $dateRows.Add([pscustomobject]@{
            pmid = $uid
            pubdate = $item.pubdate
            sortpubdate = $item.sortpubdate
            publication_date = Get-PublicationDate -SortPubDate $item.sortpubdate
        }) | Out-Null
    }

    if ($i + $BatchSize -lt $pmids.Count) {
        Start-Sleep -Milliseconds $SleepMilliseconds
    }
}

$dateRows |
    Sort-Object publication_date, pmid |
    Export-Csv -LiteralPath $pmidOut -Delimiter "`t" -NoTypeInformation

$dated = @($dateRows | Where-Object { $_.publication_date })
$latest = $dated | Sort-Object publication_date -Descending | Select-Object -First 1
$earliest = $dated | Sort-Object publication_date | Select-Object -First 1

[pscustomobject]@{
    benchmark_rows = $rows.Count
    unique_pmids = $pmids.Count
    dated_pmids = $dated.Count
    earliest_publication_date = $earliest.publication_date
    latest_publication_date = $latest.publication_date
    prospective_cutoff_date = "2022-01-01"
    pmid_manifest = $pmidOut
    key_manifest = $keysOut
} | Format-List
