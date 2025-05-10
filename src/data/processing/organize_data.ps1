# Create directory structure
$directories = @(
    "data/raw/fake",
    "data/raw/real",
    "data/processed/train/fake",
    "data/processed/train/real",
    "data/processed/val/fake",
    "data/processed/val/real",
    "data/processed/test/fake",
    "data/processed/test/real"
)

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Force -Path $dir
}

# Copy AI-generated images to raw/fake
$resolutionDirs = @("512", "768", "1024")
$genderDirs = @("man", "woman")

foreach ($resolution in $resolutionDirs) {
    foreach ($gender in $genderDirs) {
        $sourceDir = "Stable Diffusion Face Dataset/stable-diffusion-face-dataset/$resolution/$gender"
        if (Test-Path $sourceDir) {
            Get-ChildItem -Path $sourceDir -Filter "*.jpg" -Recurse | ForEach-Object {
                Copy-Item $_.FullName -Destination "data/raw/fake/$($_.Name)"
            }
        }
    }
}

# Copy real images to raw/real
$realSourceDir = "Flickr-Faces-HQ Dataset (FFHQ)"
if (Test-Path $realSourceDir) {
    Get-ChildItem -Path $realSourceDir -Filter "*.png" -Recurse | ForEach-Object {
        Copy-Item $_.FullName -Destination "data/raw/real/$($_.Name)"
    }
}

Write-Host "Data organization complete! Here's a summary of the files:"
Write-Host "Fake images (from Stable Diffusion):"
Get-ChildItem -Path "data/raw/fake" | Measure-Object | Select-Object -ExpandProperty Count
Write-Host "Real images (from FFHQ):"
Get-ChildItem -Path "data/raw/real" | Measure-Object | Select-Object -ExpandProperty Count 