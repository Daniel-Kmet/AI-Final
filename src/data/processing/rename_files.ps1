# Function to pad numbers with leading zeros
function Format-Number {
    param (
        [int]$Number,
        [int]$Digits = 4
    )
    return $Number.ToString().PadLeft($Digits, '0')
}

# Rename fake images
$fakeFiles = Get-ChildItem -Path "data/raw/fake" -Filter "*.jpg"
$counter = 1
foreach ($file in $fakeFiles) {
    $newName = "fake_" + (Format-Number $counter) + ".jpg"
    Rename-Item -Path $file.FullName -NewName $newName
    $counter++
}

# Rename real images
$realFiles = Get-ChildItem -Path "data/raw/real" -Filter "*.png"
$counter = 1
foreach ($file in $realFiles) {
    $newName = "real_" + (Format-Number $counter) + ".png"
    Rename-Item -Path $file.FullName -NewName $newName
    $counter++
}

Write-Host "File renaming complete! Here's a summary:"
Write-Host "Fake images renamed:"
Get-ChildItem -Path "data/raw/fake" | Measure-Object | Select-Object -ExpandProperty Count
Write-Host "Real images renamed:"
Get-ChildItem -Path "data/raw/real" | Measure-Object | Select-Object -ExpandProperty Count 