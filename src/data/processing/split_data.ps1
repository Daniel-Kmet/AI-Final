# Function to get random subset of files
function Get-RandomSubset {
    param (
        [array]$Files,
        [int]$Count
    )
    return $Files | Get-Random -Count $Count
}

# Get all files from raw directories
$fakeFiles = Get-ChildItem -Path "data/raw/fake" -Filter "*.jpg"
$realFiles = Get-ChildItem -Path "data/raw/real" -Filter "*.png"

# Calculate split sizes
$fakeTotal = $fakeFiles.Count
$realTotal = $realFiles.Count

$fakeTrainCount = [math]::Floor($fakeTotal * 0.8)
$fakeValCount = [math]::Floor($fakeTotal * 0.1)
$fakeTestCount = $fakeTotal - $fakeTrainCount - $fakeValCount

$realTrainCount = [math]::Floor($realTotal * 0.8)
$realValCount = [math]::Floor($realTotal * 0.1)
$realTestCount = $realTotal - $realTrainCount - $realValCount

# Split fake images
$fakeTrain = Get-RandomSubset -Files $fakeFiles -Count $fakeTrainCount
$fakeRemaining = $fakeFiles | Where-Object { $fakeTrain -notcontains $_ }
$fakeVal = Get-RandomSubset -Files $fakeRemaining -Count $fakeValCount
$fakeTest = $fakeRemaining | Where-Object { $fakeVal -notcontains $_ }

# Split real images
$realTrain = Get-RandomSubset -Files $realFiles -Count $realTrainCount
$realRemaining = $realFiles | Where-Object { $realTrain -notcontains $_ }
$realVal = Get-RandomSubset -Files $realRemaining -Count $realValCount
$realTest = $realRemaining | Where-Object { $realVal -notcontains $_ }

# Move files to their respective directories
# Training set
foreach ($file in $fakeTrain) {
    Copy-Item -Path $file.FullName -Destination "data/processed/train/fake/"
}
foreach ($file in $realTrain) {
    Copy-Item -Path $file.FullName -Destination "data/processed/train/real/"
}

# Validation set
foreach ($file in $fakeVal) {
    Copy-Item -Path $file.FullName -Destination "data/processed/val/fake/"
}
foreach ($file in $realVal) {
    Copy-Item -Path $file.FullName -Destination "data/processed/val/real/"
}

# Test set
foreach ($file in $fakeTest) {
    Copy-Item -Path $file.FullName -Destination "data/processed/test/fake/"
}
foreach ($file in $realTest) {
    Copy-Item -Path $file.FullName -Destination "data/processed/test/real/"
}

# Print summary
Write-Host "Data split complete! Here's the distribution:"
Write-Host "`nFake Images:"
Write-Host "Training:   $($fakeTrain.Count) images"
Write-Host "Validation: $($fakeVal.Count) images"
Write-Host "Testing:    $($fakeTest.Count) images"
Write-Host "`nReal Images:"
Write-Host "Training:   $($realTrain.Count) images"
Write-Host "Validation: $($realVal.Count) images"
Write-Host "Testing:    $($realTest.Count) images" 